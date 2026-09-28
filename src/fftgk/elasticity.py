"""
Elasticidad periodica por FFT, sobre el MISMO simbolo de gradiente discreto que
la homogeneizacion de permeabilidad y que el tensor de dano.

Por que aqui y no en FEniCSx
----------------------------
La afirmacion central del proyecto descansa en dos medidas: D_ij con error
menor al 1%, y la conectividad de la red de grietas. Si el campo de fase vive
en una malla de elementos finitos y hay que proyectarlo a la grilla FFT en cada
paso de carga, la interpolacion mete error justo en esas dos cantidades -- y un
error que depende de la malla y de la orientacion, que es exactamente el tipo
de sesgo que costo trabajo eliminar del estimador de dano (doc 04 seccion 2).

Resolviendo todo sobre la misma grilla no hay proyeccion, no hay interpolacion,
y las condiciones periodicas -- que es lo que exige la homogeneizacion -- salen
gratis. FEniCSx queda para VERIFICACION independiente en unos pocos puntos, que
es donde mas vale: blinda contra la objecion "su campo de fase es casero".

Formulacion
-----------
Deformacion total  eps = E + eps~,  con eps~ compatible y de media nula.
Compatible significa eps~ = sym(grad u) para algun u periodico; en Fourier,

    eps^(xi) = sym( D(xi) (x) a )   para algun a in C^d

El proyector ortogonal sobre ese subespacio se construye del mismo simbolo D:

    M : a -> sym(D (x) a)
    M^H: eps -> v,  v_j = sum_i conj(D_i) eps_ij
    M^H M = (1/2)( |D|^2 I + D D^H )
    (M^H M)^{-1} = (1/|D|^2) ( 2 I - D D^H / |D|^2 )
    Proj = M (M^H M)^{-1} M^H

(M^H M tiene a D como autovector con autovalor |D|^2, y todo lo perpendicular
a D con autovalor |D|^2/2; de ahi la inversa.)

Se resuelve  Proj[ C(x) : (E + eps~) ] = 0  con gradiente conjugado, igual que
en el problema de Darcy. Mismo esquema "rotated" por defecto, misma robustez a
alto contraste -- lo que importa porque una grieta es una zona de rigidez casi
nula, o sea contraste enorme.

Material
--------
Solo isotropo local, descrito por los campos lambda(x) y mu(x):

    C : eps = lambda tr(eps) I + 2 mu eps

Es todo lo que hace falta: la matriz es isotropa y el campo de fase degrada con
un escalar g(phi). Evita almacenar un tensor de cuarto orden por voxel.
"""

from __future__ import annotations

import numpy as np

from .fft import fftn as _fftn, ifftn as _ifftn

from .homogenize import FourierProjector, _cg

__all__ = [
    "ElasticProjector",
    "apply_C",
    "solve_elasticity",
    "effective_stiffness",
    "lame_from_E_nu",
]


def lame_from_E_nu(E, nu, plane_strain=True):
    """Constantes de Lame desde modulo de Young y Poisson.

    En 2D se asume deformacion plana (plane_strain=True), que es la hipotesis
    adecuada para una celda representativa de roca: el espesor fuera del plano
    es grande comparado con la microestructura.
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    if not plane_strain:
        lam = 2 * lam * mu / (lam + 2 * mu)   # tension plana
    return lam, mu


def apply_C(lam, mu, eps):
    """C(x) : eps(x).  lam, mu: (*N);  eps: (d,d,*N) -> (d,d,*N)."""
    tr = np.einsum("ii...->...", eps)
    d = eps.shape[0]
    eye = np.eye(d).reshape((d, d) + (1,) * (eps.ndim - 2))
    return lam[None, None, ...] * tr[None, None, ...] * eye + 2.0 * mu[None, None, ...] * eps


class ElasticProjector:
    """Proyector ortogonal sobre deformaciones compatibles de media nula."""

    def __init__(self, N, L=None, scheme="rotated", base=None):
        self.base = base if base is not None else FourierProjector(N, L, scheme)
        self.N = self.base.N
        self.d = self.base.d
        D = self.base.D
        self.D = D
        D2 = np.einsum("i...,i...->...", np.conj(D), D).real
        # Modos DEGENERADOS, no solo el modo cero. Con el esquema rotado el
        # simbolo lleva el factor (1+e^{i xi h})/2, que se anula en Nyquist: en
        # 3D a 96^3 hay 286 modos con |D|^2 = 0 (doc 18). Dividir por ellos
        # amplifica el ruido de float32 de los campos guardados hasta 1e20 en
        # el desplazamiento reconstruido. Como no portan informacion de
        # desplazamiento, se enmascaran igual que el modo cero.
        self._zero = self.base._zero | (D2 <= 1e-12 * D2.max())
        self._invD2 = np.where(self._zero, 0.0, 1.0 / np.where(self._zero, 1.0, D2))

    def desplazamiento(self, eps):
        """Recupera la FLUCTUACION de desplazamiento a partir de la deformacion.

        De eps^ = sym(D (x) u^) se despeja  u^ = (M^H M)^{-1} M^H eps^, que es
        exactamente el vector intermedio que ya calcula `project`. Sale gratis:
        el desplazamiento no hay que guardarlo ni resolverlo aparte.

        Devuelve el campo real (d, *N), de media nula.
        """
        d = self.d
        axes = tuple(range(2, d + 2))
        ehat = _fftn(eps, axes=axes)
        v = np.einsum("i...,ij...->j...", np.conj(self.D), ehat)
        Dv = np.einsum("i...,i...->...", np.conj(self.D), v)
        a = (2.0 * v - self.D * (Dv * self._invD2)[None, ...]) * self._invD2[None, ...]
        a[:, self._zero] = 0.0
        return np.real(_ifftn(a, axes=tuple(range(1, d + 1))))

    def project(self, eps):
        """Proyecta un campo de deformaciones (d,d,*N) sobre el subespacio compatible."""
        d = self.d
        axes = tuple(range(2, d + 2))
        ehat = _fftn(eps, axes=axes)
        # v = M^H eps
        v = np.einsum("i...,ij...->j...", np.conj(self.D), ehat)
        # a = (M^H M)^{-1} v = (2 v - D (D^H v)/|D|^2) / |D|^2
        Dv = np.einsum("i...,i...->...", np.conj(self.D), v)
        a = (2.0 * v - self.D * (Dv * self._invD2)[None, ...]) * self._invD2[None, ...]
        a[:, self._zero] = 0.0
        # eps_proj = sym(D (x) a)
        out = 0.5 * (self.D[:, None, ...] * a[None, :, ...]
                     + self.D[None, :, ...] * a[:, None, ...])
        out[:, :, self._zero] = 0.0
        return np.real(_ifftn(out, axes=axes))


def solve_elasticity(lam, mu, E_macro, L=None, scheme="rotated", tol=1e-10,
                     maxiter=5000, stol=1e-10, projector=None):
    """Resuelve el problema de celda para una deformacion macroscopica E_macro.

    Devuelve (eps, sigma, info) con eps = E + eps~ y sigma = C : eps.
    """
    lam = np.asarray(lam, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    N = lam.shape
    d = len(N)
    P = projector if projector is not None else ElasticProjector(N, L, scheme)

    E_macro = np.asarray(E_macro, dtype=np.float64)
    E = np.zeros((d, d) + N)
    E[:, :] = E_macro.reshape((d, d) + (1,) * d)

    load = apply_C(lam, mu, E)
    b = -P.project(load)
    scale = np.sqrt(float(np.vdot(load, load).real))

    def A(e):
        return P.project(apply_C(lam, mu, e))

    def qoi(e):
        s = apply_C(lam, mu, E + e)
        return s.reshape(d * d, -1).mean(axis=1)

    et, info = _cg(A, b, tol=tol, maxiter=maxiter, scale=scale,
                   qoi=qoi, qoi_tol=stol)
    eps = E + et
    return eps, apply_C(lam, mu, eps), info


def effective_stiffness(lam, mu, L=None, scheme="rotated", tol=1e-10,
                        maxiter=5000, stol=1e-10):
    """Tensor de rigidez efectivo en notacion de Voigt-Mandel.

    Devuelve C_eff de forma (n, n) con n = d(d+1)/2, en la base de Mandel
    (las componentes de cizalla llevan factor sqrt(2)), de modo que C_eff es
    simetrica y sus autovalores son los modulos propios del material.

    Orden de las componentes:
      2D: (xx, yy, sqrt2*xy)
      3D: (xx, yy, zz, sqrt2*yz, sqrt2*xz, sqrt2*xy)
    """
    lam = np.asarray(lam, dtype=np.float64)
    N = lam.shape
    d = len(N)
    P = ElasticProjector(N, L, scheme)
    s2 = np.sqrt(2.0)

    if d == 2:
        pares = [(0, 0), (1, 1), (0, 1)]
    else:
        pares = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    n = len(pares)

    C = np.zeros((n, n))
    for col, (i, j) in enumerate(pares):
        E = np.zeros((d, d))
        if i == j:
            E[i, j] = 1.0
        else:
            E[i, j] = E[j, i] = 1.0 / s2   # base de Mandel: |E| = 1
        _, sigma, _ = solve_elasticity(lam, np.asarray(mu, float), E, L=L,
                                       scheme=scheme, tol=tol, maxiter=maxiter,
                                       stol=stol, projector=P)
        smean = sigma.reshape(d, d, -1).mean(axis=2)
        for fila, (k, l) in enumerate(pares):
            C[fila, col] = smean[k, l] * (1.0 if k == l else s2)
    return 0.5 * (C + C.T)
