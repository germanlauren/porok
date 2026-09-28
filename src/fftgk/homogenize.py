"""
Homogeneizacion FFT-Galerkin del problema de Darcy/conductividad.

Problema local (celda unitaria periodica Y, dimension d = 2 o 3):

    q(x) = -k(x) . grad p(x)        (ley de Darcy local)
    div q(x) = 0                    (conservacion)
    p(x) = -G . x + p~(x),  p~ periodica

con G = -<grad p> el gradiente macroscopico impuesto. El tensor efectivo
K_eff se define por <q> = K_eff . G.

Se implementan DOS formulaciones variacionales:

  PRIMAL (espacio de gradientes, curl-free):
      e = grad p~ ,  e in E = {campos con media nula y rotacional nulo}
      hallar e tal que  G_curl [ k (G + e) ] = 0
      K_primal . G = < k (G + e) >

  DUAL (espacio de flujos, div-free):
      q = <q> + q~ ,  q~ in Q = {campos con media nula y divergencia nula}
      hallar q~ tal que  G_div [ k^{-1} (Q + q~) ] = 0
      K_dual^{-1} . Q = < k^{-1} (Q + q~) >

ADVERTENCIA sobre el gap primal-dual. En el CONTINUO estas dos formulaciones
acotan G.K.G por arriba y por abajo (principios de energia complementaria).
En el DISCRETO, si ambas usan el mismo simbolo de gradiente D(xi) -- de modo
que G_div = I - G_curl es exactamente el complemento ortogonal de G_curl --
los dos problemas son duales algebraicos exactos y dan el MISMO tensor. Se
verifica numericamente: el gap cae a ~1e-13. Por tanto:

    el gap primal-dual NO mide el error de discretizacion aqui;
    mide solo la convergencia del gradiente conjugado.

Sigue siendo util como test de correccion del codigo (un bug en un proyector,
en la inversion de k o en el ensamblaje rompe la igualdad), y por eso se
conserva. Para acotar el error de discretizacion se usan, en su lugar, el
refinamiento de malla y la comparacion entre esquemas (ver `scheme`). Obtener
cotas garantizadas exigiria evaluar los funcionales continuos sobre los campos
discretos con integracion exacta del material (esquema "Ga" de Vondrejc-Zeman-
Marek, frente al "GaNi" implementado aqui); queda pendiente.

Proyectores en Fourier, construidos desde el simbolo del gradiente discreto
D(xi) in C^d (ver FourierProjector para las opciones):
    G_curl(xi) = D D^H / |D|^2               (proyecta sobre gradientes)
    G_div(xi)  = I - D D^H / |D|^2           (proyecta sobre div-free)
    ambos = 0 en xi = 0  (los campos de fluctuacion tienen media nula)

Convenciones de forma (shape):
    N      : tupla de d enteros, la malla, p.ej. (256, 256)
    campo escalar   : array (*N)
    campo vectorial : array (d, *N)
    campo tensorial : array (d, d, *N)
    k puede darse como escalar-campo (*N) o tensorial (d, d, *N).
"""

from __future__ import annotations

import numpy as np

from .fft import fftn as _fftn, ifftn as _ifftn

__all__ = [
    "FourierProjector",
    "homogenize",
    "HomogenizationResult",
    "apply_k",
    "invert_k",
]


# ----------------------------------------------------------------------------
# Utilidades de campo material
# ----------------------------------------------------------------------------

def as_tensor_field(k, N):
    """Lleva k a la forma (d, d, *N). Acepta (*N) escalar o ya (d,d,*N)."""
    N = tuple(N)
    d = len(N)
    k = np.asarray(k)
    if k.shape == N:  # escalar isotropo por voxel
        eye = np.eye(d).reshape(d, d, *([1] * d))
        return eye * k[None, None, ...]
    if k.shape == (d, d) + N:
        return k
    raise ValueError(f"k con forma {k.shape} incompatible con malla {N} (d={d})")


def apply_k(k, field):
    """k(x) . field(x).  k: (d,d,*N) o (*N);  field: (d,*N) -> (d,*N)."""
    k = np.asarray(k)
    field = np.asarray(field)
    d = field.shape[0]
    if k.ndim == field.ndim - 1:  # escalar por voxel
        return k[None, ...] * field
    return np.einsum("ij...,j...->i...", k, field)


def invert_k(k, N):
    """Inversa puntual de un campo de tensores (d,d,*N) o escalar (*N)."""
    N = tuple(N)
    d = len(N)
    k = np.asarray(k)
    if k.shape == N:
        return 1.0 / k
    # mover los indices tensoriales al final para np.linalg.inv
    kt = np.moveaxis(np.moveaxis(k, 0, -1), 0, -1)  # (*N, d, d)
    inv = np.linalg.inv(kt)
    return np.moveaxis(np.moveaxis(inv, -1, 0), -1, 0)  # (d, d, *N)


# ----------------------------------------------------------------------------
# Proyectores de Fourier
# ----------------------------------------------------------------------------

class FourierProjector:
    """Proyectores ortogonales curl-free / div-free sobre la malla periodica.

    El proyector se construye a partir del SIMBOLO del operador gradiente
    discreto, D(xi) in C^d:

        G_curl(xi) = D D^H / |D|^2        (proyector ortogonal sobre range(D))
        G_div(xi)  = I - D D^H / |D|^2
        ambos nulos en xi = 0.

    Como G_curl es hermitico y D(-xi) = conj(D(xi)), los proyectores preservan
    campos reales. La eleccion de D fija la discretizacion:

    scheme = "spectral"
        D_j = i xi_j  (derivada exacta de polinomios trigonometricos).
        Esquema de Galerkin de Vondrejc-Zeman-Marek. Convergencia espectral
        para microestructuras suaves; en interfaces abruptas produce
        oscilaciones de Gibbs y el operador desarrolla un cuasi-nucleo que
        degrada mucho el CG a alto contraste.

    scheme = "fd"
        D_j = (exp(i xi_j h_j) - 1) / h_j  (diferencia adelantada; su adjunto
        es la diferencia atrasada). Equivale a un esquema de volumenes finitos
        de dos puntos en malla escalonada: sin oscilaciones, monotono, y con
        condicionamiento muy superior a alto contraste. Opcion robusta para
        microestructuras binarias fractura/matriz.

    scheme = "rotated"
        Esquema rotado de Willot (2015): promedia diferencias adelantadas
        sobre las diagonales de la celda. Mismo caracter no oscilatorio que
        "fd" pero con menos anisotropia de malla y menor error: en el limite
        diluido de Maxwell resulta ~2x mas preciso que "fd" y ~5x mas que
        "spectral" a igual malla. Es la opcion por defecto.
    """

    def __init__(self, N, L=None, scheme="rotated"):
        self.N = tuple(int(n) for n in N)
        self.d = len(self.N)
        self.L = tuple(float(x) for x in (L if L is not None else self.N))
        self.scheme = scheme
        if len(self.L) != self.d:
            raise ValueError("L debe tener la misma longitud que N")
        self.h = tuple(l / n for l, n in zip(self.L, self.N))

        # Frecuencias angulares por eje, con broadcasting a la malla completa.
        axes = []
        for i, (n, l) in enumerate(zip(self.N, self.L)):
            xi = 2.0 * np.pi * np.fft.fftfreq(n, d=l / n)
            shape = [1] * self.d
            shape[i] = n
            axes.append(np.broadcast_to(xi.reshape(shape), self.N))
        self.xi = np.stack(axes, axis=0).astype(np.float64)  # (d, *N)

        self.D = self._symbol()
        D2 = np.einsum("i...,i...->...", np.conj(self.D), self.D).real
        self._zero = D2 <= 1e-300
        self._inv_D2 = 1.0 / np.where(self._zero, 1.0, D2)

    def _symbol(self):
        xi, h, d = self.xi, self.h, self.d
        if self.scheme == "spectral":
            return 1j * xi.astype(np.complex128)
        if self.scheme == "fd":
            return np.stack(
                [(np.exp(1j * xi[j] * h[j]) - 1.0) / h[j] for j in range(d)],
                axis=0,
            )
        if self.scheme == "rotated":
            # Willot: derivada adelantada en j, promediada sobre las otras
            # direcciones con el factor (1 + exp(i xi_m h_m))/2.
            comp = []
            for j in range(d):
                s = (np.exp(1j * xi[j] * h[j]) - 1.0) / h[j]
                for m in range(d):
                    if m != j:
                        s = s * (1.0 + np.exp(1j * xi[m] * h[m])) / 2.0
                comp.append(s)
            return np.stack(comp, axis=0)
        raise ValueError(f"scheme desconocido: {self.scheme}")

    # -- nucleo comun ------------------------------------------------------
    def _hat_curl(self, fhat):
        """Aplica D D^H / |D|^2 en Fourier a un campo vectorial (d,*N)."""
        proj = np.einsum("i...,i...->...", np.conj(self.D), fhat) * self._inv_D2
        out = self.D * proj[None, ...]
        out[:, self._zero] = 0.0
        return out

    def _fwd(self, f):
        return _fftn(f, axes=tuple(range(1, self.d + 1)))

    def _bwd(self, fhat):
        return np.real(_ifftn(fhat, axes=tuple(range(1, self.d + 1))))

    def curl_free(self, f):
        """Proyecta el campo vectorial real f sobre gradientes de media nula."""
        return self._bwd(self._hat_curl(self._fwd(f)))

    def div_free(self, f):
        """Proyecta el campo vectorial real f sobre flujos solenoidales de media nula."""
        fhat = self._fwd(f)
        out = fhat - self._hat_curl(fhat)
        out[:, self._zero] = 0.0
        return self._bwd(out)


# ----------------------------------------------------------------------------
# Gradiente conjugado sobre operador-funcion
# ----------------------------------------------------------------------------

def _cg(A, b, x0=None, tol=1e-10, maxiter=2000, callback=None,
        qoi=None, qoi_tol=None, qoi_every=25, scale=None):
    """CG para A simetrico semidefinido positivo sobre el rango del proyector.

    A : callable(x)->y con la misma forma que x. Producto interno L2 en la malla.
    qoi : callable(x)->ndarray, cantidad de interes (aqui, la columna de K).
        El residuo del sistema converge mucho mas lento que las medias
        volumetricas; cuando se da `qoi` y `qoi_tol`, se corta tambien cuando
        la cantidad de interes se estabiliza en dos chequeos consecutivos.
        Esto es lo que decide el costo real del barrido.

    Devuelve (x, info) con info = dict(iters, resnorm, converged, stop).
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - A(x)
    p = r.copy()
    rs = float(np.vdot(r, r).real)
    # El residuo se normaliza contra la CARGA FISICA (||k.G||), no contra ||b||.
    # Motivo: cuando la solucion exacta es la trivial (p.ej. flujo uniforme en un
    # laminado), b es cero salvo redondeo. Normalizar contra ||b|| hace que CG
    # persiga ruido de 1e-16 sobre un operador singular, y alpha = rs/pAp explota
    # produciendo un x de orden 1 que contamina la media volumetrica. Normalizar
    # contra la carga hace que ese caso converja de inmediato, que es lo correcto.
    bnorm = float(scale) if scale is not None else np.sqrt(float(np.vdot(b, b).real))
    if bnorm <= 0.0:
        return x, {"iters": 0, "resnorm": 0.0, "converged": True, "stop": "carga-nula"}
    if np.sqrt(rs) / bnorm < tol:
        return x, {"iters": 0, "resnorm": np.sqrt(rs) / bnorm, "converged": True,
                   "stop": "solucion-trivial"}
    q_prev = None
    n_stable = 0
    it = 0
    resnorm = np.sqrt(rs) / bnorm
    for it in range(1, maxiter + 1):
        Ap = A(p)
        pAp = float(np.vdot(p, Ap).real)
        if pAp <= 0:
            break
        alpha = rs / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(np.vdot(r, r).real)
        resnorm = np.sqrt(rs_new) / bnorm
        if callback is not None:
            callback(it, resnorm)
        if resnorm < tol:
            return x, {"iters": it, "resnorm": resnorm, "converged": True,
                       "stop": "residuo"}
        if qoi is not None and qoi_tol is not None and it % qoi_every == 0:
            q = np.asarray(qoi(x), dtype=float)
            if q_prev is not None:
                scale = max(np.abs(q).max(), 1e-300)
                if np.abs(q - q_prev).max() / scale < qoi_tol:
                    n_stable += 1
                    if n_stable >= 2:
                        return x, {"iters": it, "resnorm": resnorm,
                                   "converged": True, "stop": "qoi"}
                else:
                    n_stable = 0
            q_prev = q
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, {"iters": it, "resnorm": resnorm, "converged": False,
               "stop": "maxiter"}


# ----------------------------------------------------------------------------
# Resultado
# ----------------------------------------------------------------------------

class HomogenizationResult:
    """Contenedor del tensor efectivo y sus diagnosticos."""

    def __init__(self, K_primal, K_dual, info_primal, info_dual, fields=None):
        self.K_primal = K_primal
        self.K_dual = K_dual
        self.info_primal = info_primal
        self.info_dual = info_dual
        self.fields = fields or {}

    @property
    def K(self):
        """Mejor estimacion: media de las dos cotas."""
        if self.K_dual is None:
            return self.K_primal
        return 0.5 * (self.K_primal + self.K_dual)

    @property
    def gap(self):
        """Discrepancia primal-dual relativa sobre la traza.

        NO es un estimador de error de discretizacion (ver docstring del
        modulo): primal y dual son duales algebraicos exactos con el mismo
        simbolo D. Es un test de correccion del codigo y de convergencia del
        gradiente conjugado; deberia quedar en ~1e-12 o menos.
        """
        if self.K_dual is None:
            return np.nan
        tp = np.trace(self.K_primal)
        td = np.trace(self.K_dual)
        return float(abs(tp - td) / abs(0.5 * (tp + td)))

    @property
    def converged(self):
        ok = bool(self.info_primal.get("converged", False))
        if self.info_dual is not None:
            ok = ok and bool(self.info_dual.get("converged", False))
        return ok

    def __repr__(self):
        with np.printoptions(precision=6, suppress=False):
            return (
                f"HomogenizationResult(\n  K =\n{self.K}\n"
                f"  gap_primal_dual = {self.gap:.3e}\n"
                f"  iters = ({self.info_primal['iters']}, "
                f"{self.info_dual['iters'] if self.info_dual else '-'})\n)"
            )


# ----------------------------------------------------------------------------
# Driver principal
# ----------------------------------------------------------------------------

def homogenize(k, L=None, tol=1e-10, maxiter=5000, dual=True,
               return_fields=False, projector=None, scheme="rotated",
               ktol=1e-9):
    """Calcula el tensor de permeabilidad/conductividad efectivo de k(x).

    Parameters
    ----------
    k : ndarray
        Campo material. Forma (*N) para isotropo local, o (d,d,*N) para
        anisotropo local. Debe ser estrictamente positivo (definido positivo).
    L : tuple[float] | None
        Longitudes fisicas de la celda. Por defecto N (voxels cubicos unitarios).
    tol, maxiter : float, int
        Control del gradiente conjugado.
    dual : bool
        Si True (por defecto) resuelve tambien el problema dual y devuelve la
        cota inferior. Duplica el costo pero da el estimador de error.
    return_fields : bool
        Guarda los campos locales del ultimo eje resuelto (para figuras).

    Returns
    -------
    HomogenizationResult
    """
    k = np.asarray(k, dtype=np.float64)
    # Deducir malla y dimension.
    # Campo tensorial: (d, d, *N) con d = ndim - 2 y las dos primeras iguales a d.
    # Cualquier otra cosa se interpreta como campo escalar por voxel: (*N).
    if k.ndim >= 4 and k.shape[0] == k.shape[1] == k.ndim - 2:
        N = tuple(k.shape[2:])
    else:
        N = tuple(k.shape)
    d = len(N)

    P = projector if projector is not None else FourierProjector(N, L, scheme)
    vol_mean = lambda f: f.reshape(f.shape[0], -1).mean(axis=1)  # noqa: E731

    kinv = invert_k(k, N) if dual else None

    K_primal = np.zeros((d, d))
    K_dual_inv = np.zeros((d, d))
    info_p = {"iters": 0, "resnorm": 0.0, "converged": True}
    info_d = {"iters": 0, "resnorm": 0.0, "converged": True} if dual else None
    fields = {}

    for j in range(d):
        # ---------------- PRIMAL ----------------
        G = np.zeros((d,) + N)
        G[j] = 1.0

        def A_primal(e):
            return P.curl_free(apply_k(k, e))

        load = apply_k(k, G)
        b = -P.curl_free(load)
        e, ip = _cg(A_primal, b, tol=tol, maxiter=maxiter,
                    scale=np.sqrt(float(np.vdot(load, load).real)),
                    qoi=lambda ee: vol_mean(apply_k(k, G + ee)), qoi_tol=ktol)
        q = apply_k(k, G + e)
        K_primal[:, j] = vol_mean(q)
        info_p["iters"] = max(info_p["iters"], ip["iters"])
        info_p["resnorm"] = max(info_p["resnorm"], ip["resnorm"])
        info_p["converged"] &= ip["converged"]
        info_p["stop"] = ip["stop"]

        if return_fields:
            fields[f"grad_p_{j}"] = G + e
            fields[f"flux_{j}"] = q

        # ---------------- DUAL ----------------
        if dual:
            Q = np.zeros((d,) + N)
            Q[j] = 1.0

            def A_dual(qq):
                return P.div_free(apply_k(kinv, qq))

            load_d = apply_k(kinv, Q)
            bd = -P.div_free(load_d)
            qt, idd = _cg(A_dual, bd, tol=tol, maxiter=maxiter,
                          scale=np.sqrt(float(np.vdot(load_d, load_d).real)),
                          qoi=lambda qq: vol_mean(apply_k(kinv, Q + qq)),
                          qoi_tol=ktol)
            g = apply_k(kinv, Q + qt)
            K_dual_inv[:, j] = vol_mean(g)   # <grad p> para <q> = e_j
            info_d["iters"] = max(info_d["iters"], idd["iters"])
            info_d["resnorm"] = max(info_d["resnorm"], idd["resnorm"])
            info_d["converged"] &= idd["converged"]
            info_d["stop"] = idd["stop"]

    # simetrizar (la asimetria residual es error numerico)
    K_primal = 0.5 * (K_primal + K_primal.T)
    K_dual = None
    if dual:
        K_dual_inv = 0.5 * (K_dual_inv + K_dual_inv.T)
        K_dual = np.linalg.inv(K_dual_inv)
        K_dual = 0.5 * (K_dual + K_dual.T)

    return HomogenizationResult(K_primal, K_dual, info_p, info_d, fields)
