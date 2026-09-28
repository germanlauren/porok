"""
Apertura de grieta desde el campo de fase.

Por que hace falta
------------------
Los momentos D^(m) con m > 0 (doc 04 seccion 4) pesan por la apertura, y el
cierre de Oda-Snow -- el estandar en yacimientos -- es exactamente
(1/12)[tr D^(3) I - D^(3)] (ver cierres.py). Sin una apertura FISICA, medida del
campo de desplazamiento y no postulada, esa comparacion no significaria nada:
estariamos comparando la referencia contra un modelo alimentado con un numero
inventado.

La identidad
------------
Para una grieta regularizada, el volumen de grieta es

    V_c = - INT_Omega  u . grad(phi)  dOmega

El signo menos es la convencion de phi: vale 1 en el nucleo y decae hacia
afuera, asi que grad(phi) apunta HACIA la grieta, opuesto a la normal saliente.
Sin el signo el volumen sale negativo.

que en el limite de grieta aguda converge a INT_Gamma [[u]] . n dGamma, el
salto de desplazamiento integrado sobre la superficie. Es la formula estandar
en campo de fase para fractura hidraulica (Bourdin, Chukwudozie, Yoshioka).

De ahi:

    apertura media  <b> = V_c / Gamma

y una densidad local de apertura

    b(x) ~ ( u . grad(phi) )(x) / gamma(x)

que es ruidosa donde gamma es pequena, asi que se enmascara.

Verificacion
------------
Contra la solucion de Sneddon para una grieta recta en un medio infinito bajo
traccion remota perpendicular (deformacion plana):

    volumen  V = 2 pi (1 - nu^2) sigma a^2 / E
    perfil   b(x) = 4 (1 - nu^2) sigma sqrt(a^2 - x^2) / E

Es el banco de pruebas clasico de la literatura de campo de fase para fractura
hidraulica, asi que verificar contra el conecta directamente con esa comunidad.
"""

from __future__ import annotations

import numpy as np

from .damage import gradient

__all__ = ["volumen_grieta", "apertura_media", "campo_apertura",
           "campo_conductividad", "sneddon_volumen", "sneddon_perfil"]


def volumen_grieta(u, phi, L=None, projector=None):
    """V_c / |Y| = < u . grad(phi) >.

    `u` es el campo de desplazamiento (d, *N); se puede obtener con
    ElasticProjector.desplazamiento(eps). Solo la fluctuacion contribuye al
    salto, pero la parte macroscopica se incluye si viene en `u`: al ser
    afin, su contribucion es -tr(E) <phi>, que es el cambio de volumen de la
    matriz, no apertura de grieta. Por eso conviene pasar SOLO la fluctuacion.
    """
    g = gradient(phi, L=L, projector=projector)
    return -float(np.mean(np.einsum("i...,i...->...", u, g)))


def apertura_media(u, phi, ell, L=None, projector=None):
    """Apertura media <b> = V_c / Gamma, con Gamma la densidad de grieta."""
    Vc = volumen_grieta(u, phi, L=L, projector=projector)
    Gamma = float(np.mean(phi ** 2) / ell)
    return Vc / Gamma if Gamma > 0 else 0.0


def campo_apertura(u, phi, ell, L=None, projector=None, umbral=None,
                   suavizado=None):
    """Campo local de apertura b(x), enmascarado donde no hay grieta.

    b(x) = (u . grad phi) / gamma(x),  con gamma = phi^2/(2l) + (l/2)|grad phi|^2

    Fuera de la grieta gamma tiende a cero y el cociente es ruido puro, asi que
    hay que definir DONDE hay grieta.

    umbral = None  (por defecto):  el conjunto es phi > 1/2, el mismo criterio
        que ya usan los cumulos y la percolacion. No tiene parametro libre.

    umbral = f (numero):  el conjunto es gamma > f * max(gamma). ES LO QUE
        HABIA, y esta mal para una serie: al localizar la grieta, max(gamma)
        crece, la mascara se encoge y el conjunto conductor se REDUCE aunque el
        dano aumente. Medido en la trayectoria uniaxial 2D, la permeabilidad
        construida con este campo BAJABA de 3.04 a 1.70 entre tr D = 0.165 y
        0.315 (doc 23). Se conserva solo para reproducir aquel diagnostico.
    """
    g = gradient(phi, L=L, projector=projector)
    num = -np.einsum("i...,i...->...", u, g)
    gamma = phi ** 2 / (2 * ell) + 0.5 * ell * (g ** 2).sum(axis=0)
    mascara = (phi > 0.5) if umbral is None else (gamma > umbral * gamma.max())
    b = np.zeros_like(phi)
    b[mascara] = num[mascara] / gamma[mascara]
    b = np.maximum(b, 0.0)       # una apertura negativa es interpenetracion
    if suavizado:
        from scipy.ndimage import gaussian_filter
        b = gaussian_filter(b, suavizado, mode="wrap")
    return b


# ------------------------------------------------------------------------- #
# Solucion de Sneddon, para verificacion
# ------------------------------------------------------------------------- #

def sneddon_volumen(sigma, a, E, nu):
    """Volumen (area en 2D) de una grieta de semilongitud `a` bajo traccion
    remota `sigma` perpendicular, en deformacion plana."""
    return 2.0 * np.pi * (1.0 - nu ** 2) * sigma * a ** 2 / E


def sneddon_perfil(x, sigma, a, E, nu):
    """Apertura b(x) a lo largo de la grieta, |x| <= a."""
    x = np.asarray(x, float)
    dentro = np.abs(x) < a
    b = np.zeros_like(x)
    b[dentro] = (4.0 * (1.0 - nu ** 2) * sigma / E) * np.sqrt(a ** 2 - x[dentro] ** 2)
    return b


def campo_conductividad(b, ell, lam_escala, k_m=1.0, tope=1e8):
    """Campo de conductividad equivalente a la ley cubica, k_b(x)/k_m.

        k_b/k_m = 1 + [b^3 / (12 w)] * LAMBDA,   LAMBDA = L^2 / k_m

    con w el espesor REAL de la banda donde phi > 1/2. Para el perfil de AT2
    phi = exp(-d/l) eso es d = l ln2 a cada lado, o sea w = 2 l ln 2. No queda
    ningun parametro libre: LAMBDA es fisica (tamano de celda y permeabilidad
    de matriz) y w sale del perfil.

    Con L = 10 cm y k_m = 1 mD, LAMBDA = 1e13 y a grieta formada el contraste
    local sale ~1e4, que es el que usa la referencia de Darcy desde el doc 05.
    """
    w = 2.0 * ell * np.log(2.0)
    return k_m * (1.0 + np.minimum((b ** 3 / (12.0 * w)) * lam_escala, tope))
