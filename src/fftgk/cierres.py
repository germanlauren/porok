"""
Los cierres que usa la industria, escritos para poder compararlos SIN AJUSTAR.

-------------------------------------------------------------------------
1. La identidad que hace exacta la comparacion
-------------------------------------------------------------------------

El tensor de permeabilidad de fracturas de Oda-Snow, que es el estandar en
ingenieria de yacimientos para roca fracturada, es

    k_ij^Oda = (1 / 12 V) SUM_k  A_k b_k^3 ( delta_ij - n_i n_j )

con A_k el area de la fractura k, b_k su apertura y n_k su normal. El factor
b^3/12 es la ley cubica; el parentesis proyecta fuera de la normal, porque el
flujo corre EN el plano de la fractura, no a traves.

Nuestro momento de apertura de orden 3 (doc 04 seccion 4) es

    D^(3)_ij = (1/V) SUM_k A_k b_k^3 n_i n_j

y como tr(n (x) n) = 1, se tiene tr D^(3) = (1/V) SUM_k A_k b_k^3. Por tanto

    k^Oda = (1/12) [ tr(D^(3)) I  -  D^(3) ]        EXACTO

**El cierre de la industria es exactamente una funcion de nuestro momento
D^(3).** No hay aproximacion ni constante ajustable. Eso convierte la pregunta
del proyecto en algo comprobable sin margen de discusion: si K difiere entre
trayectorias que comparten D^(0) y D^(3), entonces ni la ley cubica ni el
tensor de dano ni ninguna combinacion de ambos puede predecir K, y no es por
mala calibracion.

-------------------------------------------------------------------------
2. Los otros dos cierres
-------------------------------------------------------------------------

Exponencial de dano (Souley y col. 2001; Shao y col. 2005). Escalar e isotropo,
muy usado por su simplicidad:

    k = k_0 exp(alpha D),     alpha tipicamente 5 a 25

Fractal (familia Yu-Cheng y derivados). La permeabilidad se liga a la porosidad
por una ley de potencias con exponente ligado a la dimension fractal:

    k = k_0 (phi / phi_0)^m,   m ~ 3 a 5 segun la dimension fractal supuesta

Ambos llevan un parametro libre. Para que la comparacion sea honesta se
CALIBRAN al mejor ajuste contra la referencia en cada trayectoria por separado
y aun asi se mide el error: asi se reporta el error IRREDUCIBLE del cierre, no
el error de una calibracion desafortunada. Si un cierre falla incluso con su
mejor parametro posible, el problema es su forma funcional.
"""

from __future__ import annotations

import numpy as np

__all__ = ["oda_desde_D3", "exponencial_dano", "fractal_porosidad",
           "calibrar_escalar", "error_relativo"]


def oda_desde_D3(D3, d=None):
    """Tensor de permeabilidad de Oda-Snow desde el momento D^(3).

        k^Oda = (1/12) [ tr(D3) I - D3 ]

    Exacto, sin parametros. `D3` debe venir de damage_tensor(..., m=3) con el
    campo de apertura fisico.
    """
    D3 = np.asarray(D3, float)
    d = d if d is not None else D3.shape[0]
    return (np.trace(D3) * np.eye(d) - D3) / 12.0


def exponencial_dano(D_escalar, k0, alpha):
    """k = k0 exp(alpha D). Isotropo, un parametro."""
    return k0 * np.exp(alpha * np.asarray(D_escalar, float))


def fractal_porosidad(phi_poro, k0, phi0, m):
    """k = k0 (phi/phi0)^m. Ley de potencias tipo fractal, un parametro (m)."""
    return k0 * (np.asarray(phi_poro, float) / phi0) ** m


def calibrar_escalar(modelo, x, K_ref, p0, cotas):
    """Ajusta el UNICO parametro libre de un cierre escalar por minimos
    cuadrados en logaritmo.

    Se ajusta en log porque K recorre ordenes de magnitud y un ajuste lineal
    quedaria dominado por los puntos grandes -- justo los que cualquier modelo
    acierta. En log, el ajuste reparte el error de forma comparable en todo el
    rango, que es lo que interesa para un mapa de error.

    Devuelve (parametro_optimo, K_modelo, error_relativo_mediano).
    """
    from scipy.optimize import minimize_scalar

    x = np.asarray(x, float)
    K_ref = np.asarray(K_ref, float)
    ok = np.isfinite(x) & np.isfinite(K_ref) & (K_ref > 0)

    def costo(p):
        Km = modelo(x[ok], p)
        if np.any(Km <= 0):
            return 1e30
        return float(np.mean((np.log(Km) - np.log(K_ref[ok])) ** 2))

    res = minimize_scalar(costo, bounds=cotas, method="bounded")
    p = float(res.x)
    Km = modelo(x, p)
    err = error_relativo(Km, K_ref)
    return p, Km, float(np.nanmedian(np.abs(err)))


def error_relativo(K_modelo, K_ref):
    """Error relativo con signo, (modelo - referencia) / referencia."""
    K_modelo = np.asarray(K_modelo, float)
    K_ref = np.asarray(K_ref, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (K_modelo - K_ref) / K_ref
