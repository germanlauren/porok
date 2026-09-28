"""
Descriptores topologicos de la red de grietas.

Motivacion (ver doc 04 del proyecto). El tensor de dano D_ij es demostrablemente
ciego a la conectividad: dos microestructuras con la misma longitud total de
grieta y la misma orientacion dan el mismo D_ij dentro del error numerico,
aunque una percole y la otra no. Como la permeabilidad la controla justamente la
conectividad, hace falta un descriptor adicional.

Se implementan tres niveles, de mas barato a mas informativo:

  1. caracteristica de Euler (chi)  -- funcional de Minkowski, O(N), escalar.
     En 2D, para el complejo cubico: chi = V - E + F. Mide "numero de
     componentes menos numero de agujeros". Es el descriptor que la literatura
     de medios porosos ya relaciona con permeabilidad (Vogel, Mecke, Armstrong),
     pero solo para microestructuras estaticas y NO dañadas.

  2. estadistica de cumulos -- numero de componentes conexas periodicas,
     fraccion del cumulo mas grande, y test de percolacion por direccion.
     Mas caro pero es lo que realmente decide si hay camino de flujo.

  3. tensor de Minkowski (pendiente) -- los funcionales escalares son
     invariantes de movimiento, es decir CIEGOS A LA ANISOTROPIA por
     construccion. Es la limitacion que declara el paper de 2025 sobre
     funcionales de Minkowski como predictores de permeabilidad. Los tensores
     de Minkowski (Schroder-Turk y col.) resuelven exactamente eso y nadie los
     ha aplicado a permeabilidad inducida por dano. Ver doc 03 seccion 2.3.

Todo asume celda PERIODICA, consistente con la homogeneizacion.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "euler_characteristic_2d",
    "periodic_label",
    "cluster_stats",
    "percolates",
]


def euler_characteristic_2d(mask):
    """Caracteristica de Euler de una mascara binaria 2D periodica.

    Complejo cubico: chi = V - E + F, donde V son los pixeles activos, E los
    pares adyacentes (derecha y abajo) y F los bloques 2x2 completos. Todo con
    envoltura periodica.

    Interpretacion: chi > 0 domina el numero de componentes (red fragmentada);
    chi < 0 domina el numero de lazos (red muy conectada). El cambio de signo
    marca aproximadamente el umbral de percolacion, y por eso es un candidato
    natural a descriptor del cierre.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("euler_characteristic_2d requiere una mascara 2D")
    V = int(m.sum())
    E = int((m & np.roll(m, -1, axis=0)).sum() + (m & np.roll(m, -1, axis=1)).sum())
    F = int((m
             & np.roll(m, -1, axis=0)
             & np.roll(m, -1, axis=1)
             & np.roll(np.roll(m, -1, axis=0), -1, axis=1)).sum())
    return V - E + F


def _union_find(n):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    return find, union, parent


def periodic_label(mask, structure=None):
    """Etiquetado de componentes conexas con envoltura PERIODICA.

    scipy.ndimage.label no conoce periodicidad. Aqui se etiqueta sin ella y
    luego se fusionan las etiquetas que se tocan a traves de cada borde
    periodico, con union-find.

    Devuelve (labels, n) con etiquetas 1..n y 0 para el fondo.
    """
    from scipy import ndimage

    m = np.asarray(mask, dtype=bool)
    lab, n = ndimage.label(m, structure=structure)
    if n == 0:
        return lab, 0

    find, union, _ = _union_find(n + 1)
    for ax in range(m.ndim):
        a = np.take(lab, 0, axis=ax)
        b = np.take(lab, m.shape[ax] - 1, axis=ax)
        tocan = (a > 0) & (b > 0)
        for la, lb in zip(a[tocan].ravel(), b[tocan].ravel()):
            union(int(la), int(lb))

    raiz = np.array([find(i) if i > 0 else 0 for i in range(n + 1)])
    # renumerar de forma compacta
    unicas = np.unique(raiz[1:])
    remap = np.zeros(n + 1, dtype=int)
    for nueva, vieja in enumerate(unicas, start=1):
        remap[raiz == vieja] = nueva
    remap[0] = 0
    return remap[lab], len(unicas)


def percolates(mask, axis):
    """¿Existe un cumulo que da la vuelta a la celda en la direccion `axis`?

    Test estandar por replicacion: se apila la mascara 3 veces en esa direccion
    y se etiqueta sin periodicidad. Un cumulo que abarca de un extremo al otro
    de la pila necesariamente da la vuelta en la celda original; uno que solo
    toca ambos bordes sin envolver, no.
    """
    from scipy import ndimage

    m = np.asarray(mask, dtype=bool)
    apilada = np.concatenate([m, m, m], axis=axis)
    lab, n = ndimage.label(apilada)
    if n == 0:
        return False
    primera = set(np.unique(np.take(lab, 0, axis=axis))) - {0}
    ultima = set(np.unique(np.take(lab, apilada.shape[axis] - 1, axis=axis))) - {0}
    return len(primera & ultima) > 0


def cluster_stats(mask):
    """Resumen topologico de la mascara.

    Devuelve dict con:
      n_clusters        numero de componentes conexas periodicas
      f_total           fraccion de volumen de la fase
      f_mayor           fraccion del volumen de la fase en el cumulo mas grande
      percola           tupla de booleanos, uno por eje
      chi               caracteristica de Euler (solo 2D)
    """
    m = np.asarray(mask, dtype=bool)
    lab, n = periodic_label(m)
    total = int(m.sum())
    if total == 0:
        f_mayor = 0.0
    else:
        cuentas = np.bincount(lab.ravel())[1:]
        f_mayor = float(cuentas.max()) / total
    out = {
        "n_clusters": int(n),
        "f_total": total / m.size,
        "f_mayor": f_mayor,
        "percola": tuple(percolates(m, ax) for ax in range(m.ndim)),
    }
    if m.ndim == 2:
        out["chi"] = euler_characteristic_2d(m)
    return out
