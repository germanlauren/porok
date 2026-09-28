"""
Backend de FFT centralizado.

La FFT es la operacion dominante de todo el codigo: el proyector de Darcy, el
de elasticidad, el gradiente del campo de fase y el operador del campo de fase
la usan en cada aplicacion, y esas aplicaciones son el cuerpo del gradiente
conjugado. Cualquier factor aqui se multiplica por todo.

Medido en la maquina de desarrollo (solo 2 nucleos):

        forma          numpy.fft   scipy 1 hilo   scipy N hilos   ganancia
    (512, 512)            6.8 ms        2.9 ms         2.6 ms       2.6x
    (1024, 1024)         31.7 ms       13.4 ms         8.7 ms       3.7x
    (64, 64, 64)          5.9 ms        2.7 ms         3.1 ms       1.9x

Notese que la mayor parte de la ganancia aparece ya con UN hilo: scipy usa
pocketfft con mejor planificacion que numpy. El multihilo suma encima, y en un
nodo de HPC con decenas de nucleos esa parte sera mucho mayor que aqui.

Por defecto se usan todos los nucleos disponibles segun la AFINIDAD del proceso
(no os.cpu_count()), que es lo que respeta la asignacion de SLURM: pedir 8 CPUs
y lanzar 64 hilos degrada el rendimiento y molesta a los demas trabajos del nodo.

Se puede fijar a mano:
    from fftgk import fft as F
    F.WORKERS = 8          # o 1 para forzar un solo hilo
"""

from __future__ import annotations

import os
import numpy as np

try:
    import scipy.fft as _sfft
    HAY_SCIPY = True
except ImportError:      # pragma: no cover
    _sfft = None
    HAY_SCIPY = False


def _nucleos():
    n = os.environ.get("FFTGK_WORKERS")
    if n:
        return int(n)
    for v in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        if os.environ.get(v):
            try:
                return max(1, int(os.environ[v]))
            except ValueError:
                pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


WORKERS = _nucleos()


def fftn(x, axes=None):
    if HAY_SCIPY:
        return _sfft.fftn(x, axes=axes, workers=WORKERS)
    return np.fft.fftn(x, axes=axes)


def ifftn(x, axes=None):
    if HAY_SCIPY:
        return _sfft.ifftn(x, axes=axes, workers=WORKERS)
    return np.fft.ifftn(x, axes=axes)


def info():
    return {"backend": "scipy.fft" if HAY_SCIPY else "numpy.fft",
            "workers": WORKERS}
