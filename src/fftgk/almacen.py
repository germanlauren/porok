"""
Almacenamiento HDF5 de una corrida: campos + escalares, en un solo archivo.

Diseno pensado para el barrido en HPC
-------------------------------------
Un archivo por trayectoria. Dentro, un grupo por incremento con los campos, y
tablas de escalares acumuladas en el raiz para poder leer TODA la serie sin
tocar los campos. En la practica uno analiza cientos de veces los escalares y
dos veces los campos, asi que separarlos es lo que hace el analisis agil.

Estructura:

    /meta                       atributos: malla, l, contraste, semilla, ...
    /escalares/<nombre>         array (n_incrementos,) o (n,3,3) para tensores
    /campos/<0000>/phi          float32 comprimido
    /campos/<0000>/H
    /campos/<0000>/eps
    /campos/<0000>/apertura     si se calculo

Decisiones que importan para la cuota de disco (doc 06 seccion 4):

  * los campos van en **float32 con compresion gzip**. El campo de fase no
    necesita doble precision para analizarse; la corrida si la usa
    internamente. Reduce ~6x frente a float64 sin comprimir.
  * se guardan campos solo cada `cada_n` incrementos. Los escalares, siempre.
  * escritura incremental con flush: una corrida interrumpida conserva todo lo
    anterior. Con trabajos de horas en cola, esto no es opcional.

Si falta h5py, se cae a .npz por incremento sin romper la corrida.
"""

from __future__ import annotations

import pathlib
import numpy as np

try:
    import h5py
    HAY_H5 = True
except ImportError:      # pragma: no cover
    HAY_H5 = False

__all__ = ["Almacen", "leer_escalares", "leer_campo", "listar"]


class Almacen:
    """Escritor incremental de una corrida."""

    def __init__(self, ruta, meta=None, cada_n=1, comprimir=4):
        self.ruta = pathlib.Path(ruta)
        self.cada_n = int(cada_n)
        self.comprimir = comprimir
        self.n = 0
        self._escalares = {}
        if not HAY_H5:
            self.ruta.mkdir(parents=True, exist_ok=True)
            self._meta = dict(meta or {})
            return
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.ruta, "a") as f:
            g = f.require_group("meta")
            for k, v in (meta or {}).items():
                g.attrs[k] = v
            f.require_group("escalares")
            f.require_group("campos")
            self.n = len(f["campos"]) and int(
                max(int(k) for k in f["campos"])) + 1 or len(f["campos"])

    # ------------------------------------------------------------------ #
    def guardar(self, escalares, campos=None):
        """Anade un incremento. `escalares` dict de numeros o arrays pequenos;
        `campos` dict de arrays grandes (se guardan solo cada `cada_n`)."""
        i = self.n
        self.n += 1
        if not HAY_H5:
            np.savez_compressed(self.ruta / f"inc_{i:04d}.npz",
                                **{k: np.asarray(v) for k, v in escalares.items()},
                                **({k: np.asarray(v, np.float32)
                                    for k, v in (campos or {}).items()}
                                   if i % self.cada_n == 0 else {}))
            return
        with h5py.File(self.ruta, "a") as f:
            ge = f["escalares"]
            for k, v in escalares.items():
                v = np.asarray(v)
                if k not in ge:
                    ge.create_dataset(
                        k, shape=(0,) + v.shape, maxshape=(None,) + v.shape,
                        dtype="f8" if v.dtype.kind == "f" else v.dtype,
                        chunks=True)
                ds = ge[k]
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = v
            if campos and i % self.cada_n == 0:
                gc = f["campos"].create_group(f"{i:04d}")
                for k, v in campos.items():
                    gc.create_dataset(k, data=np.asarray(v, np.float32),
                                      compression="gzip",
                                      compression_opts=self.comprimir)
            f.flush()


# ---------------------------------------------------------------------- #
# Lectura
# ---------------------------------------------------------------------- #

def leer_escalares(ruta):
    """Devuelve un dict {nombre: array} con TODA la serie de escalares.

    No toca los campos, asi que es instantaneo aunque el archivo pese gigas.
    """
    with h5py.File(ruta, "r") as f:
        return {k: f["escalares"][k][...] for k in f["escalares"]}


def leer_campo(ruta, incremento, nombre="phi"):
    """Lee un campo de un incremento concreto."""
    with h5py.File(ruta, "r") as f:
        return f["campos"][f"{incremento:04d}"][nombre][...]


def listar(ruta):
    """Resumen del archivo: metadatos, escalares disponibles, incrementos con campos."""
    with h5py.File(ruta, "r") as f:
        meta = dict(f["meta"].attrs)
        esc = {k: f["escalares"][k].shape for k in f["escalares"]}
        incs = sorted(int(k) for k in f["campos"])
        campos = list(f["campos"][f"{incs[0]:04d}"]) if incs else []
    return {"meta": meta, "escalares": esc, "incrementos_con_campos": incs,
            "campos": campos}
