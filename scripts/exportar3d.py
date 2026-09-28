"""
Exporta el barrido 3D a UN archivo chico (resultados_3d.jsonl) para analizarlo
fuera de la maquina, sin mover los .h5 (que pesan GB).

Una linea por incremento: trayectoria, Gamma, lam, trD, D (3x3), K (3x3), R,
n_clusters, f_mayor, percola, stag, segundos. Ademas, para el ULTIMO campo
guardado de cada trayectoria, el control de resolucion: cuantos cumulos estan
por debajo de una bola de radio l/2 (bajo_l) y el tamano del mayor.

Uso
---
    python3 scripts/exportar3d.py                    # ~/porok_corridas/*_d3_*_fd.h5
    python3 scripts/exportar3d.py ruta/a/carpeta
"""
import glob
import json
import pathlib
import sys

import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "src"))
from fftgk.almacen import leer_escalares, leer_campo, listar   # noqa: E402


def control_campo(ruta, idx, ell_px):
    from scipy import ndimage
    phi = np.asarray(leer_campo(ruta, idx, "phi"), float)
    lab, ncl = ndimage.label(phi > 0.5)
    if ncl == 0:
        return dict(inc=int(idx), n_clusters=0)
    tam = np.bincount(lab.ravel())[1:]
    minimo = (4.0 / 3.0) * np.pi * (ell_px / 2.0) ** 3
    return dict(inc=int(idx), n_clusters=int(ncl),
                bajo_l=int((tam < minimo).sum()), mayor=int(tam.max()),
                f_danada=float((phi > 0.5).mean()), phi_max=float(phi.max()))


def main(carpeta):
    carpeta = pathlib.Path(carpeta).expanduser()
    archivos = sorted(glob.glob(str(carpeta / "*_d3_*_fd.h5")))
    if not archivos:
        sys.exit(f"no hay *_d3_*_fd.h5 en {carpeta}")
    sal = AQUI.parent / "resultados_3d.jsonl"
    n_lineas = 0
    with open(sal, "w") as f:
        for a in archivos:
            ruta = pathlib.Path(a)
            info = listar(ruta)
            meta = info.get("meta", {})
            tray = str(meta.get("trayectoria", ruta.name.split("_n")[0]))
            e = leer_escalares(ruta)
            n = len(e["Gamma"])
            for i in range(n):
                D = np.asarray(e["D"][i], float)
                K = np.asarray(e["K"][i], float)
                f.write(json.dumps(dict(
                    trayectoria=tray, semilla=int(meta.get("semilla", 0)),
                    inc=i + 1, Gamma=float(e["Gamma"][i]),
                    lam=float(e["lam"][i]), trD=float(np.trace(D)),
                    D=D.tolist(), K=K.tolist(), R=float(e["R"][i]),
                    n_clusters=int(e["n_clusters"][i]),
                    f_mayor=float(e["f_mayor"][i]),
                    percola=np.atleast_1d(e["percola"][i]).astype(int).tolist(),
                    stag=int(e["stag"][i]),
                    segundos=float(e["segundos"][i]))) + "\n")
                n_lineas += 1
            guardados = list(info.get("incrementos_con_campos", []))
            ctrl = (control_campo(ruta, guardados[-1], float(meta.get("ell_px", 4)))
                    if guardados else {})
            f.write(json.dumps(dict(trayectoria=tray, control_final=ctrl)) + "\n")
            print(f"  {tray:24s} {n:3d} inc  Gamma_fin={e['Gamma'][-1]:.3f}  "
                  f"trD={np.trace(np.asarray(e['D'][-1])):.3f}  "
                  f"cum={int(e['n_clusters'][-1])}  "
                  f"perc={np.atleast_1d(e['percola'][-1]).astype(int).tolist()}  "
                  f"ultimo campo: {ctrl}")
    kb = sal.stat().st_size / 1024
    print(f"\n-> {sal}   ({n_lineas} incrementos, {kb:.0f} kB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/porok_corridas")
