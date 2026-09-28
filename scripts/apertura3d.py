"""
Apertura, D^(1), D^(3) y Oda-Snow para el barrido 3D YA CORRIDO.

No hace falta repetir las 12 h de barrido: los campos phi y eps estan guardados
cada `campos_cada` incrementos, y la apertura se reconstruye de ahi. El
desplazamiento sale de la fluctuacion de deformacion con el mismo proyector.

Escribe `apertura_3d.jsonl`, chico, una linea por incremento con campos.

Uso:
    python3 scripts/apertura3d.py                 # ~/porok_corridas
    python3 scripts/apertura3d.py otra/carpeta
"""
import glob
import json
import pathlib
import sys

import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "src"))

import os
# L^2/k_m: celda de 10 cm, matriz de 1 mD. Es el unico parametro fisico.
LAMBDA = float(os.environ.get("POROK_LAMBDA", "1e13"))

from fftgk.almacen import leer_campo, leer_escalares, listar     # noqa: E402
from fftgk.elasticity import ElasticProjector                    # noqa: E402
from fftgk.homogenize import FourierProjector                       # noqa: E402
from fftgk.apertura import (volumen_grieta, campo_apertura,       # noqa: E402
                            campo_conductividad)
from fftgk.homogenize import homogenize                          # noqa: E402
from fftgk.damage import damage_tensor                           # noqa: E402
from fftgk.cierres import oda_desde_D3                           # noqa: E402


def uno(ruta, salida):
    info = listar(ruta)
    meta = info.get("meta", {})
    n = int(meta.get("n", 96))
    d = int(meta.get("dim", 3))
    ell = float(meta.get("ell_px", 4.0)) / n
    tray = str(meta.get("trayectoria", ruta.name.split("_n")[0]))
    esc = leer_escalares(ruta)
    L = (1.0,) * d
    base = FourierProjector((n,) * d, L, "rotated")
    P = ElasticProjector((n,) * d, L, "rotated", base=base)
    # el gradiente de phi usa el mismo operador que la corrida (fd)
    base_phi = FourierProjector((n,) * d, L,
                                str(meta.get("esquema_phi", "fd")))
    filas = 0
    for idx in info.get("incrementos_con_campos", []):
        phi = np.asarray(leer_campo(ruta, idx, "phi"), float)
        eps = np.asarray(leer_campo(ruta, idx, "eps"), float)
        ef = eps - eps.reshape(d, d, -1).mean(axis=2).reshape((d, d) + (1,) * d)
        u = P.desplazamiento(ef)
        Vc = volumen_grieta(u, phi, L=L, projector=base_phi)
        b = campo_apertura(u, phi, ell, L=L, projector=base_phi)
        Gam = float(np.mean(phi ** 2) / ell)
        b_med = Vc / Gam if Gam > 0 else 0.0
        D3 = damage_tensor(phi, ell, L=L, weight=b, m=3)
        # referencia COMPATIBLE con la ley cubica (doc 24): misma geometria,
        # conductividad local dada por la apertura. Es contra esta, y no contra
        # la de Darcy, contra la que se juzga a Oda-Snow.
        kb = campo_conductividad(b, ell, LAMBDA)
        Kb = homogenize(kb, L=L, tol=1e-9, maxiter=2000, dual=False,
                        ktol=1e-9).K_primal
        fila = dict(
            trayectoria=tray, inc=int(idx) + 1,
            Gamma=float(esc["Gamma"][idx]),
            trD=float(np.trace(np.asarray(esc["D"][idx]))),
            K=np.asarray(esc["K"][idx]).tolist(),
            Vc=float(Vc), b_medio=float(b_med), b_max=float(b.max()),
            D1=damage_tensor(phi, ell, L=L, weight=b, m=1).tolist(),
            D3=D3.tolist(), Koda=oda_desde_D3(D3).tolist(),
            K_cubica=float(Gam * b_med ** 3 / 12.0),
            Kb=Kb.tolist(), Lambda=LAMBDA)
        salida.write(json.dumps(fila) + "\n")
        salida.flush()
        filas += 1
        print(f"  {tray:22s} inc {idx+1:3d}  Gamma={fila['Gamma']:.3f}  "
              f"<b>={b_med:.3e}  Koda_zz={fila['Koda'][-1][-1]:.3e}  "
              f"Kb-1=({Kb[0][0]-1:.4f},{Kb[-1][-1]-1:.4f})  "
              f"OdaxLam={LAMBDA*fila['Koda'][-1][-1]:.3e}", flush=True)
    return filas


def main(carpeta, semilla=None):
    """`semilla` filtra por corrida: con varias semillas en la misma carpeta,
    y una de ellas todavia corriendo, conviene procesar solo la terminada."""
    carpeta = pathlib.Path(carpeta).expanduser()
    patron = f"*_d3_s{semilla}_fd.h5" if semilla else "*_d3_*_fd.h5"
    archivos = sorted(glob.glob(str(carpeta / patron)))
    if not archivos:
        sys.exit(f"no hay {patron} en {carpeta}")
    sal = AQUI.parent / (f"apertura_3d_s{semilla}.jsonl" if semilla
                         else "apertura_3d.jsonl")
    total = 0
    with open(sal, "w") as f:
        for a in archivos:
            total += uno(pathlib.Path(a), f)
    print(f"\n-> {sal}   ({total} incrementos, "
          f"{sal.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/porok_corridas",
         sys.argv[2] if len(sys.argv) > 2 else None)
