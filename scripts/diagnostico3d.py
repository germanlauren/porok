"""
Diagnostico de una corrida 3D: ¿el campo de fase esta RESUELTO?

Por que hace falta
------------------
El piloto 3D a 96^3 terminó con numeros que no se parecen a nada del barrido
2D publicado:

                        2D (96^2)        3D (96^3)
    stag (mediana)        8 a 12          45 a 80
    n_clusters (max)        7              4269
    R (max)               1.48             9.28

Las tres cosas apuntan al mismo sitio. R = B/A compara el termino de gradiente
con el termino de bulto de la superficie regularizada:

    A = (1/|Y|) INT phi^2 / l  dV        B = (l/|Y|) INT |grad phi|^2 dV

Para el perfil exacto de AT2 hay EQUIPARTICION: A = B, o sea R = 1. Un campo
difuso da R < 1 porque casi no tiene gradiente. Pero **R >> 1 solo puede salir
de un phi que varia en una escala mas corta que l**, y AT2 no admite eso en
equilibrio: la longitud de regularizacion es precisamente lo que impide que el
dano se concentre por debajo de ella.

Con 4269 cumulos, la lectura natural es que el campo esta lleno de motas de
dano mas chicas que l. Eso no es una red de grietas: es un campo sin converger.
Y encaja con que stag llegue al tope de 80 sin parar.

Este script decide entre las dos posibilidades sin volver a correr nada caro.

Uso
---
    python3 scripts/diagnostico3d.py ~/porok_corridas/P1_uniaxial_n96_d3_s20260914.h5

Escribe ademas `diagnostico3d.json`, que es chico y se puede mandar entero.
"""
import json
import pathlib
import sys

import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "src"))

from fftgk.almacen import leer_escalares, leer_campo, listar   # noqa: E402


def main(ruta):
    ruta = pathlib.Path(ruta).expanduser()
    info = listar(ruta)
    print("=" * 72)
    print(f"  {ruta.name}")
    print("=" * 72)
    meta = info.get("meta", {}) if isinstance(info, dict) else {}
    for k, v in meta.items():
        print(f"  {k:12s} {v}")
    e = leer_escalares(ruta)
    n = len(e["Gamma"])
    ell = float(meta.get("ell_px", 4.0)) / float(meta.get("n", 96))

    print()
    print("SERIE (uno de cada tres)")
    print(f"  {'i':>3} {'Gamma':>7} {'lam':>8} {'trD':>7} {'Kxx':>8} {'Kzz':>8} "
          f"{'R':>7} {'cum':>6} {'f_may':>7} {'stag':>5} {'perc'}")
    for i in range(0, n, 3):
        D = np.asarray(e["D"][i]); K = np.asarray(e["K"][i])
        pc = "".join("1" if b else "0" for b in np.atleast_1d(e["percola"][i]))
        print(f"  {i+1:3d} {e['Gamma'][i]:7.4f} {e['lam'][i]:8.5f} "
              f"{np.trace(D):7.4f} {K[0,0]:8.2f} {K[-1,-1]:8.2f} "
              f"{e['R'][i]:7.3f} {int(e['n_clusters'][i]):6d} "
              f"{e['f_mayor'][i]:7.4f} {int(e['stag'][i]):5d}  {pc}")

    # ---------------------------------------------------------------- #
    print()
    print("=" * 72)
    print("  EL DIAGNOSTICO: tamano de los cumulos frente a l")
    print("=" * 72)
    guardados = list(info.get("incrementos_con_campos", []))
    if not guardados:
        print("  no hay campos guardados en el archivo")
        return
    from scipy import ndimage
    resumen = []
    for idx in guardados[-3:]:
        phi = np.asarray(leer_campo(ruta, idx, "phi"), float)
        m = phi > 0.5
        lab, ncl = ndimage.label(m)
        if ncl == 0:
            print(f"  inc {idx+1}: sin fase danada")
            continue
        tam = np.bincount(lab.ravel())[1:]
        # una grieta resuelta tiene espesor ~ 2l; un cumulo con menos voxeles
        # que una bola de radio l/2 esta por debajo de la regularizacion
        h = 1.0 / phi.shape[0]
        vox_l = (ell / h)                      # l en voxeles
        minimo = (4.0 / 3.0) * np.pi * (vox_l / 2.0) ** 3
        chicos = int((tam < minimo).sum())
        # descomposicion de R
        gr = np.gradient(phi, h)
        A = float(np.mean(phi ** 2) / ell)
        B = float(ell * np.mean(sum(g ** 2 for g in gr)))
        print(f"  inc {idx+1:3d}: cumulos={ncl:5d}  "
              f"bajo l ({minimo:.0f} vox)={chicos:5d} ({100*chicos/ncl:5.1f}%)  "
              f"mayor={tam.max():7d} vox  A={A:.4e} B={B:.4e} R=B/A={B/A:.3f}")
        resumen.append(dict(inc=int(idx), n_clusters=int(ncl),
                            chicos=chicos, frac_chicos=chicos / ncl,
                            mayor=int(tam.max()), A=A, B=B, R=B / A,
                            phi_max=float(phi.max()),
                            f_danada=float(m.mean())))

    print()
    print("  COMO LEER ESTO")
    print("  - Si la mayoria de los cumulos esta BAJO l, el campo tiene")
    print("    estructura por debajo de la regularizacion: no esta resuelto,")
    print("    y los incrementos tardios no sirven. Toca subir max_stag o")
    print("    bajar dgamma cerca de la transicion.")
    print("  - Si los cumulos son grandes y aun asi R >> 1, entonces es otra")
    print("    cosa y hay que mirar el operador de gradiente.")

    sal = AQUI.parent / "diagnostico3d.json"
    sal.write_text(json.dumps({
        "archivo": ruta.name, "meta": {k: str(v) for k, v in meta.items()},
        "serie": {k: np.asarray(e[k]).tolist()
                  for k in ("Gamma", "lam", "R", "n_clusters", "f_mayor",
                            "stag", "segundos")},
        "D": np.asarray(e["D"]).tolist(), "K": np.asarray(e["K"]).tolist(),
        "percola": np.asarray(e["percola"]).tolist(),
        "campos": resumen,
    }, indent=1))
    print(f"\n-> {sal}   (chico, se puede mandar entero)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "~/porok_corridas/P1_uniaxial_n96_d3_s20260914.h5")
