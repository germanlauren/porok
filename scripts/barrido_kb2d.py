"""
Barrido 2D con apertura y referencia compatible con la ley cubica, POR TRAMOS.

Por que por tramos
------------------
El contenedor de analisis se recicla cuando la sesion se pausa, asi que un
proceso de una hora no sobrevive. Este guion guarda el estado del solver
despues de CADA incremento y retoma exactamente donde iba, de modo que se puede
llamar muchas veces seguidas con un limite de tiempo corto y el barrido igual
termina.

    python3 scripts/barrido_kb2d.py            # avanza lo que pueda
    python3 scripts/barrido_kb2d.py 100        # avanza como mucho 100 segundos

Escribe `resultados_kb2d.jsonl`, una linea por incremento, con la referencia de
Darcy (Kxx, Kyy), la referencia de apertura (Kb_xx, Kb_yy), Oda-Snow y la ley
cubica escalar. Es el archivo que alimenta el mapa de error de los cierres.
"""
import json
import pathlib
import sys
import time

import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "src"))
sys.path.insert(0, str(AQUI))

from experimento_trayectorias import medir, campo_Gc          # noqa: E402
from fftgk.phasefield import PhaseFieldSolver                 # noqa: E402
from fftgk.elasticity import lame_from_E_nu                   # noqa: E402

N = (96, 96)
ELL = 4.0 / 96
L = (1.0, 1.0)
SEMILLA = 20260914
CONTRASTE = 1e4
DGAMMA = 0.015
GAMMA_FIN = 0.63
MAX_STAG = 80

DIRS = {
    "P1_uniaxial": [(np.array([[0.0, 0.0], [0.0, 1.0]]), None)],
    "P2_cizalla": [(np.array([[-1.0, 0.0], [0.0, 1.0]]), None)],
    "P3_confinada": [(np.array([[-2.0, 0.0], [0.0, 1.0]]), None)],
    "P4_no_proporcional": [(np.array([[1.0, 0.0], [0.0, 0.0]]), 0.15),
                           (np.array([[0.0, 0.0], [0.0, 1.0]]), None)],
    "P5_excursion_grande": [(np.array([[1.0, 0.0], [0.0, 0.0]]), 0.32),
                            (np.array([[0.0, 0.0], [0.0, 1.0]]), None)],
}

CK = AQUI.parent / "ck_kb2d"
CK.mkdir(exist_ok=True)
SALIDA = AQUI.parent / "resultados_kb2d.jsonl"


def correr(nombre, limite):
    Gc = campo_Gc(N, SEMILLA)
    lam0, mu0 = lame_from_E_nu(1.0, 0.2)
    S = PhaseFieldSolver(N, lam0, mu0, Gc, ELL, L=L, esquema_phi="fd")
    lam = 0.02
    ck = CK / f"{nombre}.npz"
    if ck.exists():
        z = np.load(ck)
        S.phi = z["phi"].copy(); S.H = z["H"].copy()
        S._et = z["et"].copy(); S._tr_pos = z["tr_pos"].copy()
        S.eps = z["eps"].copy(); lam = float(z["lam"])
    if S.longitud_grieta() >= GAMMA_FIN - 1e-9:
        return "hecho"
    while S.longitud_grieta() < GAMMA_FIN:
        if time.time() > limite:
            return "tiempo"
        G = S.longitud_grieta()
        E_dir, _ = next((t for t in DIRS[nombre] if t[1] is None or G < t[1]),
                        DIRS[nombre][-1])
        info = S.paso_longitud(E_dir, DGAMMA, lam_ini=lam, max_stag=MAX_STAG,
                               tol_stag=1e-5, tol=1e-10)
        lam = info["lam"]
        m = medir(S, CONTRASTE, apertura=True)
        m.update(trayectoria=nombre, Gamma=info["Gamma"], lam=lam,
                 stag=info["stag_iters"])
        with open(SALIDA, "a") as f:
            f.write(json.dumps(m) + "\n")
        np.savez_compressed(ck, phi=S.phi, H=S.H, et=S._et,
                            tr_pos=S._tr_pos, eps=S.eps, lam=lam)
        print(f"  {nombre:20s} G={m['Gamma']:.3f} trD={m['trD']:.3f} "
              f"Kyy={m['Kyy']:8.2f} Kb=({m['Kb_xx']-1:.4f},{m['Kb_yy']-1:.4f}) "
              f"Oda={m['Lambda']*m['Kodayy']:.3e} stag={m['stag']}", flush=True)
    return "hecho"


def main(segundos=100.0):
    limite = time.time() + float(segundos)
    estado = {}
    for nombre in DIRS:
        estado[nombre] = correr(nombre, limite)
        if estado[nombre] == "tiempo":
            break
    hechas = [k for k, v in estado.items() if v == "hecho"]
    print(f"\ncompletas: {len(hechas)}/5  {hechas}")
    if SALIDA.exists():
        print(f"{sum(1 for _ in open(SALIDA))} estados en {SALIDA.name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 100.0)
