"""¿Cuanto max_stag hace falta REALMENTE en el paso del snap?

El lazo escalonado se estanca pasado el snap fragil y quema todas las
iteraciones. Pero lo que este proyecto mide son medias volumetricas: D_ij y K.
Si truncar el lazo no las mueve, truncar es aceptable y el muro desaparece.

Se guarda el estado justo antes del snap y se repite el mismo paso con distintos
max_stag, comparando D y K. Es la unica forma honesta de decidirlo: medir.

Uso:  python3 scripts/diag_snap.py preparar
      python3 scripts/diag_snap.py snap <max_stag>
"""
import sys
import time
import pathlib
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[0] / "src"))
sys.path.insert(0, str(AQUI))

from experimento_trayectorias import campo_Gc          # noqa: E402
from fftgk.phasefield import PhaseFieldSolver          # noqa: E402
from fftgk.elasticity import lame_from_E_nu            # noqa: E402
from fftgk.damage import damage_tensor, crack_density, localization_ratio  # noqa: E402
from fftgk.topology import cluster_stats               # noqa: E402
from fftgk.homogenize import homogenize                # noqa: E402

N = (128, 128)
ELL = 4.0 / 128
L = (1.0, 1.0)
E_PRE = 0.0612          # justo antes del snap (medido)
E_SNAP = 0.0673         # el paso que rompe
ESTADO = AQUI.parent / "estado_presnap.npz"


def nuevo_solver():
    Gc = campo_Gc(N, 20260914)
    lam, mu = lame_from_E_nu(1.0, 0.2)
    return PhaseFieldSolver(N, lam, mu, Gc, ELL, L=L)


def preparar():
    S = nuevo_solver()
    e = 0.03
    t0 = time.time()
    while e < E_PRE:
        S.step(np.array([[0., 0.], [0., e]]), tol_stag=1e-5, max_stag=60)
        e *= 1.12
    S.step(np.array([[0., 0.], [0., E_PRE]]), tol_stag=1e-5, max_stag=60)
    np.savez_compressed(ESTADO, phi=S.phi, H=S.H, et=S._et,
                        tr_pos=S._tr_pos)
    print(f"estado guardado: phi_max={S.phi.max():.4f} "
          f"dens={crack_density(S.phi, ELL, L=L):.4f} "
          f"R={localization_ratio(S.phi, ELL, L=L):.3f} "
          f"({time.time()-t0:.0f}s)")


def snap(max_stag, e_obj=None):
    d = np.load(ESTADO)
    S = nuevo_solver()
    S.phi = d["phi"].copy()
    S.H = d["H"].copy()
    S._et = d["et"].copy()
    S._tr_pos = d["tr_pos"].copy()
    t0 = time.time()
    eo = E_SNAP if e_obj is None else e_obj
    info = S.step(np.array([[0., 0.], [0., eo]]), tol_stag=1e-5,
                  max_stag=max_stag, qoi_tol=1e-6)
    dt = time.time() - t0
    D = damage_tensor(S.phi, ELL, L=L)
    st = cluster_stats(S.phi > 0.5)
    k = 1.0 + (1e4 - 1.0) * np.clip(S.phi, 0, 1) ** 2
    r = homogenize(k, L=L, tol=1e-10, maxiter=2500, dual=False, ktol=1e-9)
    print(f"e={eo:.5f} max_stag={max_stag:3d} usadas={info['stag_iters']:3d} "
          f"stop={info['stop']:6s} dphi={info['dphi']:.2e} t={dt:6.1f}s | "
          f"trD={np.trace(D):8.5f} Dyy={D[1,1]:8.5f} "
          f"Kyy={r.K_primal[1,1]:10.4f} Kxx={r.K_primal[0,0]:8.4f} "
          f"cum={st['n_clusters']} R={localization_ratio(S.phi, ELL, L=L):.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "preparar":
        preparar()
    else:
        snap(int(sys.argv[2]),
             float(sys.argv[3]) if len(sys.argv) > 3 else None)
