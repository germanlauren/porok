"""Atravesar el snap con subpasos adaptativos, partiendo del estado pre-snap."""
import sys
import time
import pathlib
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[0] / "src"))
sys.path.insert(0, str(AQUI))

from diag_snap import nuevo_solver, ESTADO, ELL, L, E_PRE   # noqa: E402
from fftgk.damage import crack_density, localization_ratio  # noqa: E402
from fftgk.topology import cluster_stats                    # noqa: E402

d = np.load(ESTADO)
S = nuevo_solver()
S.phi = d["phi"].copy()
S.H = d["H"].copy()
S._et = d["et"].copy()
S._tr_pos = d["tr_pos"].copy()

E0 = np.array([[0., 0.], [0., E_PRE]])
E1 = np.array([[0., 0.], [0., 0.090]])

t0 = time.time()


def cb(sol, info):
    st = cluster_stats(sol.phi > 0.5)
    print(f"  t={info['t']:.4f} e={info['E'][1,1]:.5f} div={info['divisiones']:2d} "
          f"stag={info['stag_iters']:3d} stop={info['stop']:5s} "
          f"phi_max={sol.phi.max():.4f} "
          f"dens={crack_density(sol.phi, ELL, L=L):.4f} "
          f"R={localization_ratio(sol.phi, ELL, L=L):.3f} "
          f"cum={st['n_clusters']:3d} perc={st['percola']} "
          f"[{time.time()-t0:.0f}s]", flush=True)


hechos = S.cargar_hasta(E0, E1, tol_stag=1e-5, max_stag=40, callback=cb)
print(f"{len(hechos)} subpasos, {time.time()-t0:.0f}s, "
      f"ultimo stop={hechos[-1]['stop']}")
