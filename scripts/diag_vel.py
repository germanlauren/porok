"""Costo por paso con el nuevo criterio de parada del lazo escalonado."""
import sys
import time
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from experimento_trayectorias import campo_Gc          # noqa: E402
from fftgk.phasefield import PhaseFieldSolver          # noqa: E402
from fftgk.elasticity import lame_from_E_nu            # noqa: E402
from fftgk.damage import crack_density, localization_ratio  # noqa: E402
from fftgk.topology import cluster_stats               # noqa: E402

N = (128, 128)
Gc = campo_Gc(N, 20260914)
lam, mu = lame_from_E_nu(1.0, 0.2)
ell = 4.0 / 128
S = PhaseFieldSolver(N, lam, mu, Gc, ell, L=(1.0, 1.0))
e = 0.01
total = 0.0
for i in range(30):
    t = time.time()
    d = S.step(np.array([[0., 0.], [0., e]]), tol_stag=1e-5, max_stag=60,
               qoi_tol=1e-6)
    dt = time.time() - t
    total += dt
    st = cluster_stats(S.phi > 0.5)
    print(f"  e={e:.4f} phi_max={S.phi.max():.4f} "
          f"dens={crack_density(S.phi, ell, L=(1., 1.)):.4f} "
          f"R={localization_ratio(S.phi, ell, L=(1., 1.)):.3f} "
          f"cum={st['n_clusters']:3d} perc={st['percola']} "
          f"stag={d['stag_iters']:3d} stop={d['stop']:6s} t={dt:6.1f}s",
          flush=True)
    e *= 1.10
    if e > 0.16:
        break
print(f"total {total:.0f}s")
