"""Sensibilidad al parametro de rigidez residual eta_res: costo vs resultado.

Post-snap la grieta tiene g(phi) ~ eta_res, o sea contraste de rigidez 1/eta.
Con eta = 1e-6 eso son seis ordenes de magnitud y el CG de elasticidad se
vuelve el cuello de botella. La pregunta es si subir eta cambia el RESULTADO
o solo el costo. Si no cambia el resultado, es aceleracion gratis.
"""
import sys
import time
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from experimento_trayectorias import campo_Gc          # noqa: E402
from fftgk.phasefield import PhaseFieldSolver          # noqa: E402
from fftgk.elasticity import lame_from_E_nu            # noqa: E402
from fftgk.damage import crack_density, localization_ratio, damage_tensor  # noqa: E402
from fftgk.topology import cluster_stats               # noqa: E402
from fftgk.homogenize import homogenize                # noqa: E402

N = (128, 128)
Gc = campo_Gc(N, 20260914)
lam, mu = lame_from_E_nu(1.0, 0.2)
ell = 4.0 / 128
L = (1.0, 1.0)

print(f"{'eta':>8} {'pasos':>6} {'t[s]':>7} {'phi_max':>8} {'dens':>8} "
      f"{'R':>7} {'cum':>5} {'trD':>8} {'Kyy':>10}", flush=True)
for eta in (1e-3, 1e-4, 1e-6):
    S = PhaseFieldSolver(N, lam, mu, Gc, ell, L=L, eta_res=eta)
    e = 0.01
    t0 = time.time()
    pasos = 0
    while e < 0.105:
        S.step(np.array([[0.0, 0.0], [0.0, e]]), tol_stag=1e-5, max_stag=60)
        pasos += 1
        e *= 1.10
    dt = time.time() - t0
    D = damage_tensor(S.phi, ell, L=L)
    st = cluster_stats(S.phi > 0.5)
    k = 1.0 + (1e4 - 1.0) * np.clip(S.phi, 0, 1) ** 2
    r = homogenize(k, L=L, tol=1e-10, maxiter=2500, dual=False, ktol=1e-9)
    print(f"{eta:8.0e} {pasos:6d} {dt:7.0f} {S.phi.max():8.4f} "
          f"{crack_density(S.phi, ell, L=L):8.4f} "
          f"{localization_ratio(S.phi, ell, L=L):7.3f} {st['n_clusters']:5d} "
          f"{np.trace(D):8.4f} {r.K_primal[1,1]:10.3f}", flush=True)
