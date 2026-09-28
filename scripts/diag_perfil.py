"""Donde se va el tiempo en un paso post-snap, y como escala con la malla."""
import sys
import time
import pathlib
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[0] / "src"))
sys.path.insert(0, str(AQUI))

from experimento_trayectorias import campo_Gc          # noqa: E402
from fftgk.phasefield import PhaseFieldSolver, psi_split  # noqa: E402
from fftgk.elasticity import lame_from_E_nu            # noqa: E402
from fftgk.damage import crack_density                 # noqa: E402

lam, mu = lame_from_E_nu(1.0, 0.2)
L = (1.0, 1.0)

print(f"{'N':>5} {'pasos':>6} {'t_total':>8} {'t/paso':>8} {'stag/paso':>10} "
      f"{'t_elas':>8} {'t_phi':>7} {'cg_elas':>8} {'cg_phi':>7} {'dens':>7}")
for n in (64, 96, 128):
    N = (n, n)
    Gc = campo_Gc(N, 20260914)
    ell = 4.0 / n                      # mismo l/h, l escala con la malla
    S = PhaseFieldSolver(N, lam, mu, Gc, ell, L=L)
    e = 0.03
    t0 = time.time()
    pasos = 0
    stag_tot = 0
    t_el = t_ph = 0.0
    cg_el = cg_ph = 0
    while e < 0.066:
        # instrumentar un paso a mano para separar los dos costos
        for _ in range(40):
            t = time.time()
            its, info_e = S._resolver_elasticidad(
                np.array([[0., 0.], [0., e]]), tol=1e-10)
            t_el += time.time() - t
            cg_el += info_e["iters"]
            pp, _, _ = psi_split(S.lam0, S.mu0, S.eps, 2)
            S.H = np.maximum(S.H, pp)
            t = time.time()
            p, info_p = S._resolver_phi(tol=1e-10)
            t_ph += time.time() - t
            cg_ph += info_p.get("iters", 0)
            dphi = float(np.abs(p - S.phi).max())
            S.phi = np.maximum(S.phi, p)
            stag_tot += 1
            if dphi < 1e-5:
                break
        pasos += 1
        e *= 1.12
    dt = time.time() - t0
    print(f"{n:5d} {pasos:6d} {dt:8.1f} {dt/pasos:8.2f} {stag_tot/pasos:10.1f} "
          f"{t_el:8.1f} {t_ph:7.1f} {cg_el:8d} {cg_ph:7d} "
          f"{crack_density(S.phi, ell, L=L):7.4f}", flush=True)
