"""Atravesar el punto limite con control por longitud de grieta.

Bajo control de deformacion la corrida se detenia en e ~ 0.0624 (doc 09). Si el
control por longitud funciona, deberia pasar de largo y, en el tramo inestable,
mostrar la CARGA BAJANDO mientras la grieta crece. Eso es la firma del
snap-back, y es la prueba de que el metodo hace lo que promete.
"""
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
from fftgk.damage import localization_ratio            # noqa: E402
from fftgk.topology import cluster_stats               # noqa: E402

N = (96, 96)
ELL = 4.0 / 96
L = (1.0, 1.0)
Gc = campo_Gc(N, 20260914)
lam0, mu0 = lame_from_E_nu(1.0, 0.2)

# --------------------------------------------------------------------- #
# 0. verificar la propiedad estructural que hace barato el metodo
# --------------------------------------------------------------------- #
S = PhaseFieldSolver(N, lam0, mu0, Gc, ELL, L=L)
E_dir = np.array([[0.0, 0.0], [0.0, 1.0]])
S._resolver_elasticidad(0.03 * E_dir, tol=1e-12)
p1, _, _ = psi_split(S.lam0, S.mu0, S.eps, 2)
S._resolver_elasticidad(0.09 * E_dir, tol=1e-12)
p3, _, _ = psi_split(S.lam0, S.mu0, S.eps, 2)
err = np.abs(p3 - 9.0 * p1).max() / p3.max()
print(f"[0] escalado psi+(3e) = 9 psi+(e):  error relativo = {err:.2e}")
print("    (si esto no fuera exacto, resolver la elasticidad una sola vez")
print("     por iteracion escalonada seria invalido)\n", flush=True)

# --------------------------------------------------------------------- #
# 1. recorrido con control por longitud de grieta
# --------------------------------------------------------------------- #
S = PhaseFieldSolver(N, lam0, mu0, Gc, ELL, L=L)
print(f"{'paso':>5} {'dGamma':>8} {'Gamma':>9} {'e = lam':>9} {'phi_max':>8} "
      f"{'R':>7} {'cum':>5} {'perc_y':>7} {'stag':>5} {'t[s]':>6}")
lam = 0.02
t0 = time.time()
dG = 0.01
for paso in range(1, 46):
    info = S.paso_longitud(E_dir, dG, lam_ini=lam, max_stag=30,
                           tol_stag=1e-5, tol=1e-10)
    lam = info["lam"]
    st = cluster_stats(S.phi > 0.5)
    print(f"{paso:5d} {dG:8.4f} {info['Gamma']:9.4f} {lam:9.5f} "
          f"{S.phi.max():8.4f} {localization_ratio(S.phi, ELL, L=L):7.3f} "
          f"{st['n_clusters']:5d} {str(st['percola'][1]):>7} "
          f"{info['stag_iters']:5d} {time.time()-t0:6.0f}", flush=True)
    if info["stop"] == "objetivo-inalcanzable":
        print("    objetivo inalcanzable: se detiene")
        break
    if S.phi.max() > 0.999 and st["percola"][1]:
        dG *= 1.5          # ya percolo, avanzar mas rapido
