"""
EXPERIMENTO CENTRAL v2: ¿es K una funcion univaluada del tensor de dano?

Cambios respecto a v1 (ver doc 08):
  * operador de gradiente corregido a forma de divergencia (bug real)
  * subpasos ADAPTATIVOS para atravesar el snap fragil sin truncar el lazo
    escalonado -- truncarlo dejaba sin converger justamente R y el numero de
    cumulos, que son el corazon de la afirmacion
  * arranque en caliente del CG y corte por cantidad de interes
  * malla 96^2 en vez de 128^2: el mecanismo no necesita mas resolucion, y el
    costo por subpaso cerca del punto critico es de decenas de segundos

Registra un JSONL con un renglon por estado. Escribe incrementalmente, asi que
una corrida interrumpida no pierde lo ya hecho (ver doc 06 seccion 4: la cuota
de /home en Guane obliga a esta disciplina).

Uso:  python3 scripts/experimento_v2.py [nombre_trayectoria ...]
"""
import sys
import json
import time
import pathlib
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[0] / "src"))
sys.path.insert(0, str(AQUI))

from experimento_trayectorias import campo_Gc, trayectoria, medir  # noqa: E402
from fftgk.phasefield import PhaseFieldSolver                      # noqa: E402
from fftgk.elasticity import lame_from_E_nu                        # noqa: E402
from fftgk.damage import crack_density                             # noqa: E402

N = (96, 96)
ELL_PX = 4.0
L = (1.0, 1.0)
ELL = ELL_PX * L[0] / N[0]
CONTRASTE = 1e4
SEMILLA = 20260914
SALIDA = AQUI.parent / "resultados_v2.jsonl"

# (nombre, deformacion final). Las tres cubren tr eps > 0, = 0 y < 0, es decir
# tres niveles de confinamiento, que en yacimientos es la trayectoria de
# esfuerzo efectivo durante la depletacion.
TRAYECTORIAS = {
    "P1_uniaxial": 0.13,
    "P2_cizalla": 0.13,
    "P3_confinada": 0.18,
}


def correr(nombre, e_fin):
    Gc = campo_Gc(N, SEMILLA)
    lam, mu = lame_from_E_nu(1.0, 0.2)
    S = PhaseFieldSolver(N, lam, mu, Gc, ELL, L=L)
    t0 = time.time()
    n_reg = [0]

    def cb(sol, info):
        dens = crack_density(sol.phi, ELL, L=L)
        if dens < 0.02:
            return
        m = medir(sol, CONTRASTE)
        m.update(trayectoria=nombre, t=info["t"], e=float(np.abs(info["E"]).max()),
                 stag=info["stag_iters"], stop=info["stop"],
                 divisiones=info["divisiones"], segundos=round(time.time() - t0, 1))
        with open(SALIDA, "a") as f:
            f.write(json.dumps(m) + "\n")
        n_reg[0] += 1
        print(f"  {nombre:16s} t={info['t']:.3f} e={m['e']:.5f} "
              f"trD={m['trD']:7.3f} Kyy={m['Kyy']:9.2f} Kxx={m['Kxx']:8.2f} "
              f"R={m['R']:.3f} cum={m['n_clusters']:3d} chi={m['chi']:4d} "
              f"perc_y={m['percola_y']} [{m['segundos']:.0f}s]", flush=True)

    E0 = np.zeros((2, 2))
    E1 = trayectoria(nombre, e_fin)
    hechos = S.cargar_hasta(E0, E1, tol_stag=1e-5, max_stag=40,
                            callback=cb, dt_ini=0.05)
    print(f"  -> {nombre}: {len(hechos)} subpasos, {n_reg[0]} registros, "
          f"{time.time()-t0:.0f}s, ultimo stop={hechos[-1]['stop']}", flush=True)


def main():
    cuales = sys.argv[1:] or list(TRAYECTORIAS)
    print("=" * 76)
    print(f"EXPERIMENTO v2  malla {N}  l={ELL_PX}px  contraste={CONTRASTE:.0e}  "
          f"semilla={SEMILLA}")
    print("=" * 76, flush=True)
    for nombre in cuales:
        print(f"\n--- {nombre} ---", flush=True)
        correr(nombre, TRAYECTORIAS[nombre])


if __name__ == "__main__":
    main()
