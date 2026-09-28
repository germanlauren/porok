"""
EXPERIMENTO CENTRAL: ¿es K una funcion univaluada del tensor de dano?

Ahora con control por longitud de grieta (doc 10), que es lo que permite
recorrer la transicion completa en vez de detenerse en el punto limite.

Diseno
------
Tres trayectorias de carga sobre LA MISMA microestructura (mismo campo Gc,
misma semilla). Lo unico que cambia es el CAMINO en el espacio de
deformaciones. En cada incremento de longitud de grieta se registra el par
(D_ij, K) junto con los descriptores topologicos.

    si los puntos colapsan sobre una curva   -> K = f(D), hipotesis muerta
    si la nube se abre                       -> hipotesis viva

  P1 uniaxial    eps = diag(0, e)     tr > 0
  P2 cizalla     eps = diag(-e, e)    tr = 0
  P3 confinada   eps = diag(-2e, e)   tr < 0

La variable fisica detras es el CONFINAMIENTO, que en yacimientos es la
trayectoria de esfuerzo efectivo durante la depletacion. Con la separacion de
Amor el confinamiento apaga la parte volumetrica de la fuerza motriz y deja
solo la desviatorica, cambiando el equilibrio entre una fuerza motriz isotropa
y una que selecciona orientacion.

Verificacion de convergencia, gratis
------------------------------------
max_stag = 80 aqui, frente a 30 en la corrida del doc 10. Comparando la curva
lambda(Gamma) de P1 entre las dos se ve si la rama descendente estaba
convergida o solo era cualitativamente correcta. No cuesta nada extra.

Escritura incremental a JSONL: una corrida interrumpida no pierde lo hecho.
"""
import sys
import json
import time
import pathlib
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[0] / "src"))
sys.path.insert(0, str(AQUI))

from experimento_trayectorias import campo_Gc, medir    # noqa: E402
from fftgk.phasefield import PhaseFieldSolver           # noqa: E402
from fftgk.elasticity import lame_from_E_nu             # noqa: E402

N = (96, 96)
ELL = 4.0 / 96
L = (1.0, 1.0)
CONTRASTE = 1e4
SEMILLA = 20260914
MAX_STAG = 80
GAMMA_FIN = 0.62
DGAMMA = 0.015
# Operador del campo de fase (doc 18). El barrido PUBLICADO se hizo con
# "rotated", que en 3D tiene modos sin gradiente; el correcto es "fd".
#
# Los archivos se separan por esquema A PROPOSITO: relanzar este script con el
# operador nuevo sobre los nombres viejos agregaria filas al archivo publicado
# y REANUDARIA desde checkpoints calculados con el operador viejo, mezclando las
# dos cosas sin ningun aviso. Asi, "rotated" sigue escribiendo exactamente donde
# siempre (y reproduce el barrido publicado bit a bit), y "fd" va aparte.
import os
ESQUEMA_PHI = os.environ.get("POROK_ESQUEMA_PHI", "fd")
# Momentos pesados por apertura y Oda-Snow: solo desde que la extraccion de
# apertura esta verificada contra Sneddon (doc 22). Van a un archivo aparte
# para no mezclarlos con el barrido publicado.
APERTURA = os.environ.get("POROK_APERTURA", "0") == "1"
if ESQUEMA_PHI == "rotated":
    SALIDA = AQUI.parent / "resultados_final.jsonl"
    CHECK = AQUI.parent / "checkpoints"
else:
    suf = "_ap" if APERTURA else ""
    SALIDA = AQUI.parent / f"resultados_{ESQUEMA_PHI}{suf}.jsonl"
    CHECK = AQUI.parent / f"checkpoints_{ESQUEMA_PHI}{suf}"
CHECK.mkdir(exist_ok=True)

DIRECCIONES = {
    "P1_uniaxial": np.array([[0.0, 0.0], [0.0, 1.0]]),
    "P2_cizalla": np.array([[-1.0, 0.0], [0.0, 1.0]]),
    "P3_confinada": np.array([[-2.0, 0.0], [0.0, 1.0]]),
}

# Trayectorias NO PROPORCIONALES: la direccion de carga cambia a mitad de
# camino. Es el caso que el doc 02 identifico como el corazon de la afirmacion
# y que faltaba correr.
#
# P4 carga primero en x y despues en y. Termina cargando en la MISMA direccion
# que P1 y llega al mismo Gamma, pero con una historia distinta. Como el campo
# de historia H es un maximo, el dano de la excursion NO se borra: sesga que
# microfisuras estan ya activadas cuando empieza el tramo en y.
#
# Esto es lo que aisla la dependencia de trayectoria de la dependencia de
# direccion: P1 y P4 comparten direccion final, asi que si difieren no puede
# atribuirse a que "las cargaron distinto al final".
TRAMOS = {
    "P4_no_proporcional": [
        (np.array([[1.0, 0.0], [0.0, 0.0]]), 0.15),   # traccion en x hasta G=0.15
        (np.array([[0.0, 0.0], [0.0, 1.0]]), None),   # luego traccion en y
    ],
    # P5: la MISMA idea con una excursion mucho mayor.
    #
    # P4 resulto INDISTINGUIBLE de P1 (doc 13). La explicacion es estructural:
    # el campo de historia de Miehe es un MAXIMO, H = max_s psi+(s). Una
    # excursion cuya energia motriz es menor que la de la carga posterior queda
    # simplemente sobreescrita, y no deja rastro. P5 prueba ese mecanismo:
    # si la excursion domina en magnitud, el maximo no puede borrarla.
    "P5_excursion_grande": [
        (np.array([[1.0, 0.0], [0.0, 0.0]]), 0.32),   # traccion en x hasta G=0.32
        (np.array([[0.0, 0.0], [0.0, 1.0]]), None),   # luego traccion en y
    ],
}


def correr(nombre, gamma_fin=GAMMA_FIN):
    """Recorre una trayectoria. REANUDABLE: guarda el estado tras cada
    incremento, asi que un reinicio del contenedor no cuesta nada. Con corridas
    de horas y contenedores que se reciclan, esto no es comodidad sino
    condicion para que el experimento termine alguna vez."""
    tramos = TRAMOS.get(nombre) or [(DIRECCIONES[nombre], None)]
    Gc = campo_Gc(N, SEMILLA)
    lam0, mu0 = lame_from_E_nu(1.0, 0.2)
    S = PhaseFieldSolver(N, lam0, mu0, Gc, ELL, L=L, esquema_phi=ESQUEMA_PHI)
    lam = 0.02
    ck = CHECK / f"{nombre}.npz"
    if ck.exists():
        d = np.load(ck)
        S.phi = d["phi"].copy(); S.H = d["H"].copy()
        S._et = d["et"].copy(); S._tr_pos = d["tr_pos"].copy()
        S.eps = d["eps"].copy(); lam = float(d["lam"])
        print(f"  [reanudado desde Gamma={S.longitud_grieta():.4f}, "
              f"lam={lam:.5f}]", flush=True)
    t0 = time.time()
    n = 0
    while S.longitud_grieta() < gamma_fin and n < 60:
        G_actual = S.longitud_grieta()
        E_dir, _ = next((t for t in tramos
                         if t[1] is None or G_actual < t[1]), tramos[-1])
        info = S.paso_longitud(E_dir, DGAMMA, lam_ini=lam, max_stag=MAX_STAG,
                               tol_stag=1e-5, tol=1e-10)
        lam = info["lam"]
        n += 1
        m = medir(S, CONTRASTE, apertura=APERTURA)
        m.update(trayectoria=nombre, Gamma=info["Gamma"], lam=lam,
                 stag=info["stag_iters"], stop=info["stop"],
                 segundos=round(time.time() - t0, 1))
        with open(SALIDA, "a") as f:
            f.write(json.dumps(m) + "\n")
        print(f"  {nombre:14s} G={m['Gamma']:.4f} lam={lam:.5f} "
              f"trD={m['trD']:6.3f} Dyy/trD={m['Dyy']/max(m['trD'],1e-30):.3f} "
              f"Kyy={m['Kyy']:9.2f} Kxx={m['Kxx']:8.2f} R={m['R']:.3f} "
              f"cum={m['n_clusters']:3d} chi={m['chi']:4d} "
              f"perc=({m['percola_x']:d},{m['percola_y']:d}) "
              f"stag={m['stag']:3d} [{m['segundos']:.0f}s]", flush=True)
        np.savez_compressed(ck, phi=S.phi, H=S.H, et=S._et,
                            tr_pos=S._tr_pos, eps=S.eps, lam=lam)
        if info["stop"] == "objetivo-inalcanzable":
            break
    print(f"  -> {nombre}: {n} incrementos, {time.time()-t0:.0f}s", flush=True)


def main():
    args = sys.argv[1:]
    gamma_fin = GAMMA_FIN
    if args and args[0].startswith("G="):
        gamma_fin = float(args[0][2:]); args = args[1:]
    cuales = args or (list(DIRECCIONES) + list(TRAMOS))
    print("=" * 78)
    print(f"operador del campo de fase: {ESQUEMA_PHI}   ->  {SALIDA.name}")
    print(f"EXPERIMENTO FINAL  malla {N}  l=4px  contraste={CONTRASTE:.0e}  "
          f"semilla={SEMILLA}  max_stag={MAX_STAG}")
    print("control por longitud de grieta, dGamma=%.3f hasta Gamma=%.2f"
          % (DGAMMA, GAMMA_FIN))
    print("=" * 78, flush=True)
    for nombre in cuales:
        print(f"\n--- {nombre} ---", flush=True)
        correr(nombre, gamma_fin)


if __name__ == "__main__":
    main()
