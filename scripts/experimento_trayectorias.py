"""
EXPERIMENTO CENTRAL: ¿es K una funcion univaluada del tensor de dano?

Diseno
------
En vez de intentar construir dos trayectorias que lleguen al MISMO D_ij -- que
exige adivinar de antemano el mecanismo -- se hace lo robusto: se corren varias
trayectorias de carga sobre LA MISMA microestructura, y en cada paso se registra
el par (D_ij, K). Despues se mira si los puntos caen sobre una sola curva.

    si K = f(D) colapsan todas las trayectorias sobre una curva  -> hipotesis muerta
    si la nube se abre                                            -> hipotesis viva

Esto responde la pregunta directamente y no depende de que el mecanismo que
imaginamos sea el correcto.

Trayectorias
------------
Todas comparten la microestructura (mismo campo Gc, misma semilla): esa es la
variable de control. Lo que cambia es el CAMINO en el espacio de deformaciones.

  P1 traccion uniaxial   eps = diag(0, e)        tr > 0
  P2 cizalla pura        eps = diag(-e, e)       tr = 0
  P3 confinada           eps = diag(-2e, e)      tr < 0
  P4 no proporcional     excursion biaxial en x y luego traccion en y

La variable fisica detras de P1-P3 es el CONFINAMIENTO, que en yacimientos es
justamente la trayectoria de esfuerzo efectivo durante la depletacion. P4 rota
la direccion de carga a mitad de camino, que es la definicion de carga no
proporcional.

Con la separacion de Amor, el confinamiento apaga la parte volumetrica de la
fuerza motriz (<tr eps>_+ = 0 en compresion) y deja solo la desviatorica. Eso
cambia el equilibrio entre una fuerza motriz isotropa y una que selecciona
orientacion -- que es justo lo que puede decidir entre localizar y fragmentar.

Control de parada
-----------------
Cada trayectoria se carga hasta alcanzar la MISMA densidad de grieta objetivo,
no la misma deformacion. Asi la comparacion es a igual "cantidad de dano", que
es lo que un cierre basado en estado usaria.

Alcance
-------
Nivel Darcy para K (ver piloto 0, doc 05): basta para saber si el efecto existe.
Stokes-Brinkman y la apertura fisica son para que el numero sea publicable.
"""

import os
import sys
import json
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fftgk.phasefield import PhaseFieldSolver                      # noqa: E402
from fftgk.elasticity import lame_from_E_nu                        # noqa: E402
from fftgk.damage import damage_tensor, crack_density, localization_ratio  # noqa: E402
from fftgk.topology import cluster_stats                           # noqa: E402
from fftgk.homogenize import homogenize                            # noqa: E402


# --------------------------------------------------------------------------- #
# Microestructura
# --------------------------------------------------------------------------- #

def campo_Gc(N, semilla, Gc_medio=2e-3, dispersion=0.15, corr_px=6.0,
             n_fallas=40, largo=0.10, debilidad=0.12, ancho_px=2.0):
    """Tenacidad heterogenea CON MICROFISURAS PREEXISTENTES.

    Primera version de este experimento uso solo un campo lognormal suave. No
    funciono, y el fallo fue informativo: la razon de localizacion R se quedo en
    0.016 y chi en 0, es decir, el dano crecio DIFUSO y nunca se formaron
    grietas. Con heterogeneidad suave, AT2 bajo control de deformacion da dano
    repartido, no fractura.

    Eso importa mas alla de lo numerico: un campo de fase difuso produce un
    D_ij que NO es densidad de grieta, y la bandera R (doc 04 seccion 5) lo
    detecta automaticamente. Es un resultado a reportar: los cierres basados en
    estado usan una variable de dano que confunde dano difuso con grietas
    localizadas, y esas dos cosas tienen firmas de permeabilidad opuestas.

    Para llegar al regimen de grietas se siembran microfisuras: segmentos
    cortos, de posicion y orientacion aleatorias, donde Gc esta muy reducido.
    Es ademas lo fisicamente correcto -- la roca tiene microfisuras
    preexistentes, y la conectividad emerge de CUALES se propagan y se enlazan.
    Que la trayectoria de carga elija ese subconjunto es justamente el mecanismo
    que el experimento quiere poner a prueba.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(semilla)
    g = gaussian_filter(rng.standard_normal(N), corr_px, mode="wrap")
    g = (g - g.mean()) / g.std()
    Gc = Gc_medio * np.exp(dispersion * g)

    # microfisuras: distancia a segmentos cortos, con envoltura periodica
    x = (np.arange(N[0]) + 0.5) / N[0]
    y = (np.arange(N[1]) + 0.5) / N[1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    w = ancho_px / N[0]
    for _ in range(n_fallas):
        cx, cy = rng.random(2)
        th = rng.random() * np.pi
        t = np.array([np.cos(th), np.sin(th)])
        dx = X - cx
        dx -= np.round(dx)
        dy = Y - cy
        dy -= np.round(dy)
        # distancia al segmento centrado en (cx,cy), direccion t, longitud `largo`
        s = dx * t[0] + dy * t[1]
        s = np.clip(s, -largo / 2, largo / 2)
        d = np.hypot(dx - s * t[0], dy - s * t[1])
        Gc = np.minimum(Gc, np.where(d < w, Gc_medio * debilidad, Gc))
    return Gc


# --------------------------------------------------------------------------- #
# Trayectorias de carga
# --------------------------------------------------------------------------- #

def trayectoria(nombre, e):
    """Devuelve el tensor de deformacion macroscopica para el parametro e."""
    if nombre == "P1_uniaxial":
        return np.array([[0.0, 0.0], [0.0, e]])
    if nombre == "P2_cizalla":
        return np.array([[-e, 0.0], [0.0, e]])
    if nombre == "P3_confinada":
        return np.array([[-2.0 * e, 0.0], [0.0, e]])
    raise ValueError(nombre)


def trayectoria_no_proporcional(e, e_exc):
    """P4: primero una excursion de traccion en x, luego traccion en y.

    Se implementa como una lista de estados; el llamador la recorre en orden.
    El punto es que el dano acumulado en la excursion NO se borra (el campo de
    historia es un maximo), asi que sesga donde nuclean las grietas siguientes.
    """
    pasos = []
    for a in np.linspace(0.0, e_exc, 8):
        pasos.append(np.array([[a, 0.0], [0.0, 0.0]]))
    for a in np.linspace(e_exc, 0.0, 5)[1:]:
        pasos.append(np.array([[a, 0.0], [0.0, 0.0]]))
    for a in np.linspace(0.0, e, 30)[1:]:
        pasos.append(np.array([[0.0, 0.0], [0.0, a]]))
    return pasos


# --------------------------------------------------------------------------- #
# Medicion en un estado
# --------------------------------------------------------------------------- #

def medir_apertura(S):
    """Momentos pesados por apertura y el cierre de Oda-Snow.

    Hasta ahora no se reportaban porque la extraccion de apertura no estaba
    verificada (doc 22). Ya lo esta: el volumen de grieta converge a Sneddon
    con error 1.8 (l/a) y extrapola a 1.003, y el momento de b^3 -- que es el
    que pesa la ley cubica -- queda dentro del 5% para l/a <= 0.12.

    Devuelve tambien <b> = V_c/Gamma, que es lo que alimenta la ley cubica
    escalar.
    """
    from fftgk.apertura import (volumen_grieta, campo_apertura,
                                campo_conductividad)
    from fftgk.cierres import oda_desde_D3
    LAMBDA = float(os.environ.get("POROK_LAMBDA", "1e13"))
    d = S.d
    eps_f = S.eps - S.eps.reshape(d, d, -1).mean(axis=2).reshape(
        (d, d) + (1,) * d)
    u = S.P.desplazamiento(eps_f)
    Vc = volumen_grieta(u, S.phi, L=S.L, projector=S.base_phi)
    b = campo_apertura(u, S.phi, S.ell, L=S.L, projector=S.base_phi)
    Gam = crack_density(S.phi, S.ell, L=S.L)
    b_med = Vc / Gam if Gam > 0 else 0.0
    D1 = damage_tensor(S.phi, S.ell, L=S.L, weight=b, m=1)
    D3 = damage_tensor(S.phi, S.ell, L=S.L, weight=b, m=3)
    Koda = oda_desde_D3(D3)
    fuera = dict(Vc=float(Vc), b_medio=float(b_med), b_max=float(b.max()),
                 K_cubica=float(Gam * b_med ** 3 / 12.0), Lambda=LAMBDA)

    # ---- referencia COMPATIBLE con la ley cubica -------------------------
    # Comparar Oda-Snow contra nuestra K de Darcy es comparar dos modelos de
    # flujo distintos: Oda supone que TODO el caudal va por la fractura abierta
    # con conductancia b^3/12, mientras que la referencia deja conducir tambien
    # a la matriz danada. Para ponerlos en igualdad de condiciones se resuelve
    # una SEGUNDA homogeneizacion sobre el mismo campo geometrico, pero con la
    # conductividad local dada por la propia apertura:
    #
    #     k_b(x) = k_matriz + (b(x)^3 / 12) / w,    w = 2l  (espesor de banda)
    #
    # En unidades fisicas eso es  k_b/k_m = 1 + [b^3/(12 w)] * LAMBDA  con el
    # grupo adimensional LAMBDA = L^2 / k_m (tamano de celda al cuadrado sobre
    # permeabilidad de matriz). Con L = 10 cm y k_m = 1 mD = 1e-15 m^2 sale
    # LAMBDA = 1e13, y a grieta formada (b ~ 1e-3 L) da un contraste ~1e4:
    # exactamente el contraste que la referencia de Darcy usa desde el doc 05,
    # lo cual no estaba puesto a proposito y es una comprobacion de escala.
    #
    # que es la traduccion estandar de fractura a medio poroso equivalente:
    # la transmisividad b^3/12 repartida en el espesor de banda que ocupa la
    # grieta regularizada. Oda-Snow es la aproximacion analitica de ESTE
    # problema (fracturas rectas, sin interaccion), asi que su error contra
    # esta referencia si es atribuible al cierre y no al modelo de flujo.
    kb = campo_conductividad(b, S.ell, LAMBDA)
    rb = homogenize(kb, L=S.L, tol=1e-10, maxiter=2500, dual=False, ktol=1e-9)
    Kb = rb.K_primal
    fuera["Kb_xx"] = float(Kb[0, 0])
    fuera["Kb_yy"] = float(Kb[1, 1])
    fuera["Kb_xy"] = float(Kb[0, 1])
    for nom, T in (("D1", D1), ("D3", D3), ("Koda", Koda)):
        fuera[nom + "xx"] = float(T[0, 0])
        fuera[nom + "yy"] = float(T[1, 1])
        fuera[nom + "xy"] = float(T[0, 1])
    return fuera


def medir(S, contraste, k_m=1.0, apertura=False):
    """Todas las cantidades de interes en el estado actual del solver."""
    D = damage_tensor(S.phi, S.ell, L=S.L)
    R = localization_ratio(S.phi, S.ell, L=S.L)
    st = cluster_stats(S.phi > 0.5)
    k = k_m + (k_m * contraste - k_m) * np.clip(S.phi, 0, 1) ** 2
    r = homogenize(k, L=S.L, tol=1e-10, maxiter=2500, dual=False, ktol=1e-9)
    K = r.K_primal
    return {
        "trD": float(np.trace(D)),
        "Dxx": float(D[0, 0]), "Dyy": float(D[1, 1]), "Dxy": float(D[0, 1]),
        "R": float(R),
        "n_clusters": st["n_clusters"],
        "f_mayor": float(st["f_mayor"]),
        "chi": int(st["chi"]),
        "percola_x": bool(st["percola"][0]),
        "percola_y": bool(st["percola"][1]),
        "Kxx": float(K[0, 0]), "Kyy": float(K[1, 1]), "Kxy": float(K[0, 1]),
        "phi_max": float(S.phi.max()),
        **(medir_apertura(S) if apertura else {}),
    }


# --------------------------------------------------------------------------- #

def correr(nombre, N, Gc, ell_px, contraste, trD_objetivo, e_max,
           n_pasos=60, salida=None):
    lam, mu = lame_from_E_nu(1.0, 0.2)
    L = (1.0, 1.0)
    ell = ell_px * L[0] / N[0]
    S = PhaseFieldSolver(N, lam, mu, Gc, ell, L=L)

    # Rampa geometrica fina. El sistema se rompe de golpe (snap fragil): pasado
    # el umbral, phi salta de ~0.3 a ~1 en un solo paso. Con pasos gruesos ese
    # salto se atraviesa sin resolverlo; con pasos finos se sigue la transicion.
    if nombre == "P4_no_proporcional":
        pasos = trayectoria_no_proporcional(e_max, 0.6 * e_max)
    else:
        es = 0.01 * 1.10 ** np.arange(n_pasos)
        pasos = [trayectoria(nombre, e) for e in es if e <= e_max]
    registros = []
    for i, E in enumerate(pasos):
        S.step(E, tol_stag=1e-5, max_stag=60)
        dens = crack_density(S.phi, ell, L=L)
        if dens < 0.05:
            continue                      # aun sin dano medible
        m = medir(S, contraste)
        m["paso"] = i
        m["e"] = float(np.abs(E).max())
        m["trayectoria"] = nombre
        registros.append(m)
        print(f"  {nombre:20s} paso {i:3d}  trD={m['trD']:7.3f}  "
              f"Kyy={m['Kyy']:10.3f}  chi={m['chi']:5d}  "
              f"cum={m['n_clusters']:3d}  R={m['R']:.3f}  "
              f"perc_y={m['percola_y']}", flush=True)
        if salida is not None:
            with open(salida, "a") as f:
                f.write(json.dumps(m) + "\n")
        if m["trD"] >= trD_objetivo:
            break
    return registros


def main():
    N = (128, 128)
    ell_px = 4.0
    contraste = 1e4
    trD_objetivo = 4.0
    semilla = 20260914
    Gc = campo_Gc(N, semilla)
    salida = pathlib.Path("resultados_trayectorias.jsonl")
    if salida.exists():
        salida.unlink()

    print("=" * 78)
    print("EXPERIMENTO: K como funcion de D_ij a lo largo de trayectorias")
    print(f"malla {N}, l={ell_px}px, contraste k={contraste:.0e}, "
          f"microestructura semilla {semilla}")
    print("=" * 78)

    todos = []
    for nombre, e_max in (("P1_uniaxial", 0.30),
                          ("P2_cizalla", 0.30),
                          ("P3_confinada", 0.40),
                          ("P4_no_proporcional", 0.20)):
        print(f"\n--- {nombre} ---", flush=True)
        todos += correr(nombre, N, Gc, ell_px, contraste, trD_objetivo,
                        e_max, salida=salida)

    print(f"\n{len(todos)} estados registrados en {salida}")


if __name__ == "__main__":
    main()
