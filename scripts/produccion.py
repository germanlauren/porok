"""
Corredor de produccion para HPC. 2D o 3D, salida HDF5, reanudable.

Uso
---
    python3 scripts/produccion.py --trayectoria P1_uniaxial --n 128 --dim 2
    python3 scripts/produccion.py --trayectoria P1_uniaxial --n 96  --dim 3 \
        --gamma-fin 0.8 --salida /scratch/corridas

Cada trayectoria escribe un HDF5 con los escalares de toda la serie y los
campos cada `--campos-cada` incrementos. Ver src/fftgk/almacen.py para la
estructura y el porque de las decisiones de formato.

Reanudable: si existe el checkpoint de esa trayectoria, continua desde ahi.
Pensado para colas con limite de tiempo: se relanza el mismo comando y sigue.
"""
import sys
import time
import argparse
import pathlib
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[0] / "src"))
sys.path.insert(0, str(AQUI))

from fftgk.phasefield import PhaseFieldSolver              # noqa: E402
from fftgk.elasticity import lame_from_E_nu                # noqa: E402
from fftgk.damage import damage_tensor, localization_ratio  # noqa: E402
from fftgk.topology import cluster_stats                   # noqa: E402
from fftgk.homogenize import homogenize                    # noqa: E402
from fftgk.almacen import Almacen                          # noqa: E402


# Cobertura de semillas = largo / separacion media. Es el numero adimensional
# que decide si las microfisuras pueden enlazarse o no, y el unico que hay que
# mantener igual al pasar de 2D a 3D.
#
# El valor 0.6325 reproduce EXACTAMENTE las 40 semillas del barrido 2D
# publicado: (0.6325/0.10)^2 = 40.0. En 3D la misma cobertura pide 253.
COBERTURA = 0.10 * np.sqrt(40.0)


def campo_Gc(N, semilla, Gc_medio=2e-3, dispersion=0.15, corr_px=6.0,
             n_fallas=None, largo=0.10, debilidad=0.12, ancho_px=2.0,
             cobertura=COBERTURA):
    """Tenacidad heterogenea con microfisuras preexistentes. Funciona en 2D y 3D.

    En 3D las microfisuras son discos: se siembra un segmento y se toma la
    distancia al plano perpendicular a una normal aleatoria, acotada a un radio.

    CUANTAS SEMILLAS, Y POR QUE NO 40 EN AMBAS DIMENSIONES
    ------------------------------------------------------
    Sembrar el mismo NUMERO de microfisuras en 2D y en 3D parece lo natural y
    es justamente el error. La separacion media entre semillas va como
    (|Y|/n)^(1/d), asi que con n = 40:

        2D: separacion 0.158 L, largo 0.100 L  ->  largo/sep = 0.63
        3D: separacion 0.292 L, largo 0.100 L  ->  largo/sep = 0.34

    En 2D las semillas casi se tocan y la grieta las enlaza; en 3D quedan a
    triple distancia de su propio tamano y cada disco queda aislado. El piloto
    3D a 96^3 lo mostro sin lugar a dudas: el dano nunca localizo, termino en
    2847 cumulos de los cuales el 99.9% eran mas chicos que l, el mayor tenia
    88 voxeles de 884736, y la razon entre el gradiente espectral y el de
    diferencias finitas subio a 36 (en 2D es 2.0 y constante). Es decir, un
    campo difuso con ruido de malla, no una red de grietas.

    Es el mismo modo de falla que tuvo la PRIMERA version 2D antes de sembrar
    microfisuras, y por eso se sembraron.

    La solucion es fijar la COBERTURA -- largo dividido separacion -- en vez
    del numero, con lo que n sale de la dimension:

        n = (cobertura / largo)^d

    Con la cobertura por defecto eso da 40 en 2D (identico al barrido
    publicado, bit a bit) y 253 en 3D. Pasar `n_fallas` explicitamente sigue
    funcionando y tiene prioridad.
    """
    from scipy.ndimage import gaussian_filter
    d = len(N)
    if n_fallas is None:
        n_fallas = int(round((cobertura / largo) ** d))
    rng = np.random.default_rng(semilla)
    g = gaussian_filter(rng.standard_normal(N), corr_px, mode="wrap")
    g = (g - g.mean()) / g.std()
    Gc = Gc_medio * np.exp(dispersion * g)

    ejes = [(np.arange(n) + 0.5) / n for n in N]
    X = np.meshgrid(*ejes, indexing="ij")
    w = ancho_px / N[0]
    for _ in range(n_fallas):
        c = rng.random(d)
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        dx = [X[i] - c[i] for i in range(d)]
        dx = [a - np.round(a) for a in dx]              # periodico
        s = sum(dx[i] * v[i] for i in range(d))         # distancia al plano
        r2 = sum(dx[i] ** 2 for i in range(d)) - s ** 2  # radio en el plano
        dentro = np.sqrt(np.maximum(r2, 0.0)) < largo / 2
        Gc = np.where(dentro & (np.abs(s) < w), Gc_medio * debilidad, Gc)
    return Gc


# Fraccion de gamma_fin en la que la trayectoria NO PROPORCIONAL rota la
# direccion de carga. Se expresa como fraccion y no como valor absoluto de
# Gamma para que la misma definicion valga en 2D y en 3D, donde gamma_fin es
# distinto. En 2D estos valores reproducen el barrido del articulo: la
# excursion de P4 es SUBDOMINANTE y se borra, la de P5 DOMINA y no se borra.
CAMBIO = {"P4_no_proporcional": 0.45, "P5_excursion_grande": 0.80}


def direccion(nombre, d, tramo=0):
    """Direccion de deformacion macroscopica. El ultimo eje es el de traccion.

    `tramo` selecciona el segmento de una trayectoria no proporcional: 0 es la
    excursion inicial (traccion en el PRIMER eje) y 1 es la carga final
    (traccion en el ULTIMO eje, la misma de P1). Que P4 y P5 terminen cargando
    como P1 es lo que aisla la dependencia de HISTORIA de la dependencia de
    direccion final.
    """
    E = np.zeros((d, d))
    if nombre == "P1_uniaxial":
        E[-1, -1] = 1.0
    elif nombre == "P2_cizalla":
        E[-1, -1] = 1.0
        E[0, 0] = -1.0
    elif nombre == "P3_confinada":
        E[-1, -1] = 1.0
        for i in range(d - 1):
            E[i, i] = -2.0 / (d - 1)
    elif nombre == "P4_transversal":
        E[0, 0] = 1.0
    elif nombre in CAMBIO:
        if tramo == 0:
            E[0, 0] = 1.0                 # excursion transversal
        else:
            E[-1, -1] = 1.0               # carga final, igual que P1
    else:
        raise ValueError(nombre)
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trayectoria", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--ell-px", type=float, default=4.0)
    ap.add_argument("--contraste", type=float, default=1e4)
    ap.add_argument("--semilla", type=int, default=20260914)
    ap.add_argument("--dgamma", type=float, default=0.015)
    ap.add_argument("--gamma-fin", type=float, default=0.62)
    ap.add_argument("--max-stag", type=int, default=80)
    ap.add_argument("--campos-cada", type=int, default=5)
    ap.add_argument("--salida", default=str(AQUI.parent / "corridas"))
    ap.add_argument("--max-incrementos", type=int, default=200)
    ap.add_argument("--esquema-phi", default="fd",
                    choices=("fd", "rotated", "spectral"),
                    help="operador del campo de fase (doc 18). fd es el "
                         "correcto; rotated tiene modos sin gradiente en 3D.")
    ap.add_argument("--switch-frac", type=float, default=None,
                    help="fraccion de gamma-fin en que rota la carga; solo "
                         "para trayectorias no proporcionales. Por defecto "
                         "0.45 (P4) y 0.80 (P5).")
    ap.add_argument("--apertura", action="store_true",
                    help="calcular tambien V_c, <b>, D^(1), D^(3), Oda-Snow "
                         "y la ley cubica escalar (doc 22). Barato: una "
                         "reconstruccion de desplazamiento por incremento.")
    ap.add_argument("--gamma-ref", type=float, default=None,
                    help="Gamma de referencia para el punto de rotacion "
                         "(frac * gamma_ref). Por defecto gamma-fin. Al "
                         "EXTENDER una corrida a un gamma-fin mayor hay que "
                         "fijarlo al gamma-fin original, o P4/P5 rotarian "
                         "en otro punto y la historia quedaria rota.")
    a = ap.parse_args()

    d = a.dim
    N = (a.n,) * d
    L = (1.0,) * d
    ell = a.ell_px / a.n
    base = pathlib.Path(a.salida)
    base.mkdir(parents=True, exist_ok=True)
    # El operador va en el NOMBRE: los pilotos 3D viejos se calcularon con
    # "rotated" y quedaron fragmentados (doc 18). Si el nombre no lo
    # distinguiera, una corrida nueva encontraria su checkpoint y REANUDARIA
    # desde ese estado roto, en silencio.
    etiqueta = f"{a.trayectoria}_n{a.n}_d{d}_s{a.semilla}_{a.esquema_phi}"
    h5 = base / f"{etiqueta}.h5"
    ck = base / f"{etiqueta}.ck.npz"

    Gc = campo_Gc(N, a.semilla)
    lam0, mu0 = lame_from_E_nu(1.0, 0.2)
    S = PhaseFieldSolver(N, lam0, mu0, Gc, ell, L=L,
                         esquema_phi=a.esquema_phi)
    lam = 0.02
    if ck.exists():
        z = np.load(ck)
        S.phi = z["phi"].copy(); S.H = z["H"].copy()
        S._et = z["et"].copy(); S._tr_pos = z["tr_pos"].copy()
        S.eps = z["eps"].copy(); lam = float(z["lam"])
        print(f"[reanudado desde Gamma={S.longitud_grieta():.4f}]", flush=True)

    alm = Almacen(h5, meta=dict(
        trayectoria=a.trayectoria, n=a.n, dim=d, ell_px=a.ell_px,
        contraste=a.contraste, semilla=a.semilla, dgamma=a.dgamma,
        max_stag=a.max_stag, esquema_phi=a.esquema_phi),
        cada_n=a.campos_cada)

    # Para las trayectorias no proporcionales la direccion se decide en CADA
    # incremento segun la longitud de grieta alcanzada, no de una vez.
    frac = CAMBIO.get(a.trayectoria)
    if a.switch_frac is not None:
        frac = a.switch_frac
    g_ref = a.gamma_ref if a.gamma_ref is not None else a.gamma_fin
    G_cambio = frac * g_ref if frac is not None else None
    if G_cambio is not None:
        print(f"[trayectoria no proporcional: rota la carga en "
              f"Gamma={G_cambio:.4f}]", flush=True)

    t0 = time.time()
    i = 0
    tramo_previo = None
    while S.longitud_grieta() < a.gamma_fin and i < a.max_incrementos:
        if G_cambio is None:
            E_dir = direccion(a.trayectoria, d)
        else:
            tramo = 0 if S.longitud_grieta() < G_cambio else 1
            E_dir = direccion(a.trayectoria, d, tramo)
            if tramo != tramo_previo:
                if tramo_previo is not None:
                    print(f"  >> rotacion de carga en "
                          f"Gamma={S.longitud_grieta():.4f}", flush=True)
                tramo_previo = tramo
        info = S.paso_longitud(E_dir, a.dgamma, lam_ini=lam,
                               max_stag=a.max_stag, tol_stag=1e-5, tol=1e-10)
        lam = info["lam"]
        i += 1

        D0 = damage_tensor(S.phi, ell, L=L)
        R = localization_ratio(S.phi, ell, L=L)
        st = cluster_stats(S.phi > 0.5)
        k = 1.0 + (a.contraste - 1.0) * np.clip(S.phi, 0, 1) ** 2
        rk = homogenize(k, L=L, tol=1e-10, maxiter=3000, dual=False, ktol=1e-9)

        if a.apertura:
            # Momentos pesados por apertura y Oda-Snow. Verificado contra
            # Sneddon en el doc 22; antes de eso no se reportaban.
            from fftgk.apertura import volumen_grieta, campo_apertura
            from fftgk.cierres import oda_desde_D3
            ef = S.eps - S.eps.reshape(d, d, -1).mean(axis=2).reshape(
                (d, d) + (1,) * d)
            uu = S.P.desplazamiento(ef)
            Vc = volumen_grieta(uu, S.phi, L=L, projector=S.base_phi)
            bb = campo_apertura(uu, S.phi, ell, L=L, projector=S.base_phi)
            Gam = float(np.mean(S.phi ** 2) / ell)
            b_med = Vc / Gam if Gam > 0 else 0.0
            D3 = damage_tensor(S.phi, ell, L=L, weight=bb, m=3)
            ap = dict(Vc=Vc, b_medio=b_med,
                      D1=damage_tensor(S.phi, ell, L=L, weight=bb, m=1),
                      D3=D3, Koda=oda_desde_D3(D3),
                      K_cubica=Gam * b_med ** 3 / 12.0)
        else:
            ap = {}

        esc = dict(Gamma=info["Gamma"], lam=lam, D=D0, K=rk.K_primal, R=R,
                   n_clusters=st["n_clusters"], f_mayor=st["f_mayor"],
                   percola=np.array(st["percola"], dtype=np.int8),
                   phi_max=float(S.phi.max()), stag=info["stag_iters"],
                   segundos=time.time() - t0, **ap)
        if d == 2:
            esc["chi"] = st["chi"]
        alm.guardar(esc, campos=dict(phi=S.phi, H=S.H, eps=S.eps))
        np.savez_compressed(ck, phi=S.phi, H=S.H, et=S._et,
                            tr_pos=S._tr_pos, eps=S.eps, lam=lam)

        print(f"  {i:3d} G={info['Gamma']:.4f} lam={lam:.5f} "
              f"trD={np.trace(D0):.4f} K={np.diag(rk.K_primal).round(2)} "
              f"R={R:.3f} cum={st['n_clusters']} perc={st['percola']} "
              f"stag={info['stag_iters']} [{time.time()-t0:.0f}s]", flush=True)
        if info["stop"] == "objetivo-inalcanzable":
            break
    print(f"-> {i} incrementos en {time.time()-t0:.0f}s -> {h5}", flush=True)


if __name__ == "__main__":
    main()
