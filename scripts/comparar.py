"""
Los numeros titulares del articulo, en dos barridos lado a lado.

    python3 scripts/comparar.py resultados_final.jsonl resultados_fd.jsonl

Existe para una sola pregunta (doc 18): al cambiar el operador del campo de
fase de "rotated" a "fd", ¿se sostienen las conclusiones? Se calcula cada
numero EXACTAMENTE como en el manuscrito, para que la comparacion sea de
manzanas con manzanas.
"""
import itertools, json, sys
import numpy as np


def cargar(ruta):
    R = [json.loads(l) for l in open(ruta)]
    d = {}
    for r in R:
        d.setdefault(r["trayectoria"], {})[round(r["trD"], 3)] = r
    return R, d



def historia_comun(a, b):
    """P4 y P5 son el MISMO estado hasta que P4 rota (Gamma<=0.165): no
    son dos caminos distintos y no cuentan como evidencia a D igualado."""
    return ({a["trayectoria"], b["trayectoria"]} ==
            {"P4_no_proporcional", "P5_excursion_grande"}
            and max(a.get("Gamma", a["trD"]), b.get("Gamma", b["trD"])) <= 0.165 + 1e-9)

def vecD(r):
    return np.array([r["Dxx"], r["Dyy"], r["Dxy"]])


def titulares(ruta):
    R, d = cargar(ruta)
    out = {}
    # 1. piso escalar: dispersion de Kyy a trD igualado, niveles con >=3 caminos
    niv = {}
    for r in R:
        niv.setdefault(round(r["trD"], 3), []).append(r["Kyy"])
    disp = {t: (max(k) - min(k)) / np.mean(k) * 100
            for t, k in niv.items() if len(k) >= 3}
    t_peor = max(disp, key=disp.get)
    out["escalar_max_%"] = disp[t_peor]
    out["escalar_max_trD"] = t_peor
    out["escalar_mediana_%"] = float(np.median(list(disp.values())))
    # 2. cota tensorial: pares de trayectorias distintas con |dD|/|D| < 2%
    pares = []
    for a, b in itertools.combinations(R, 2):
        if a["trayectoria"] == b["trayectoria"] or a["trD"] <= 0.10 or historia_comun(a, b):
            continue
        rel = np.linalg.norm(vecD(a) - vecD(b)) / np.linalg.norm(vecD(a))
        if rel < 0.02:
            pares.append(abs(b["Kyy"] - a["Kyy"]) / a["Kyy"] * 100)
    out["tensor_n_pares"] = len(pares)
    out["tensor_max_%"] = max(pares) if pares else float("nan")
    out["tensor_mediana_%"] = float(np.median(pares)) if pares else float("nan")
    # 3. memoria: P4 y P5 contra P1 a trD igualado
    p1 = d["P1_uniaxial"]
    for nom, clave in (("P4_no_proporcional", "P4"), ("P5_excursion_grande", "P5")):
        tr = d.get(nom, {})
        com = sorted(set(p1) & set(tr))
        if not com:
            continue
        dif = [abs(tr[k]["Kyy"] - p1[k]["Kyy"]) / p1[k]["Kyy"] * 100 for k in com]
        out[f"{clave}_max_%"] = max(dif)
        out[f"{clave}_final_%"] = dif[-1]
        out[f"{clave}_final_trD"] = com[-1]
    # 4. exponencial de dano, mejor alpha (minimos cuadrados en log)
    x = np.array([r["trD"] for r in R]); K = np.array([r["Kyy"] for r in R])
    k0 = K.min()
    from scipy.optimize import minimize_scalar
    c = lambda a: np.mean((np.log(k0 * np.exp(a * x)) - np.log(K)) ** 2)
    a = minimize_scalar(c, bounds=(0.5, 40), method="bounded").x
    e = (k0 * np.exp(a * x) - K) / K * 100
    out["exp_alpha"] = a
    out["exp_mediano_%"] = float(np.median(np.abs(e)))
    out["exp_sesgo_trD>0.3_%"] = float(np.mean(e[x > 0.3]))
    # 5. cierre tensorial lineal, trD > 0.15
    filas, rhs = [], []
    for r in R:
        t = r["Dxx"] + r["Dyy"]
        if t <= 0.15:
            continue
        for kk, dd in (("Kxx", "Dxx"), ("Kyy", "Dyy")):
            filas.append([1.0, t, r[dd]]); rhs.append(r[kk])
    A, b = np.array(filas), np.array(rhs)
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    out["lineal_tensor_mediano_%"] = float(np.median(np.abs((A @ p - b) / b * 100)))
    # 6. P1: pico de carga y formacion del cumulo unico
    ks = sorted(p1)
    lam = [p1[k]["lam"] for k in ks]
    out["P1_lam_pico"] = max(lam); out["P1_Gamma_pico"] = ks[int(np.argmax(lam))]
    cruce = [k for k in ks if p1[k]["f_mayor"] >= 0.999]
    out["P1_cumulo_unico_trD"] = cruce[0] if cruce else float("nan")
    out["n_estados"] = len(R)
    out["max_clusters"] = max(r["n_clusters"] for r in R)
    return out


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    ta, tb = titulares(a), titulares(b)
    print(f"{'':28s} {a[:22]:>22s} {b[:22]:>22s}   cambio")
    for k in ta:
        va, vb = ta[k], tb.get(k, float("nan"))
        if isinstance(va, float) and va not in (0,) and np.isfinite(va) and np.isfinite(vb):
            cambio = f"{(vb - va) / abs(va) * 100:+6.1f}%"
        else:
            cambio = ""
        print(f"{k:28s} {va:22.4g} {vb:22.4g}   {cambio}")
