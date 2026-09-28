"""
Los mismos tres tests del 2D (doc 11 y doc 19), ahora en 3D.

A. Familia escalar   K = f(tr D): dispersion de K a tr D igualado.
B. Familia tensorial K = F(D_ij): pares de trayectorias distintas con el mismo
   tensor D completo (6 componentes en 3D).
C. Memoria de trayectoria: P4 y P5 frente a P1 en el rango comun.

La direccion de traccion es el ULTIMO eje, asi que el K que se compara es Kzz.
"""
import itertools
import json
import pathlib

import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
R = [json.loads(l) for l in open(AQUI.parent / "resultados_3d.jsonl")]
CTRL = {r["trayectoria"]: r["control_final"] for r in R if "control_final" in r}
R = [r for r in R if "control_final" not in r]

TOL_TENSOR = 0.02
# P4 rota en Gamma=0.36 y P5 en 0.64. Antes de su propia rotacion, P4 y P5 son
# el MISMO estado: no cuentan como dos caminos distintos.
G_P4, G_P5 = 0.36, 0.64


def vecD(r):
    D = np.asarray(r["D"])
    return np.array([D[0][0], D[1][1], D[2][2], D[0][1], D[0][2], D[1][2]])


def Kzz(r):
    return np.asarray(r["K"])[2][2]


def historia_comun(a, b):
    """Pares que NO son dos caminos distintos y por tanto no cuentan como
    evidencia a D igualado.

    1) P4 y P5 antes de que P4 rote: son literalmente el mismo estado.
    2) P1 y P4 DESPUES de que la excursion de P4 se borra. Medido aqui: a
       Gamma>=0.48 el tensor D de P4 difiere del de P1 en menos de 1% y su K
       en 0.07%. Es decir, P4 ya ES P1. Incluir esos pares infla la cota
       tensorial con la diferencia entre dos puntos de la MISMA trayectoria."""
    par = {a["trayectoria"], b["trayectoria"]}
    g = max(a["Gamma"], b["Gamma"])
    if par == {"P4_no_proporcional", "P5_excursion_grande"}:
        return g <= G_P4 + 1e-9
    if par == {"P1_uniaxial", "P4_no_proporcional"}:
        return min(a["Gamma"], b["Gamma"]) >= 0.48
    return False


print("=" * 72)
print("0. SANIDAD DE LOS CAMPOS (ultimo campo guardado)")
print("=" * 72)
print(f"  {'trayectoria':22s} {'cumulos':>8} {'bajo l':>7} {'mayor':>8} "
      f"{'f_danada':>9} {'phi_max':>8}")
for t, c in CTRL.items():
    print(f"  {t:22s} {c['n_clusters']:8d} {c.get('bajo_l',0):7d} "
          f"{c.get('mayor',0):8d} {c['f_danada']*100:8.3f}% {c['phi_max']:8.3f}")

print()
print("=" * 72)
print("1. LOCALIZACION Y PERCOLACION")
print("=" * 72)
for t in sorted({r["trayectoria"] for r in R}):
    g = [r for r in R if r["trayectoria"] == t]
    snap = next((x for x in g if x["f_mayor"] > 0.5 and x["n_clusters"] > 0
                 and x["Gamma"] > 0.5), None)
    perc = any(any(x["percola"]) for x in g)
    kk = [r for r in g]
    print(f"  {t:22s} Gamma_max={g[-1]['Gamma']:.3f}  "
          f"f_mayor final={g[-1]['f_mayor']:.3f}  "
          f"cumulos final={g[-1]['n_clusters']:3d}  percola={perc}")

print()
print("=" * 72)
print("A. FAMILIA ESCALAR  K = f(tr D)   -- rango comun a las 5 trayectorias")
print("=" * 72)
niveles = {}
for r in R:
    niveles.setdefault(round(r["trD"], 3), []).append(r)
filas = []
for t in sorted(niveles):
    g = niveles[t]
    if len(g) < 5:            # solo niveles donde estan las cinco
        continue
    k = np.array([Kzz(x) for x in g])
    filas.append((t, len(g), (k.max() - k.min()) / k.mean() * 100, k.mean()))
print(f"  {'tr D':>7} {'paths':>6} {'K medio':>9} {'dispersion':>11}")
for t, n, disp, km in filas[::4]:
    print(f"  {t:7.3f} {n:6d} {km:9.2f} {disp:10.2f}%")
peor = max(filas, key=lambda f: f[2])
print(f"\n  peor caso: {peor[2]:.1f}% a tr D = {peor[0]:.3f}   "
      f"(mediana {np.median([f[2] for f in filas]):.1f}%)")
post = [f for f in filas if f[0] >= 0.855]
if post:
    print(f"  solo estados agrietados (tr D >= 0.855): "
          f"max {max(f[2] for f in post):.1f}%, "
          f"mediana {np.median([f[2] for f in post]):.1f}%")

print()
print("=" * 72)
print("B. FAMILIA TENSORIAL  K = F(D_ij)   -- 6 componentes")
print("=" * 72)
pares = []
for a, b in itertools.combinations(R, 2):
    if a["trayectoria"] == b["trayectoria"] or historia_comun(a, b):
        continue
    da, db = vecD(a), vecD(b)
    rel = np.linalg.norm(da - db) / np.linalg.norm(da)
    if rel < TOL_TENSOR and a["trD"] > 0.10:
        dk = abs(Kzz(b) - Kzz(a)) / Kzz(a) * 100
        pares.append((rel * 100, dk, a, b))
pares.sort(key=lambda p: -p[1])
print(f"  {len(pares)} pares de trayectorias distintas con |dD|/|D| < 2%")
print(f"  {'|dD|':>7} {'|dKzz|':>9}   trayectorias")
for rel, dk, a, b in pares[:8]:
    print(f"  {rel:6.2f}% {dk:8.2f}%   {a['trayectoria'][:16]:17s} vs "
          f"{b['trayectoria'][:16]:17s} (tr D = {a['trD']:.3f})")
if pares:
    print(f"\n  peor discrepancia: {max(p[1] for p in pares):.2f}%   "
          f"mediana {np.median([p[1] for p in pares]):.2f}%")
    print(f"  razon piso escalar / cota tensorial: "
          f"{peor[2] / max(p[1] for p in pares):.0f}x")

print()
print("=" * 72)
print("C. MEMORIA DE TRAYECTORIA   (P4 y P5 contra P1, a Gamma igualado)")
print("=" * 72)
ref = {round(r["Gamma"], 3): r for r in R if r["trayectoria"] == "P1_uniaxial"}
for t, g_rot in (("P4_no_proporcional", G_P4), ("P5_excursion_grande", G_P5)):
    fil = []
    for r in R:
        if r["trayectoria"] != t:
            continue
        p1 = ref.get(round(r["Gamma"], 3))
        if p1 is None or r["Gamma"] <= g_rot:
            continue
        dk = abs(Kzz(r) - Kzz(p1)) / Kzz(p1) * 100
        dd = (np.linalg.norm(vecD(r) - vecD(p1)) / np.linalg.norm(vecD(p1)) * 100)
        fil.append((r["Gamma"], dk, dd))
    if not fil:
        continue
    print(f"\n  {t}  (rota en Gamma={g_rot})")
    print(f"    {'Gamma':>7} {'|dKzz| vs P1':>13} {'|dD| vs P1':>12}")
    for G, dk, dd in fil[::4]:
        print(f"    {G:7.3f} {dk:12.3f}% {dd:11.3f}%")
    print(f"    ultimo: {fil[-1][0]:.3f} -> dK {fil[-1][1]:.3f}%, "
          f"dD {fil[-1][2]:.3f}%   |   maximo dK {max(f[1] for f in fil):.3f}%")

print()
print("=" * 72)
print("D. EL CIERRE CONCRETO  k = k0 exp(alpha tr D)  -- mejor ajuste global")
print("=" * 72)
import sys
sys.path.insert(0, str(AQUI.parent / "src"))
from fftgk.cierres import calibrar_escalar, error_relativo   # noqa: E402

x = np.array([r["trD"] for r in R])
K = np.array([Kzz(r) for r in R])
k0 = float(K.min())
alpha, Km, _ = calibrar_escalar(lambda t, a: k0 * np.exp(a * t), x, K,
                                10.0, (0.5, 40.0))
err = error_relativo(Km, K) * 100
print(f"  alpha optimo = {alpha:.2f}   (la literatura usa 5-25)")
print(f"  error mediano = {np.median(np.abs(err)):.1f}%   "
      f"maximo = {np.max(np.abs(err)):.1f}%")

salida = dict(
    escalar_max=float(peor[2]), escalar_trD=float(peor[0]),
    escalar_mediana=float(np.median([f[2] for f in filas])),
    tensorial_max=float(max(p[1] for p in pares)) if pares else None,
    tensorial_mediana=float(np.median([p[1] for p in pares])) if pares else None,
    n_pares=len(pares), control=CTRL,
    exp_alpha=float(alpha), exp_err_mediano=float(np.median(np.abs(err))),
    exp_err_max=float(np.max(np.abs(err))),
)
(AQUI.parent / "cierres_error_3d.json").write_text(json.dumps(salida, indent=2))
print("\n-> cierres_error_3d.json")
