"""
Cuanto se equivocan los cierres, sin ajustar nada.

La prueba es no parametrica a proposito. Un cierre escalar dice que K es una
funcion de tr D, cualquiera que sea la funcion. Entonces, en dos estados con el
mismo tr D, el cierre esta OBLIGADO a dar el mismo K. La dispersion real de K a
tr D igualado es, por tanto, el error IRREDUCIBLE de la familia entera: ninguna
calibracion, ninguna forma funcional, ningun parametro extra puede bajarlo.

La misma prueba se aplica a la familia tensorial K = F(D_ij): se buscan pares de
trayectorias distintas cuyo tensor D completo coincida, y se mide cuanto difiere
K. Aqui la respuesta es la contraria y es el resultado util del trabajo.

Ademas se calibra el exponencial de dano (k = k0 exp(alpha tr D)) al mejor
ajuste posible sobre TODOS los puntos, para poner una cifra al cierre concreto
que la industria escribe, no solo a la familia.
"""
import itertools
import json
import pathlib

import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
import os
# Fuente de datos: POROK_DATOS=resultados_fd.jsonl para el barrido con el
# operador corregido (doc 18). Por defecto, el barrido publicado.
DATOS = AQUI.parent / os.environ.get("POROK_DATOS", "resultados_final.jsonl")
R = [json.loads(l) for l in open(DATOS)]

TOL_TRAZA = 1e-6          # los niveles de Gamma son comunes por construccion
TOL_TENSOR = 0.02         # 2% en norma de Frobenius relativa



def historia_comun(a, b):
    """P4 y P5 son el MISMO estado hasta que P4 rota (Gamma<=0.165): no
    son dos caminos distintos y no cuentan como evidencia a D igualado."""
    return ({a["trayectoria"], b["trayectoria"]} ==
            {"P4_no_proporcional", "P5_excursion_grande"}
            and max(a.get("Gamma", a["trD"]), b.get("Gamma", b["trD"])) <= 0.165 + 1e-9)

def vecD(r):
    return np.array([r["Dxx"], r["Dyy"], r["Dxy"]])


print("=" * 70)
print("A. FAMILIA ESCALAR   K = f(tr D)   -- error irreducible")
print("=" * 70)
niveles = {}
for r in R:
    niveles.setdefault(round(r["trD"], 3), []).append(r)

filas = []
for t in sorted(niveles):
    g = niveles[t]
    if len(g) < 2:
        continue
    k = np.array([x["Kyy"] for x in g])
    disp = (k.max() - k.min()) / k.mean() * 100
    filas.append((t, len(g), disp, k.mean()))
print(f"  {'tr D':>7} {'paths':>6} {'K medio':>9} {'dispersion':>11}")
for t, n, disp, km in filas:
    marca = "  <--" if disp == max(f[2] for f in filas) else ""
    print(f"  {t:7.3f} {n:6d} {km:9.2f} {disp:10.2f}%{marca}")
peor = max(filas, key=lambda f: f[2])
print(f"\n  peor caso: {peor[2]:.1f}% a tr D = {peor[0]:.3f}")
print(f"  mediana  : {np.median([f[2] for f in filas]):.1f}%")

print()
print("=" * 70)
print("B. FAMILIA TENSORIAL   K = F(D_ij)   -- misma prueba")
print("=" * 70)
pares = []
for a, b in itertools.combinations(R, 2):
    if a["trayectoria"] == b["trayectoria"] or historia_comun(a, b):
        continue
    da, db = vecD(a), vecD(b)
    rel = np.linalg.norm(da - db) / np.linalg.norm(da)
    if rel < TOL_TENSOR and a["trD"] > 0.10:
        dk = abs(b["Kyy"] - a["Kyy"]) / a["Kyy"] * 100
        pares.append((rel * 100, dk, a, b))
pares.sort(key=lambda p: -p[1])
print(f"  {len(pares)} pares de trayectorias distintas con |dD|/|D| < {TOL_TENSOR*100:.0f}%")
print(f"  {'|dD|':>7} {'|dK_yy|':>9}   trayectorias")
for rel, dk, a, b in pares[:6]:
    print(f"  {rel:6.2f}% {dk:8.2f}%   {a['trayectoria'][:16]:17s} vs {b['trayectoria'][:16]}"
          f"   (tr D = {a['trD']:.3f})")
print(f"\n  peor discrepancia con D igualado: {max(p[1] for p in pares):.2f}%")
print(f"  mediana                        : {np.median([p[1] for p in pares]):.2f}%")

print()
print("=" * 70)
print("C. EL CIERRE CONCRETO   k = k0 exp(alpha tr D)   -- mejor ajuste global")
print("=" * 70)
import sys
sys.path.insert(0, str(AQUI.parent / "src"))
from fftgk.cierres import calibrar_escalar, error_relativo

x = np.array([r["trD"] for r in R])
K = np.array([r["Kyy"] for r in R])
# k0 se fija por el limite sin dano: el primer punto de la serie monotona
k0 = min(K)


def modelo(t, alpha):
    return k0 * np.exp(alpha * t)


alpha, Km, med = calibrar_escalar(modelo, x, K, 10.0, (0.5, 40.0))
err = error_relativo(Km, K) * 100
print(f"  alpha optimo = {alpha:.2f}   (la literatura usa 5-25)")
print(f"  error mediano = {np.median(np.abs(err)):.1f}%")
print(f"  error maximo  = {np.max(np.abs(err)):.1f}%")
print(f"  sesgo a tr D > 0.3: {np.mean(err[x > 0.3]):+.1f}%")

# y la ley de potencias, con la misma variable
def modelo_pot(t, m):
    return k0 * (t / x.min()) ** m


m, Kp, medp = calibrar_escalar(modelo_pot, x, K, 1.0, (0.1, 6.0))
errp = error_relativo(Kp, K) * 100
print(f"\n  ley de potencias  k = k0 (tr D / trD_0)^m :  m = {m:.2f}")
print(f"  error mediano = {np.median(np.abs(errp)):.1f}%   maximo = {np.max(np.abs(errp)):.1f}%")

salida = {
    "escalar_irreducible_max": float(peor[2]),
    "escalar_irreducible_trD": float(peor[0]),
    "escalar_irreducible_mediana": float(np.median([f[2] for f in filas])),
    "tensorial_max": float(max(p[1] for p in pares)),
    "tensorial_mediana": float(np.median([p[1] for p in pares])),
    "n_pares_tensorial": len(pares),
    "exp_alpha": alpha, "exp_err_mediano": float(np.median(np.abs(err))),
    "exp_err_max": float(np.max(np.abs(err))),
    "pot_m": m, "pot_err_mediano": float(np.median(np.abs(errp))),
    "pot_err_max": float(np.max(np.abs(errp))),
}
(AQUI.parent / ("cierres_error.json" if DATOS.name == "resultados_final.jsonl"
               else "cierres_error_" + DATOS.stem.replace("resultados_", "") + ".json")).write_text(json.dumps(salida, indent=2))
print("\n-> cierres_error.json")
