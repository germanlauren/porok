"""
Figura 5: el 3D confirma los dos resultados, con la misma prueba no parametrica.

(a) Dispersion de K a tr D igualado, en 2D y 3D, contra Gamma/Gamma_localizacion.
    Normalizar por el Gamma de localizacion es lo unico que hace comparables las
    dos dimensiones: el 3D localiza en 0.855 y el 2D en 0.375, asi que a Gamma
    crudo se estarian comparando estados distintos de la misma historia.

(b) Los mismos pares cruzados con el TENSOR igualado, en 2D (3 componentes) y
    3D (6 componentes). La nube 3D cae igual de bajo: no hay meseta.

Uso:  python3 scripts/figura3d.py
"""
import itertools
import json
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from figuras import (AZUL, ROJO, VERDE, TINTA2, GRIS, limpiar,   # noqa: E402
                     guardar as _guardar)

SAL = AQUI.parent / "figuras_fd"
SAL.mkdir(exist_ok=True)

G_LOC_2D, G_LOC_3D = 0.375, 0.855
G_P4_3D = 0.36


def leer(nombre):
    R = [json.loads(l) for l in open(AQUI.parent / nombre)]
    return [r for r in R if "control_final" not in r]


def comun2d(a, b):
    return ({a["trayectoria"], b["trayectoria"]} ==
            {"P4_no_proporcional", "P5_excursion_grande"}
            and max(a["Gamma"], b["Gamma"]) <= 0.165 + 1e-9)


def comun3d(a, b):
    par = {a["trayectoria"], b["trayectoria"]}
    if par == {"P4_no_proporcional", "P5_excursion_grande"}:
        return max(a["Gamma"], b["Gamma"]) <= G_P4_3D + 1e-9
    if par == {"P1_uniaxial", "P4_no_proporcional"}:
        # tras borrarse la excursion, P4 ES P1: no son dos caminos
        return min(a["Gamma"], b["Gamma"]) >= 0.48
    return False


def K2(r):
    return r["Kyy"]


def K3(r):
    return np.asarray(r["K"])[2][2]


def D2(r):
    return np.array([r["Dxx"], r["Dyy"], r["Dxy"]])


def D3(r):
    D = np.asarray(r["D"])
    return np.array([D[0][0], D[1][1], D[2][2], D[0][1], D[0][2], D[1][2]])


def dispersion(R, Kf, minimo):
    niveles = {}
    for r in R:
        niveles.setdefault(round(r["trD"], 3), []).append(r)
    x, y = [], []
    for t in sorted(niveles):
        g = niveles[t]
        if len(g) < minimo:
            continue
        k = np.array([Kf(q) for q in g])
        x.append(t)
        y.append((k.max() - k.min()) / k.mean() * 100)
    return np.array(x), np.array(y)


def nube(R, Kf, Df, comun):
    p = []
    for a, b in itertools.combinations(R, 2):
        if a["trayectoria"] == b["trayectoria"] or comun(a, b):
            continue
        if a["trD"] <= 0.10:
            continue
        rel = np.linalg.norm(Df(a) - Df(b)) / np.linalg.norm(Df(a))
        p.append((rel * 100, abs(Kf(b) - Kf(a)) / Kf(a) * 100))
    p = np.array(p)
    return p[(p[:, 0] > 1e-3) & (p[:, 1] > 1e-3)]


def panel_escalar_2d(ax, R2):
    x, y = dispersion(R2, K2, 3)
    ax.plot(x, y, "-", color=ROJO, lw=1.6)
    ax.fill_between(x, 0, y, color=ROJO, alpha=0.10, lw=0)
    j = int(np.argmax(y))
    ax.plot(x[j], y[j], "o", ms=5, color=ROJO, zorder=5)
    ax.annotate(f"{y[j]:.1f}%", (x[j], y[j]), xytext=(-4, 5),
                textcoords="offset points", fontsize=8, fontweight="bold",
                color=ROJO, ha="right")
    ax.set_xlabel(r"crack density  tr $\mathbf{D}$")
    ax.set_ylabel(r"spread in $K_{yy}$  (%)")
    ax.set_title(r"(a)  2D: irreducible error of $K=f(\mathrm{tr}\,\mathbf{D})$",
                 loc="left")
    ax.set_ylim(bottom=0)
    limpiar(ax)


def panel_nube_2d(ax, p2):
    d2 = p2[:, 0] < 2.0
    ax.loglog(p2[~d2, 0], p2[~d2, 1], ".", ms=3, color=GRIS, alpha=0.8, zorder=1)
    ax.loglog(p2[d2, 0], p2[d2, 1], "o", ms=4.5, color=VERDE, zorder=3,
              markeredgecolor="white", markeredgewidth=0.6)
    guia = np.array([0.03, 60.0])
    ax.plot(guia, guia, "--", color=TINTA2, lw=0.7, zorder=2)
    ax.annotate("1:1", (30, 34), fontsize=7, color=TINTA2, ha="right")
    ax.axvline(2.0, color=VERDE, lw=0.7, ls=":")
    ax.annotate(f"matched $\\mathbf{{D}}$ (within 2%):\n"
                f"$|\\Delta K_{{yy}}|\\leq{p2[d2,1].max():.1f}\\%$",
                (0.045, 12), fontsize=7.5, color=VERDE, fontweight="bold")
    ax.set_xlabel(r"tensor mismatch  $\|\Delta\mathbf{D}\|/\|\mathbf{D}\|$  (%)")
    ax.set_ylabel(r"$|\Delta K_{yy}| / K_{yy}$   (%)")
    ax.set_title(r"(b)  2D: all cross-path pairs", loc="left")
    ax.set_xlim(0.03, 60)
    ax.set_ylim(0.005, 100)
    limpiar(ax, eje="both")


def main():
    R2, R3 = leer("resultados_fd.jsonl"), leer("resultados_3d.jsonl")
    fig, ejes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    panel_escalar_2d(ejes[0][0], R2)
    a1, a2 = ejes[1][0], ejes[1][1]

    # ---------------- (a) piso escalar, normalizado por la localizacion
    for R, Kf, gl, col, eti in ((R2, K2, G_LOC_2D, GRIS, "2D  (96$^2$)"),
                                (R3, K3, G_LOC_3D, AZUL, "3D  (96$^3$)")):
        x, y = dispersion(R, Kf, 5)
        a1.plot(x / gl, y, "-", color=col, lw=1.8, solid_capstyle="round",
                label=eti)
    a1.axvline(1.0, color=ROJO, lw=0.8, ls=":")
    a1.annotate("crack forms", (1.0, 41), xytext=(4, 0),
                textcoords="offset points", fontsize=7, color=ROJO, va="top")
    x3, y3 = dispersion(R3, K3, 5)
    a1.plot(x3[-1] / G_LOC_3D, y3[-1], "o", ms=5, color=AZUL, zorder=5)
    a1.annotate(f"{y3[-1]:.1f}%", (x3[-1] / G_LOC_3D, y3[-1]), xytext=(-3, 6),
                textcoords="offset points", fontsize=8, fontweight="bold",
                color=AZUL, ha="right")
    a1.set_xlabel(r"$\Gamma\,/\,\Gamma_{\mathrm{loc}}$")
    a1.set_ylabel(r"spread in $K$ at matched tr $\mathbf{D}$  (%)")
    a1.set_xlabel(r"$\Gamma\,/\,\Gamma_{\mathrm{loc}}$")
    a1.set_title(r"(c)  2D vs 3D, staged by localization", loc="left")
    a1.set_ylim(bottom=0)
    a1.legend(loc="upper left")
    limpiar(a1)

    # ---------------- (b) cota tensorial en las dos dimensiones
    p2 = nube(R2, K2, D2, comun2d)
    p3 = nube(R3, K3, D3, comun3d)
    panel_nube_2d(ejes[0][1], p2)
    # Solo la nube 3D: superponer tambien la 2D (miles de pares) tapa el punto
    # de la figura, que es donde CAE la nube cuando el tensor se iguala. El
    # valor 2D va como anotacion.
    d2 = p2[:, 0] < 2.0
    d3 = p3[:, 0] < 2.0
    a2.loglog(p3[~d3, 0], p3[~d3, 1], ".", ms=3, color=AZUL, alpha=0.5,
              zorder=2, label="3D, all cross-path pairs")
    a2.loglog(p3[d3, 0], p3[d3, 1], "o", ms=5, color=VERDE, zorder=3,
              markeredgecolor="white", markeredgewidth=0.6,
              label=r"3D, matched $\mathbf{D}$ (within 2%)")
    guia = np.array([0.03, 60.0])
    a2.plot(guia, guia, "--", color=TINTA2, lw=0.7, zorder=2)
    a2.annotate("1:1", (30, 34), fontsize=7, color=TINTA2, ha="right")
    a2.axvline(2.0, color=VERDE, lw=0.7, ls=":")
    a2.annotate(f"matched $\\mathbf{{D}}$:  "
                f"$|\\Delta K|\\leq{p3[d3,1].max():.2f}\\%$  (3D)\n"
                f"{' ':16s}$\\leq{p2[d2,1].max():.2f}\\%$  (2D)",
                (0.045, 18), fontsize=7.5, color=VERDE, fontweight="bold")
    a2.set_xlabel(r"tensor mismatch  $\|\Delta\mathbf{D}\|/\|\mathbf{D}\|$  (%)")
    a2.set_ylabel(r"$|\Delta K| / K$   (%)")
    a2.set_title(r"(d)  3D: cross-path pairs, six components", loc="left")
    a2.set_xlim(0.03, 60)
    a2.set_ylim(0.005, 100)
    a2.legend(loc="lower right")
    limpiar(a2, eje="both")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SAL / f"fig3_escalar_vs_tensor.{ext}")
    plt.close(fig)
    print("fig3_escalar_vs_tensor (2x2, 2D+3D) ->", SAL)


if __name__ == "__main__":
    main()
