"""
Figuras del manuscrito. Salida en PDF vectorial para la revista y PNG para vista.

Paleta validada con el script de la guia de visualizacion (los seis chequeos
pasan, incluida la separacion para daltonismo). El aviso de contraste del ambar
se cubre con etiquetas directas sobre las curvas, que es el relieve exigido.

Orden de las figuras = orden del argumento:

  1  la referencia: se atraviesa el punto limite y se obtiene K(tr D) en cinco
     trayectorias
  2  la historia importa: a tr D igualado, K depende del camino
  3  ... pero no mas alla de D_ij: con el TENSOR igualado, K colapsa
  4  cuanto se equivocan los cierres concretos
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
import os
# Fuente de datos: POROK_DATOS=resultados_fd.jsonl para el barrido con el
# operador corregido (doc 18). Por defecto, el barrido publicado.
DATOS = AQUI.parent / os.environ.get("POROK_DATOS", "resultados_final.jsonl")
sys.path.insert(0, str(AQUI.parent / "src"))
SAL = AQUI.parent / ("figuras" if DATOS.name == "resultados_final.jsonl"
                     else "figuras_" + DATOS.stem.replace("resultados_", ""))
SAL.mkdir(exist_ok=True)

# paleta categorica validada: ALL CHECKS PASS (light, surface #fcfcfb)
AZUL, ROJO, AMBAR, VERDE = "#2e75b6", "#d64545", "#e8a33d", "#1f8a70"
TINTA, TINTA2, GRIS, PLOMO = "#1a1a1a", "#4a4a4a", "#cfcfcf", "#9a9a9a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": TINTA2, "axes.linewidth": 0.6,
    "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.labelcolor": TINTA, "text.color": TINTA,
    "grid.color": GRIS, "grid.linewidth": 0.4,
    "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

ETIQUETA = {
    "P1_uniaxial": "Uniaxial",
    "P2_cizalla": "Shear",
    "P3_confinada": "Confined",
    "P4_no_proporcional": "Sub-dominant excursion",
    "P5_excursion_grande": "Dominant excursion",
}


def cargar():
    d = {}
    for l in open(DATOS):
        r = json.loads(l)
        d.setdefault(r["trayectoria"], {})[round(r["trD"], 3)] = r
    return d


def registros():
    return [json.loads(l) for l in open(DATOS)]


def serie(tr, campo):
    ks = sorted(tr)
    return np.array(ks), np.array([tr[k][campo] for k in ks])


def limpiar(ax, eje="y"):
    """Ejes recesivos: solo la rejilla, sin marco superior/derecho."""
    ax.grid(axis=eje, lw=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)



def historia_comun(a, b):
    """P4 y P5 son el MISMO estado hasta que P4 rota (Gamma<=0.165): no
    son dos caminos distintos y no cuentan como evidencia a D igualado."""
    return ({a["trayectoria"], b["trayectoria"]} ==
            {"P4_no_proporcional", "P5_excursion_grande"}
            and max(a.get("Gamma", a["trD"]), b.get("Gamma", b["trD"])) <= 0.165 + 1e-9)

def vecD(r):
    return np.array([r["Dxx"], r["Dyy"], r["Dxy"]])


# --------------------------------------------------------------------- #
def figura1(d):
    """La referencia. Izquierda: se atraviesa el punto limite -- sin eso no hay
    rama de ablandamiento y el barrido se corta justo donde empieza a importar.
    Derecha: K(tr D) en las cinco trayectorias."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    p1 = d["P1_uniaxial"]
    g, lam = serie(p1, "Gamma"), serie(p1, "lam")[1]
    g = g[0]
    a1.plot(g, lam, "-", color=AZUL, lw=1.6)
    i = int(np.argmax(lam))
    a1.plot(g[i], lam[i], "o", ms=5, color=ROJO, zorder=5)
    a1.annotate("limit point\nload control stops here", (g[i], lam[i]),
                xytext=(-62, -58), textcoords="offset points", fontsize=7,
                color=ROJO, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=ROJO,
                                shrinkA=2, shrinkB=3))
    a1.annotate("softening branch:\nload falls while\nthe crack grows",
                (g[-1], lam[-1]), xytext=(2, -40), textcoords="offset points",
                fontsize=7, color=TINTA2, ha="right", va="top")
    a1.set_xlabel(r"prescribed crack density  $\Gamma$")
    a1.set_ylabel(r"load factor  $\lambda$")
    a1.set_ylim(top=lam[i] * 1.06)
    a1.set_title("(a)  crack-length control", loc="left")
    limpiar(a1)

    colores = {"P1_uniaxial": PLOMO, "P2_cizalla": VERDE, "P3_confinada": AMBAR,
               "P4_no_proporcional": AZUL, "P5_excursion_grande": ROJO}
    anchos = {"P1_uniaxial": 3.2}
    for nombre in ("P1_uniaxial", "P2_cizalla", "P3_confinada",
                   "P4_no_proporcional", "P5_excursion_grande"):
        x, y = serie(d[nombre], "Kyy")
        a2.plot(x, y, "-", color=colores[nombre], lw=anchos.get(nombre, 1.3),
                solid_capstyle="round", label=ETIQUETA[nombre])
    a2.set_xlabel(r"crack density  tr $\mathbf{D}$")
    a2.set_ylabel(r"$K_{yy}\,/\,K_{\mathrm{matrix}}$")
    a2.set_title("(b)  effective permeability, five loading paths", loc="left")
    a2.legend(loc="upper left")
    limpiar(a2)

    fig.tight_layout()
    guardar(fig, "fig1_referencia")


# --------------------------------------------------------------------- #
def figura2(d):
    """A tr D igualado el camino importa -- y la localizacion dice por que."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    p1, p4, p5 = d["P1_uniaxial"], d["P4_no_proporcional"], d["P5_excursion_grande"]

    for tr, col, eti in ((p4, AZUL, "P4"), (p5, ROJO, "P5")):
        comun = sorted(set(p1) & set(tr))
        x = np.array(comun)
        y = np.array([abs(tr[k]["Kyy"] - p1[k]["Kyy"]) / p1[k]["Kyy"] * 100
                      for k in comun])
        a1.semilogy(x, y, "-", color=col, lw=1.6)
        a1.annotate(eti, (x[-1], y[-1]), textcoords="offset points",
                    xytext=(4, 0), color=col, fontsize=8, va="center",
                    fontweight="bold")
    # Cada linea se rotula por separado: una sola nota en medio se montaba
    # sobre la curva azul.
    for xx, col, eti in ((0.165, AZUL, "P4 switches to $y$"),
                         (0.330, ROJO, "P5 switches to $y$")):
        a1.axvline(xx, color=col, lw=0.7, ls=":", alpha=0.8)
        a1.text(xx, 0.02, eti, transform=a1.get_xaxis_transform(),
                fontsize=7, color=col, rotation=90, ha="right", va="bottom")
    a1.set_xlabel(r"crack density  tr $\mathbf{D}$")
    a1.set_ylabel(r"$|\Delta K_{yy}| / K_{yy}$   (%)")
    a1.set_title("(a)  departure from the monotonic path", loc="left")
    a1.set_xlim(right=max(0.45, max(d["P5_excursion_grande"]) * 1.07))
    limpiar(a1)
    a1.grid(axis="y", which="both", lw=0.4, alpha=0.5)

    p2 = d["P2_cizalla"]
    for tr, col, eti, dy in ((p1, AZUL, "Uniaxial", 4), (p2, VERDE, "Shear", -2)):
        x, y = serie(tr, "R")
        a2.plot(x, y, "-", color=col, lw=1.6)
        a2.annotate(eti, (x[-1], y[-1]), textcoords="offset points",
                    xytext=(4, dy), color=col, fontsize=8,
                    fontweight="bold", va="center")
    # El umbral R = 1 vale en el continuo, pero evaluado sobre la malla el
    # valor absoluto depende del esquema de derivada (el gradiente espectral da
    # ~2x el de diferencias finitas). Asi que el hito se marca donde la fase
    # danada se vuelve UN SOLO CUMULO CONEXO, que ningun esquema puede mover.
    x1, f1 = serie(p1, "f_mayor")
    cruce = [x for x, f in zip(x1, f1) if f >= 0.999]
    if cruce:
        xc = cruce[0]
        yc = float(np.interp(xc, *serie(p1, "R")))
        a2.plot(xc, yc, "o", ms=5.5, color=AZUL, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)
        # La nota va arriba a la izquierda, que es el unico hueco libre: abajo
        # se montaba sobre la curva de cizalla.
        a2.annotate("damaged phase becomes\na single connected cluster",
                    (xc, yc), xytext=(0.04, 0.90), textcoords="axes fraction",
                    fontsize=7, color=AZUL, ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=AZUL,
                                    shrinkA=2, shrinkB=4))
    a2.set_xlabel(r"crack density  tr $\mathbf{D}$")
    a2.set_ylabel(r"localization ratio  $R$")
    a2.set_title("(b)  the same states, by localization", loc="left")
    a2.set_xlim(right=max(serie(p2, "R")[0]) * 1.22)
    limpiar(a2)

    fig.tight_layout()
    guardar(fig, "fig2_camino")


# --------------------------------------------------------------------- #
def figura3(d, R):
    """El resultado util: lo que falla es la familia ESCALAR, no la tensorial."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # (a) dispersion de K a tr D igualado = error irreducible de K = f(tr D)
    niveles = {}
    for r in R:
        niveles.setdefault(round(r["trD"], 3), []).append(r)
    x, y, n = [], [], []
    for t in sorted(niveles):
        g = niveles[t]
        if len(g) < 3:      # con menos de tres caminos la dispersion no informa
            continue
        k = np.array([q["Kyy"] for q in g])
        x.append(t)
        y.append((k.max() - k.min()) / k.mean() * 100)
        n.append(len(g))
    x, y = np.array(x), np.array(y)
    a1.plot(x, y, "-", color=ROJO, lw=1.6)
    a1.fill_between(x, 0, y, color=ROJO, alpha=0.10, lw=0)
    j = int(np.argmax(y))
    a1.plot(x[j], y[j], "o", ms=5, color=ROJO, zorder=5)
    a1.annotate(f"{y[j]:.1f}%", (x[j], y[j]), xytext=(-4, 5),
                textcoords="offset points", fontsize=8, fontweight="bold",
                color=ROJO, ha="right")
    a1.set_xlabel(r"crack density  tr $\mathbf{D}$")
    a1.set_ylabel(r"spread in $K_{yy}$ at matched tr $\mathbf{D}$  (%)")
    a1.set_title(r"(a)  irreducible error of any $K=f(\mathrm{tr}\,\mathbf{D})$",
                 loc="left")
    a1.set_ylim(bottom=0)
    limpiar(a1)

    # (b) mismo test con el tensor completo igualado
    pares = []
    for a, b in itertools.combinations(R, 2):
        if a["trayectoria"] == b["trayectoria"] or historia_comun(a, b):
            continue
        rel = np.linalg.norm(vecD(a) - vecD(b)) / np.linalg.norm(vecD(a))
        if a["trD"] > 0.10:
            pares.append((rel * 100,
                          abs(b["Kyy"] - a["Kyy"]) / a["Kyy"] * 100))
    # En log-log se ve lo que importa: la nube baja hasta el suelo cuando el
    # desajuste del tensor tiende a cero. Si hubiera una variable oculta, la
    # nube se aplanaria en una meseta en vez de seguir bajando.
    pares = np.array(pares)
    pares = pares[(pares[:, 0] > 1e-3) & (pares[:, 1] > 1e-3)]
    dentro = pares[:, 0] < 2.0
    a2.loglog(pares[~dentro, 0], pares[~dentro, 1], ".", ms=3, color=GRIS,
              alpha=0.8, zorder=1)
    a2.loglog(pares[dentro, 0], pares[dentro, 1], "o", ms=4.5, color=VERDE,
              zorder=3, markeredgecolor="white", markeredgewidth=0.6)
    guia = np.array([0.03, 60.0])
    a2.plot(guia, guia, "-", color=TINTA2, lw=0.7, ls="--", zorder=2)
    a2.annotate("1:1", (30, 34), fontsize=7, color=TINTA2, ha="right")
    a2.axvline(2.0, color=VERDE, lw=0.7, ls=":")
    a2.annotate(f"matched $\\mathbf{{D}}$ (within 2%):\n"
                f"$|\\Delta K_{{yy}}|\\leq{pares[dentro,1].max():.1f}\\%$",
                (0.045, 12), fontsize=7.5, color=VERDE, fontweight="bold")
    a2.set_xlabel(r"tensor mismatch  $\|\Delta\mathbf{D}\|/\|\mathbf{D}\|$  (%)")
    a2.set_ylabel(r"$|\Delta K_{yy}| / K_{yy}$   (%)")
    a2.set_title(r"(b)  all cross-path pairs, tr $\mathbf{D}>0.10$", loc="left")
    a2.set_xlim(0.03, 60)
    a2.set_ylim(0.005, 100)
    limpiar(a2, eje="both")

    fig.tight_layout()
    guardar(fig, "fig3_escalar_vs_tensor")


# --------------------------------------------------------------------- #
def figura4(R):
    """Cuanto se equivoca cada cierre concreto, con su MEJOR parametro."""
    from fftgk.cierres import calibrar_escalar

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    t = np.array([r["trD"] for r in R])
    K = np.array([r["Kyy"] for r in R])
    o = np.argsort(t)
    t, K = t[o], K[o]
    Rs = [R[i] for i in o]
    k0 = K[0]

    alpha, Kexp, _ = calibrar_escalar(lambda tt, a: k0 * np.exp(a * tt),
                                      t, K, 10.0, (0.5, 40.0))
    e_exp = (Kexp - K) / K * 100

    # cierre tensorial lineal, ajustado sobre Kxx y Kyy a la vez
    filas, rhs = [], []
    for r in Rs:
        tt = r["Dxx"] + r["Dyy"]
        for kk, dd in (("Kxx", "Dxx"), ("Kyy", "Dyy")):
            filas.append([1.0, tt, r[dd]])
            rhs.append(r[kk])
    A, b = np.array(filas), np.array(rhs)
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    e_ten = ((A @ p - b) / b * 100)[1::2]          # la componente yy
    # y el mismo ajuste sin el termino tensorial
    p2, *_ = np.linalg.lstsq(A[:, :2], b, rcond=None)
    e_esc = ((A[:, :2] @ p2 - b) / b * 100)[1::2]

    # Marcadores, no lineas: cada punto es un ESTADO, y a un mismo tr D llegan
    # varias trayectorias. Unirlos por orden de tr D dibujaria un zigzag que
    # solo refleja el orden del archivo.
    for e, col, mrk, eti in (
            (e_exp, ROJO, "o", r"$k_0e^{\alpha\,\mathrm{tr}\mathbf{D}}$"
                               f"  ($\\alpha$={alpha:.1f}, best fit)"),
            (e_esc, AMBAR, "s", r"linear in tr $\mathbf{D}$"),
            (e_ten, VERDE, "^", r"linear in $\mathbf{D}_{ij}$")):
        ax.plot(t, e, mrk, ms=3.6, color=col, label=eti, alpha=0.85,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.axhline(0, color=TINTA2, lw=0.7)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([-100, -10, 0, 10, 100])
    ax.set_yticklabels(["-100", "-10", "0", "10", "100"])
    ax.set_xlabel(r"crack density  tr $\mathbf{D}$")
    ax.set_ylabel("closure error  (%)")
    ax.legend(loc="upper left")
    limpiar(ax)
    fig.tight_layout()
    guardar(fig, "fig4_cierres")


def guardar(fig, nombre):
    for ext in ("pdf", "png"):
        fig.savefig(SAL / f"{nombre}.{ext}")
    plt.close(fig)
    print(nombre)


if __name__ == "__main__":
    d = cargar()
    R = registros()
    figura1(d)
    figura2(d)
    figura3(d, R)
    figura4(R)
    print("->", SAL)
