"""
Ecuaciones del manuscrito compuestas con tipografia matematica (STIX, la
familia de Times) y exportadas a PNG de 400 dpi con fondo transparente.

Por que imagen y no OMML
------------------------
La primera version las escribio como OMML nativo de Word. El XML es valido --
pandoc lo lee y lo traduce a LaTeX correcto -- pero ni LibreOffice ni el propio
pandoc renderizan los operadores grandes (la integral sale vacia), y desde aqui
no hay forma de comprobar que Word las muestre bien. Una ecuacion que no se
puede verificar no se entrega. Compuestas asi, se ven identicas en Word, en
LibreOffice y en el PDF, con barra de fraccion horizontal y subindices y
superindices de verdad.

Uso:  python3 scripts/ecuaciones.py     -> figuras_eq/eq01.png ... eq15.png
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAL = pathlib.Path(__file__).resolve().parent.parent / "figuras_eq"
SAL.mkdir(exist_ok=True)

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "text.color": "#000000",
})

ECS = [
    # 1 energia regularizada
    r"$E[u,\varphi]=\int_Y\left\{\left[(1-\varphi)^2+\eta\right]\,"
    r"\psi^{+}(\varepsilon)+\psi^{-}(\varepsilon)\right\}dV"
    r"+\int_Y G_c\,\gamma_\ell\,dV$",
    # 2 densidad de superficie AT2
    r"$\gamma_\ell=\dfrac{\varphi^{2}}{2\ell}"
    r"+\dfrac{\ell}{2}\left|\nabla\varphi\right|^{2}$",
    # 3 irreversibilidad y ecuacion de dano
    r"$H(x,t)=\max_{s\leq t}\psi^{+}\!\left(\varepsilon(x,s)\right),"
    r"\qquad G_c\left[\dfrac{\varphi}{\ell}-\ell\,\Delta\varphi\right]"
    r"=2(1-\varphi)\,H$",
    # 4 control por longitud de grieta
    r"$\Gamma\!\left(\varphi(\lambda)\right)=\Gamma_{n-1}+\Delta\Gamma,"
    r"\qquad \Gamma=\dfrac{1}{|Y|}\int_Y\dfrac{\varphi^{2}}{\ell}\,dV$",
    # 5 tensor de dano
    r"$D_{ij}=\Gamma\,N_{ij},\qquad N_{ij}="
    r"\dfrac{\int_Y\partial_i\varphi\,\partial_j\varphi\,dV}"
    r"{\int_Y\left|\nabla\varphi\right|^{2}dV}$",
    # 6 problema de celda de Darcy
    r"$\nabla\cdot\left[k(x)\left(\nabla\tilde{p}+G\right)\right]=0,"
    r"\qquad k(x)=k_m\left[1+(\chi-1)\varphi^{2}\right]$",
    # 7 tensor efectivo
    r"$\langle q\rangle=-\,K\cdot G,\qquad \langle q\rangle="
    r"\dfrac{1}{|Y|}\int_Y k(x)\left(\nabla\tilde{p}+G\right)dV$",
    # 8 piso del cierre escalar
    r"$S(t)=\dfrac{\max_{K\in\mathcal{S}_t}K-\min_{K\in\mathcal{S}_t}K}"
    r"{\mathrm{mean}_{K\in\mathcal{S}_t}K},\qquad "
    r"\mathcal{S}_t=\left\{K:\mathrm{tr}\,D=t\right\}$",
    # 9 criterio de tensor igualado
    r"$\dfrac{\left\|D^{a}-D^{b}\right\|}{\left\|D^{a}\right\|}<0.02,"
    r"\qquad \delta K=\dfrac{\left|K^{a}-K^{b}\right|}{K^{a}}$",
    # 10 los dos cierres que se prueban
    r"$k=k_0\,e^{\alpha\,\mathrm{tr}\,D},\qquad "
    r"K_{ij}=K_0\,\delta_{ij}+a\,(\mathrm{tr}\,D)\,\delta_{ij}+b\,D_{ij}$",
    # 11 ley cubica y Oda-Snow
    r"$K_{\mathrm{cl}}=\dfrac{\Gamma\,\langle b\rangle^{3}}{12},\qquad "
    r"K_{\mathrm{Oda}}=\dfrac{1}{12}\left[\mathrm{tr}\,D^{(3)}\,I"
    r"-D^{(3)}\right]$",
    # 12 volumen de grieta y apertura media
    r"$V_c=-\int_Y u\cdot\nabla\varphi\,dV,\qquad "
    r"\langle b\rangle=\dfrac{V_c}{\Gamma}$",
    # 13 Sneddon y convergencia
    r"$V_s=\dfrac{2\pi\left(1-\nu^{2}\right)\sigma\,a^{2}}{E},\qquad "
    r"\dfrac{V_c}{V_s}=1+1.81\left(\dfrac{\ell}{a}\right)^{1.11}$",
    # 14 conductividad de apertura
    r"$\dfrac{k_b(x)}{k_m}=1+\Lambda\,\dfrac{b^{3}}{12\,w},\qquad "
    r"\Lambda=\dfrac{L^{2}}{k_m}$",
    # 15 factor de correccion
    r"$\dfrac{K_b}{k_m}-1=c\left(\Lambda\,K_{\mathrm{Oda}}\right)^{m},"
    r"\qquad c=1.48\times10^{-3},\qquad m=0.26$",
]


# Fracciones que aparecen DENTRO del texto corrido. Van tambien con barra
# horizontal, por encargo, asi que se componen igual y se insertan en linea.
INLINE = {
    "l_a": r"$\frac{\ell}{a}$",
    "l_h": r"$\frac{\ell}{h}$",
    "G_Gloc": r"$\frac{\Gamma}{\Gamma_{\mathrm{loc}}}$",
    "L24": r"$\frac{L}{24}$",
    "Dyy_trD": r"$\frac{D_{yy}}{\mathrm{tr}\,D}$",
    "medio": r"$\frac{1}{2}$",
}


def main(dpi=400, tam=11):
    for i, tex in enumerate(ECS, start=1):
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, tex, fontsize=tam)
        nombre = SAL / f"eq{i:02d}.png"
        fig.savefig(nombre, dpi=dpi, transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        from PIL import Image
        w, h = Image.open(nombre).size
        # a 400 dpi, el ancho en puntos es w/400*72; Word trabaja a 96 dpi
        print(f"eq{i:02d}  {w}x{h}px  ->  {w*96//dpi} x {h*96//dpi} px Word")


def inline(dpi=400, tam=10):
    from PIL import Image
    for nombre, tex in INLINE.items():
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, tex, fontsize=tam)
        ruta = SAL / f"in_{nombre}.png"
        fig.savefig(ruta, dpi=dpi, transparent=True, bbox_inches="tight",
                    pad_inches=0.01)
        plt.close(fig)
        w, h = Image.open(ruta).size
        print(f"in_{nombre}  {w}x{h}px")


if __name__ == "__main__":
    main()
    inline()
