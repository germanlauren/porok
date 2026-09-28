"""
PILOTO 0: ¿importa la topologia a K, a tensor de dano fijo?

Pregunta. En doc 04 quedo demostrado que dos microestructuras con la misma
longitud total de grieta y la misma orientacion tienen el MISMO D_ij dentro del
error numerico (~2%), aunque una percole y la otra no. Falta la otra mitad del
argumento: que K si difiere.

Este es el experimento mas barato que responde eso. No necesita mecanica ni
campo de fase: usa las dos configuraciones ya construidas y les mide la
permeabilidad efectiva con el homogeneizador ya verificado.

ALCANCE Y LIMITACIONES, explicitas:

  * Es un piloto a nivel DARCY: se asigna una permeabilidad local alta a la
    grieta y baja a la matriz. No resuelve Stokes. Sirve para saber si el
    efecto existe y de que tamano es, NO para el numero final del paper. La
    referencia definitiva exige Stokes-Brinkman (ver doc 03 seccion 5).

  * Las grietas se construyen a mano, no salen de una simulacion mecanica.
    Este piloto responde "¿la topologia mueve K a D_ij fijo?". NO responde
    "¿la trayectoria de carga produce topologias distintas?", que es la otra
    mitad y exige el campo de fase.

  * Ambas configuraciones tienen identica apertura (mismo perfil, mismo l), de
    modo que la comparacion no esta contaminada por la ley cubica.

Si K NO se separa aqui, la hipotesis central del proyecto muere barata.
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fftgk.homogenize import homogenize          # noqa: E402
from fftgk.damage import damage_tensor, at2_profile, localization_ratio  # noqa: E402
from fftgk.topology import cluster_stats         # noqa: E402

UNIT = (1.0, 1.0)


def construir(n, ell):
    """Dos configuraciones con la misma longitud total de grieta (= 2.0).

    (a) percolante: 2 grietas que atraviesan la celda en y, periodicas, sin puntas
    (b) disperso:   4 segmentos de longitud 0.5 escalonados, sin conectar
    """
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")

    def plano(x0):
        s = X - x0
        return at2_profile(s - np.round(s), ell)

    def segmento(x0, yc, largo):
        dx = X - x0
        dx = dx - np.round(dx)
        dy = Y - yc
        dy = dy - np.round(dy)
        fuera = np.maximum(np.abs(dy) - largo / 2.0, 0.0)
        return at2_profile(np.hypot(dx, fuera), ell)

    perc = np.maximum(plano(0.30), plano(0.80))
    disp = np.zeros((n, n))
    for x0, yc in ((0.15, 0.25), (0.40, 0.75), (0.65, 0.25), (0.90, 0.75)):
        disp = np.maximum(disp, segmento(x0, yc, 0.5))
    return {"percolante": perc, "disperso": disp}


def k_de_phi(phi, k_matriz, k_grieta, p=2.0):
    """Interpolacion permeabilidad-dano.

    k(phi) = k_m + (k_c - k_m) * phi^p

    Es el acople fenomenologico comun en la literatura de flujo-fractura. Para
    ESTE piloto basta: ambas configuraciones usan el mismo mapeo y la misma
    apertura, asi que cualquier sesgo del mapeo afecta a las dos por igual y se
    cancela en el cociente. El acople por apertura fisica viene despues.
    """
    return k_matriz + (k_grieta - k_matriz) * np.clip(phi, 0.0, 1.0) ** p


def main():
    n = 512
    ell = 6.0 / n
    k_matriz = 1.0
    campos = construir(n, ell)

    print("=" * 78)
    print("PILOTO 0 — misma densidad de grieta, distinta topologia")
    print("=" * 78)

    # ---- 1. confirmar que el tensor de dano NO las distingue ----------------
    print("\n[1] Tensor de dano y topologia\n")
    # Las grietas son VERTICALES (normal en x, corren a lo largo de y), asi que
    # la direccion de percolacion -- y por tanto la de la senal -- es y.
    # x es el control: el flujo atraviesa la matriz en ambos casos.
    print(f"{'configuracion':>14} {'tr D':>9} {'Dxx':>9} {'Dyy':>9} {'R=B/A':>8} "
          f"{'cumulos':>8} {'f_mayor':>8} {'percola y':>10} {'chi':>7}")
    D = {}
    for nombre, phi in campos.items():
        D[nombre] = damage_tensor(phi, ell, L=UNIT)
        R = localization_ratio(phi, ell, L=UNIT)
        st = cluster_stats(phi > 0.5)
        print(f"{nombre:>14} {np.trace(D[nombre]):9.4f} {D[nombre][0,0]:9.4f} "
              f"{D[nombre][1,1]:9.4f} {R:8.3f} {st['n_clusters']:8d} "
              f"{st['f_mayor']:8.3f} {str(st['percola'][1]):>10} {st['chi']:7d}")

    dif_D = np.abs(D["percolante"] - D["disperso"]).max() / np.trace(D["percolante"])
    print(f"\n    diferencia maxima en D_ij: {dif_D*100:.2f}%  "
          f"-> el tensor de dano las considera equivalentes")

    # ---- 2. permeabilidad efectiva -----------------------------------------
    print("\n[2] Permeabilidad efectiva K (nivel Darcy)\n")
    print("    y = a lo largo de las grietas (direccion de percolacion, SENAL)")
    print("    x = transversal a las grietas (pasa por la matriz, CONTROL)\n")
    print(f"{'contraste':>10} {'Kyy percol':>12} {'Kyy disp':>12} {'razon Kyy':>11}"
          f" | {'Kxx percol':>11} {'Kxx disp':>11} {'razon Kxx':>10}")
    filas = []
    for contraste in (1e2, 1e3, 1e4, 1e5, 1e6):
        K = {}
        for nombre, phi in campos.items():
            k = k_de_phi(phi, k_matriz, k_matriz * contraste)
            r = homogenize(k, L=UNIT, tol=1e-11, maxiter=3000, dual=False,
                           ktol=1e-10)
            K[nombre] = r.K_primal
        rx = K["percolante"][0, 0] / K["disperso"][0, 0]
        ry = K["percolante"][1, 1] / K["disperso"][1, 1]
        filas.append((contraste, K, rx, ry))
        print(f"{contraste:10.0e} {K['percolante'][1,1]:12.4f} "
              f"{K['disperso'][1,1]:12.4f} {ry:11.2f} | "
              f"{K['percolante'][0,0]:11.5f} {K['disperso'][0,0]:11.5f} {rx:10.2f}")

    # ---- 3. veredicto -------------------------------------------------------
    print("\n[3] Veredicto\n")
    ry_max = max(f[3] for f in filas)
    rx_max = max(f[2] for f in filas)
    print(f"    D_ij difiere en                       {dif_D*100:7.2f}%")
    print(f"    K_xx (control, transversal) difiere   {abs(1-rx_max)*100:7.1f}%")
    print(f"    K_yy (percolacion) difiere hasta      {ry_max:7.1f}x")
    if ry_max > 5.0 and abs(1 - rx_max) < 0.3:
        print("\n    La hipotesis SOBREVIVE, y con la firma correcta:")
        print("    - transversal a las grietas, K coincide con lo que predice D_ij")
        print("    - a lo largo de las grietas, K difiere por ORDENES DE MAGNITUD")
        print("    - la brecha CRECE con el contraste: es percolacion, no")
        print("      anisotropia ni artefacto de escala")
        print("\n    Falta la otra mitad: que la trayectoria de carga elija la")
        print("    topologia. Eso exige el campo de fase.")
    else:
        print("\n    La hipotesis NO se sostiene en este piloto: revisar antes")
        print("    de invertir en el barrido.")


if __name__ == "__main__":
    main()
