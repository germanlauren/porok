"""
Verificacion de la apertura contra Sneddon, con los tres confundidores
controlados uno por uno.

Que salia mal en test_sneddon.py
--------------------------------
La grieta se sembraba en y=0.5, que con x=(i+0.5)/n cae ENTRE dos filas de la
malla. La distancia minima al nucleo es entonces h/2 y el perfil de AT2 da
phi_max = exp(-h/2l) = 0.92 con l/h=6, o sea g = (1-phi)^2 = 6.4e-3: la grieta
conserva medio por ciento de la rigidez de la matriz y NO esta abierta. Como el
espesor de la banda escala con l, al refinar la malla a l/h fijo la grieta se
vuelve mas rigida y se abre menos. Por eso V/Vs bajaba de 1.36 a 0.77 en vez de
converger: el limite no era el de grieta aguda, era el de grieta que se cierra.

Aqui el nucleo se pone SOBRE una fila (phi=1 exacto, g=1e-6).

Los otros dos confundidores
---------------------------
1. Imagenes periodicas. La celda es periodica, asi que no hay una grieta sino
   una red. Para la fila colineal el resultado exacto (Tada, Koiter) da un area
   de apertura

       A = (4(1-nu^2) sigma L^2 / (pi E)) ln sec(pi a / L)

   que sobre Sneddon es un factor f = ln sec(x) / (x^2/2), x = pi a / L. Con
   a/L = 0.05 vale 1.004, o sea el 0.4%: por eso conviene una grieta CORTA
   frente a la celda, aunque cueste malla.

2. Longitud de regularizacion. Una grieta regularizada se abre como una grieta
   aguda algo mas larga. El error es O(l/a), asi que el test correcto es la
   PENDIENTE: bajar l/a con l/h FIJO y extrapolar a l/a -> 0.

Se verifica el volumen y, ademas, la FORMA del perfil b(x), que es la parte que
de verdad importa para la ley cubica, porque esta pesa b^3 y una forma mala se
amplifica al cubo.

Uso:  python3 scripts/verif_apertura.py
"""
import json
import pathlib
import sys

import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "src"))

from fftgk.elasticity import (ElasticProjector, lame_from_E_nu,      # noqa: E402
                              solve_elasticity)
from fftgk.apertura import (volumen_grieta, campo_apertura,          # noqa: E402
                            sneddon_volumen, sneddon_perfil)
from fftgk.damage import at2_profile                                 # noqa: E402

E_MOD, NU = 1.0, 0.2
L = (1.0, 1.0)
LPX = 6.0            # l en pixeles: FIJO, para que la grieta siga resuelta
EPS_MACRO = 2e-4


def factor_periodico(a, largo=1.0):
    """Fila colineal de grietas: area exacta / area de Sneddon."""
    x = np.pi * a / largo
    return float(np.log(1.0 / np.cos(x)) / (x ** 2 / 2.0))


def celda(n, semi):
    """phi de una grieta recta centrada, con el nucleo SOBRE una fila."""
    ell = LPX / n
    ejes = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(ejes, ejes, indexing="ij")
    y0 = ejes[n // 2]                      # una fila de la malla, no el borde
    dx, dy = X - 0.5, Y - y0
    fuera = np.maximum(np.abs(dx) - semi, 0.0)
    dist = np.hypot(fuera, dy)
    return ell, at2_profile(dist, ell), X, Y, y0


def una(n, semi, cierre=1e-6):
    ell, phi, X, Y, y0 = celda(n, semi)
    g = (1.0 - phi) ** 2 + cierre
    lam0, mu0 = lame_from_E_nu(E_MOD, NU)
    Em = np.array([[0.0, 0.0], [0.0, EPS_MACRO]])
    P = ElasticProjector((n, n), L, "rotated")
    eps, sig, info = solve_elasticity(g * lam0, g * mu0, Em, L=L, tol=1e-12,
                                      stol=1e-13, projector=P, maxiter=20000)
    u = P.desplazamiento(eps - Em.reshape(2, 2, 1, 1))
    Vc = volumen_grieta(u, phi, L=L)
    sigma = float(sig.reshape(4, -1).mean(axis=1)[3])
    Vs = sneddon_volumen(sigma, semi, E_MOD, NU) * factor_periodico(semi)
    # Perfil b(x): la apertura local es la INTEGRAL TRANSVERSAL de la misma
    # densidad cuya integral total da el volumen, no el maximo del campo
    # b(x,y), que depende del umbral de enmascarado.
    from fftgk.damage import gradient
    gr = gradient(phi, L=L, projector=P)
    dens = -np.einsum("i...,i...->...", u, gr)
    perfil = dens.sum(axis=1) / n
    return dict(n=n, ell=ell, ell_a=ell / semi, phi_max=float(phi.max()),
                Vc=Vc, Vs=Vs, razon=Vc / Vs, sigma=sigma, iters=info["iters"],
                perfil=perfil, x=(np.arange(n) + 0.5) / n - 0.5)


def main():
    semi = 0.05
    print(f"grieta a = {semi}  (factor de imagenes periodicas "
          f"{factor_periodico(semi):.4f})")
    print(f"  {'N':>5} {'l/a':>7} {'phi_max':>8} {'V medido':>12} "
          f"{'V Sneddon':>12} {'V/Vs':>7} {'iters':>6}")
    filas = []
    for n in (192, 256, 384, 512, 768):
        r = una(n, semi)
        filas.append(r)
        print(f"  {r['n']:5d} {r['ell_a']:7.3f} {r['phi_max']:8.4f} "
              f"{r['Vc']:12.4e} {r['Vs']:12.4e} {r['razon']:7.4f} "
              f"{r['iters']:6d}", flush=True)

    x = np.array([r["ell_a"] for r in filas])
    y = np.array([r["razon"] for r in filas])
    c1, c0 = np.polyfit(x, y, 1)
    q2, q1, q0 = np.polyfit(x, y, 2)
    # Modelo con sentido fisico: la grieta regularizada se abre como una aguda
    # de semilongitud a_ef = a (1 + c l/a), y el volumen va como a_ef^2.
    from numpy.polynomial import polynomial as _P   # noqa: F401
    cc = np.polyfit(x, np.sqrt(y), 1)
    print(f"\n  lineal     V/Vs = {c0:.4f} + {c1:.4f}(l/a)        -> {c0:.4f}")
    print(f"  cuadratico V/Vs = {q0:.4f} + {q1:.4f}(l/a) + ...  -> {q0:.4f}")
    print(f"  longitud efectiva  sqrt(V/Vs) = {cc[1]:.4f} + {cc[0]:.4f}(l/a)"
          f"  -> V/Vs = {cc[1]**2:.4f}")
    c0 = cc[1] ** 2
    print(f"  extrapolacion adoptada a l/a -> 0 :  {c0:.4f}   "
          f"(error {abs(c0-1)*100:.2f}%)")

    # forma del perfil en la malla mas fina
    r = filas[-1]
    dentro = np.abs(r["x"]) < semi * 0.98
    bs = sneddon_perfil(r["x"], r["sigma"], semi, E_MOD, NU)
    med, teo = r["perfil"][dentro], bs[dentro]
    esc = float(np.dot(med, teo) / np.dot(teo, teo))    # mejor escala global
    resid = np.abs(med - esc * teo).max() / med.max() * 100
    print(f"\n  perfil b(x) en N={r['n']}: escala {esc:.3f}, "
          f"desviacion de forma maxima {resid:.1f}% del pico")
    print(f"  (la escala absorbe el mismo O(l/a) del volumen; lo que se "
          f"verifica aqui es la FORMA eliptica)")

    (AQUI.parent / "verif_apertura.json").write_text(json.dumps(dict(
        semi=semi, f_periodico=factor_periodico(semi),
        filas=[{k: v for k, v in r.items() if k not in ("perfil", "x")}
               for r in filas],
        extrapolacion=c0, pendiente=c1, escala_perfil=esc,
        desviacion_forma_pct=resid), indent=2))
    print("\n-> verif_apertura.json")


if __name__ == "__main__":
    main()
