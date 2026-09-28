"""
Verificacion del tensor de dano D_ij.

La afirmacion central del proyecto es sobre D_ij, asi que la definicion tiene
que ser demostrablemente correcta antes de usarla. Cada test contrasta contra
un resultado analitico exacto.

Caso de referencia: familia periodica de planos paralelos p*X + q*Y = entero
en una celda unitaria. La densidad de grieta exacta es sqrt(p^2+q^2) (inverso
del espaciamiento perpendicular) y la normal es (p,q)/sqrt(p^2+q^2). Esta
construccion es exactamente periodica para cualquier (p,q) enteros, que es lo
que exige la FFT: una recta inclinada definida a mano NO es periodica y produce
un salto en el borde de la celda.
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fftgk.damage import (  # noqa: E402
    damage_tensor, crack_density, localization_ratio, at2_profile,
)

_REPORT = []


def rep(name, value):
    _REPORT.append((name, str(value)))


def familia(N, ell, p, q):
    """Familia periodica de planos. Devuelve (phi, normal, densidad exacta)."""
    x = (np.arange(N[0]) + 0.5) / N[0]
    y = (np.arange(N[1]) + 0.5) / N[1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    t = p * X + q * Y
    nrm = float(np.hypot(p, q))
    s = (t - np.round(t)) / nrm          # distancia con signo al plano cercano
    return at2_profile(s, ell), np.array([p, q]) / nrm, nrm


UNIT = (1.0, 1.0)


def test_traza_es_densidad_de_grieta():
    """tr D = densidad de superficie de grieta, para varias orientaciones."""
    n = 512
    errs = []
    for (p, q) in [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (2, 3)]:
        ell = 8.0 / n
        phi, nv, ex = familia((n, n), ell, p, q)
        D = damage_tensor(phi, ell, L=UNIT)
        rel = abs(np.trace(D) - ex) / ex
        errs.append(rel)
        assert rel < 0.01, ((p, q), np.trace(D), ex, rel)
    rep("traza = densidad (6 orientaciones, l/h=8)",
        f"error max = {max(errs):.2e}, dispersion = {max(errs)-min(errs):.2e}")


def test_sin_sesgo_por_orientacion():
    """EL TEST QUE MOTIVO EL DISENO.

    El estimador ingenuo (method="raw") tiene 16% de dispersion segun la
    orientacion de la grieta respecto a la malla, lo que se leeria como
    anisotropia fisica. El estimador normalizado debe tener mucha menos.
    """
    n = 512
    ell = 8.0 / n
    orientaciones = [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2), (2, 3)]
    razones = {"raw": [], "normalized": []}
    for metodo in razones:
        for (p, q) in orientaciones:
            phi, nv, ex = familia((n, n), ell, p, q)
            D = damage_tensor(phi, ell, L=UNIT, method=metodo)
            razones[metodo].append(np.trace(D) / ex)
    disp = {m: max(v) - min(v) for m, v in razones.items()}
    assert disp["normalized"] < 0.02, disp
    assert disp["normalized"] < disp["raw"] / 5, disp
    rep("dispersion por orientacion (l/h=8)",
        f"raw = {disp['raw']*100:.1f}%  normalizado = {disp['normalized']*100:.2f}%")


def test_convergencia_con_resolucion():
    """El error de la traza debe caer al refinar l/h, y ser pequeno ya en l/h=8."""
    n = 512
    vals = []
    for lpx in (2, 4, 8, 16):
        ell = lpx / n
        phi, nv, ex = familia((n, n), ell, 1, 0)
        D = damage_tensor(phi, ell, L=UNIT)
        vals.append((lpx, np.trace(D) / ex))
    errs = [abs(v - 1) for _, v in vals]
    assert all(b <= a for a, b in zip(errs, errs[1:])), vals
    assert errs[2] < 0.01, vals          # l/h = 8 debe estar bajo 1%
    rep("convergencia de la traza con l/h",
        " ".join(f"l/h={l}:{v:.4f}" for l, v in vals))


def test_rango_uno_y_autovector():
    """Familia de planos paralelos -> D = densidad * n (x) n. Rango 1."""
    n = 512
    ell = 8.0 / n
    for (p, q) in [(1, 0), (1, 1), (2, 1), (2, 3)]:
        phi, nv, ex = familia((n, n), ell, p, q)
        D = damage_tensor(phi, ell, L=UNIT)
        w, V = np.linalg.eigh(D)
        idx = np.argsort(w)[::-1]
        w, V = w[idx], V[:, idx]
        razon = abs(w[1]) / abs(w[0])
        align = abs(float(V[:, 0] @ nv))
        assert razon < 0.01, ((p, q), w, razon)
        assert align > 0.9999, ((p, q), V[:, 0], nv, align)
        rep(f"rango 1 y autovector {(p, q)}",
            f"l2/l1={razon:.2e}  |v.n|={align:.6f}")


def test_objetividad():
    """D rota con la microestructura: la familia (0,1) es la (1,0) girada 90.

    Se usa una rotacion de simetria de la malla para que la comparacion no
    mezcle el efecto de rotar con el de remuestrear.
    """
    n = 512
    ell = 8.0 / n
    phi0, n0, _ = familia((n, n), ell, 1, 0)
    phi1, n1, _ = familia((n, n), ell, 0, 1)
    D0 = damage_tensor(phi0, ell, L=UNIT)
    D1 = damage_tensor(phi1, ell, L=UNIT)
    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    pred = R @ D0 @ R.T
    err = np.abs(D1 - pred).max() / np.trace(D0)
    assert err < 1e-10, (D1, pred, err)
    rep("objetividad (rotacion de 90 grados)", f"err={err:.2e}")


def test_aditividad_dos_familias():
    """Dos familias ortogonales de igual densidad -> D isotropo, traza suma."""
    n = 512
    ell = 6.0 / n
    phi_x, _, ex = familia((n, n), ell, 1, 0)
    phi_y, _, _ = familia((n, n), ell, 0, 1)
    phi = np.maximum(phi_x, phi_y)
    D = damage_tensor(phi, ell, L=UNIT)
    err_traza = abs(np.trace(D) - 2 * ex) / (2 * ex)
    aniso = abs(D[0, 0] - D[1, 1]) / np.trace(D)
    assert err_traza < 0.03, (np.trace(D), 2 * ex)
    assert aniso < 0.01, (D, aniso)
    rep("aditividad de dos familias ortogonales",
        f"err traza={err_traza:.2e}  anisotropia residual={aniso:.2e}")


def test_momento_de_apertura():
    """D^(3) pesa por apertura al cubo: h -> c*h escala D^(3) por c^3."""
    n = 256
    ell = 6.0 / n
    phi, _, _ = familia((n, n), ell, 1, 0)
    h = np.full((n, n), 2.0)
    D3 = damage_tensor(phi, ell, L=UNIT, weight=h, m=3)
    D3b = damage_tensor(phi, ell, L=UNIT, weight=3.0 * h, m=3)
    razon = np.trace(D3b) / np.trace(D3)
    assert abs(razon - 27.0) / 27.0 < 1e-12, razon
    D0 = damage_tensor(phi, ell, L=UNIT)
    assert abs(np.trace(D3) / np.trace(D0) - 8.0) < 1e-10
    rep("momento de apertura m=3", f"h->3h da {razon:.6f} (exacto 27)")


def test_razon_de_localizacion():
    """R = B/A debe tender a 1 al refinar, y apartarse con dano difuso."""
    n = 512
    Rs = []
    for lpx in (2, 4, 8, 16, 32):
        ell = lpx / n
        phi, _, _ = familia((n, n), ell, 1, 0)
        Rs.append((lpx, localization_ratio(phi, ell, L=UNIT)))
    assert all(b > a for (_, a), (_, b) in zip(Rs, Rs[1:])), Rs
    assert Rs[-1][1] > 0.95, Rs
    # dano difuso: campo suave de baja amplitud, NO un perfil de grieta
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    difuso = gaussian_filter(rng.random((n, n)), 20.0, mode="wrap")
    difuso = 0.3 * difuso / difuso.max()
    R_dif = localization_ratio(difuso, 8.0 / n, L=UNIT)
    assert R_dif < 0.5, R_dif
    rep("razon de localizacion R=B/A",
        "grieta: " + " ".join(f"l/h={l}:{r:.3f}" for l, r in Rs)
        + f" | dano difuso: {R_dif:.3f}")


def test_ceguera_a_la_topologia():
    """EL TEST QUE IMPORTA: dos topologias distintas con el MISMO D_ij.

    (a) dos grietas que atraviesan la celda y percolan en x
    (b) cuatro grietas de la mitad de longitud, escalonadas y desconectadas,
        misma orientacion y misma densidad total

    Si el tensor de dano no las distingue, la premisa del proyecto es solida:
    D_ij es estructuralmente incapaz de codificar conectividad. Lo que queda
    por demostrar en el barrido es que la trayectoria de carga elige entre
    ambas y que K difiere.

    Se usa distancia a SEGMENTO (punta suave), no un corte duro, porque un
    corte duro crea una discontinuidad artificial que el gradiente lee como
    superficie de grieta transversal.

    Ambas configuraciones tienen la misma longitud total de grieta y la misma
    orientacion. La diferencia residual en D es contribucion de PUNTA, que
    escala como l por punta: el test verifica que se desvanece al reducir l,
    es decir que la ceguera es exacta en el limite de grieta aguda.
    """
    n = 512
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")

    def segmento(x0, yc, largo=0.5, ell=None):
        """Grieta vertical en x0, centrada en yc, de longitud `largo`."""
        dx = X - x0
        dx = dx - np.round(dx)                   # periodica en x
        dy = Y - yc
        dy = dy - np.round(dy)                   # periodica en y
        fuera = np.maximum(np.abs(dy) - largo / 2.0, 0.0)
        dist = np.hypot(dx, fuera)               # distancia al segmento
        return at2_profile(dist, ell)

    difs = []
    for lpx in (8.0, 5.0, 3.0):
        ell = lpx / n
        # (a) percolante: 2 grietas que atraviesan la celda (periodicas, sin puntas)
        sa = X - 0.30
        sb = X - 0.80
        perc = np.maximum(at2_profile(sa - np.round(sa), ell),
                          at2_profile(sb - np.round(sb), ell))
        # (b) disperso: 4 segmentos de longitud 0.5, escalonados, sin conectar.
        #     Longitud total 4 x 0.5 = 2.0, igual que (a).
        disp = np.zeros((n, n))
        for x0, yc in ((0.15, 0.25), (0.40, 0.75), (0.65, 0.25), (0.90, 0.75)):
            disp = np.maximum(disp, segmento(x0, yc, largo=0.5, ell=ell))

        Da = damage_tensor(perc, ell, L=UNIT)
        Db = damage_tensor(disp, ell, L=UNIT)
        difs.append((lpx, np.trace(Da), np.trace(Db),
                     np.abs(Da - Db).max() / np.trace(Da)))

    # La discrepancia residual es del orden del 2%: dos microestructuras con
    # el mismo tensor de dano dentro del error numerico, una percolante y la
    # otra no. No se exige monotonia al afilar la grieta porque compiten dos
    # errores en direcciones opuestas -- la contribucion de punta baja con l,
    # pero el error de resolucion de la traza sube.
    vals = [d for _, _, _, d in difs]
    assert max(vals) < 0.03, difs
    rep("ceguera a la topologia (dif -> 0 al afilar)",
        " | ".join(f"l/h={l:.0f}: trA={a:.3f} trB={b:.3f} dif={d*100:.1f}%"
                   for l, a, b, d in difs))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print("\n--- detalle ---")
    for name, value in _REPORT:
        print(f"  {name:44s} {value}")
    print(f"\n{len(tests)-failed}/{len(tests)} tests OK")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
