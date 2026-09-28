"""
Suite de verificacion del nucleo FFT-Galerkin.

Cada test contrasta contra un resultado EXACTO o una cota rigurosa conocida,
no contra otra corrida del mismo codigo.
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fftgk.homogenize import homogenize  # noqa: E402


# ---------------------------------------------------------------------------
# Referencias analiticas
# ---------------------------------------------------------------------------

def hashin_shtrikman(k_vals, f_vals, d, k0):
    """Cota HS d-dimensional con medio de referencia k0.

    K^HS(k0) = [ <1/(k + (d-1) k0)> ]^{-1} - (d-1) k0
    k0 = max(k) -> cota superior ; k0 = min(k) -> cota inferior.
    """
    k_vals = np.asarray(k_vals, float)
    f_vals = np.asarray(f_vals, float)
    s = np.sum(f_vals / (k_vals + (d - 1) * k0))
    return 1.0 / s - (d - 1) * k0


def maxwell_2d(km, ki, f):
    """Resultado exacto del ensamble de cilindros recubiertos de Hashin (2D).
    Coincide con la cota HS y con el limite diluido de Maxwell a O(f^2)."""
    return km * (ki * (1 + f) + km * (1 - f)) / (ki * (1 - f) + km * (1 + f))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_medio_homogeneo():
    """k uniforme -> K = k I exactamente, primal y dual."""
    for N in [(32, 32), (16, 16, 16)]:
        d = len(N)
        k = np.full(N, 3.7)
        r = homogenize(k, tol=1e-12, ktol=1e-13)
        err_p = np.abs(r.K_primal - 3.7 * np.eye(d)).max()
        err_d = np.abs(r.K_dual - 3.7 * np.eye(d)).max()
        assert err_p < 1e-12, (N, err_p)
        assert err_d < 1e-12, (N, err_d)
        yield_report("medio homogeneo", N, max(err_p, err_d))


def test_laminado_2d():
    """Laminado: K_xx = media armonica, K_yy = media aritmetica (EXACTO)."""
    N = (64, 64)
    k1, k2 = 1.0, 100.0
    frac = 0.375  # fraccion de la fase k1
    n1 = int(round(frac * N[0]))
    k = np.empty(N)
    k[:n1, :] = k1
    k[n1:, :] = k2
    f1 = n1 / N[0]
    f2 = 1 - f1

    harm = 1.0 / (f1 / k1 + f2 / k2)
    arit = f1 * k1 + f2 * k2

    r = homogenize(k, tol=1e-13, maxiter=20000, ktol=1e-13)
    for name, K in (("primal", r.K_primal), ("dual", r.K_dual)):
        assert abs(K[0, 0] - harm) / harm < 1e-9, (name, K[0, 0], harm)
        assert abs(K[1, 1] - arit) / arit < 1e-9, (name, K[1, 1], arit)
        assert abs(K[0, 1]) < 1e-9 * arit, (name, K[0, 1])
    yield_report("laminado 2D (armonica/aritmetica)", N,
                 max(abs(r.K[0, 0] - harm) / harm, abs(r.K[1, 1] - arit) / arit))


def test_laminado_3d():
    """Mismo test en 3D: K_xx armonica, K_yy = K_zz aritmetica."""
    N = (24, 24, 24)
    k1, k2 = 0.5, 20.0
    n1 = 9
    k = np.empty(N)
    k[:n1] = k1
    k[n1:] = k2
    f1 = n1 / N[0]
    f2 = 1 - f1
    harm = 1.0 / (f1 / k1 + f2 / k2)
    arit = f1 * k1 + f2 * k2
    r = homogenize(k, tol=1e-13, maxiter=20000, ktol=1e-13)
    assert abs(r.K[0, 0] - harm) / harm < 1e-9
    assert abs(r.K[1, 1] - arit) / arit < 1e-9
    assert abs(r.K[2, 2] - arit) / arit < 1e-9
    yield_report("laminado 3D", N, abs(r.K[0, 0] - harm) / harm)


def test_cotas_hashin_shtrikman():
    """Microestructura aleatoria: K debe caer dentro de las cotas HS."""
    rng = np.random.default_rng(7)
    N = (128, 128)
    km, ki = 1.0, 50.0
    for f in (0.15, 0.35, 0.55):
        mask = rng.random(N) < f
        # suavizar para tener una microestructura con escala, no ruido blanco
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(mask.astype(float), 3.0, mode="wrap")
        thr = np.quantile(field, 1 - f)
        mask = field > thr
        fi = mask.mean()
        k = np.where(mask, ki, km)
        lo = hashin_shtrikman([km, ki], [1 - fi, fi], 2, km)
        hi = hashin_shtrikman([km, ki], [1 - fi, fi], 2, ki)
        r = homogenize(k, tol=1e-11, maxiter=5000, ktol=1e-11)
        Kiso = 0.5 * np.trace(r.K)
        assert lo - 1e-8 <= Kiso <= hi + 1e-8, (f, lo, Kiso, hi)
        # y el orden de las cotas variacionales
        assert np.trace(r.K_dual) <= np.trace(r.K_primal) + 1e-8
        yield_report(f"cotas HS (f={fi:.3f})", N,
                     f"{lo:.4f} <= {Kiso:.4f} <= {hi:.4f}, gap={r.gap:.2e}")


def test_limite_diluido_maxwell():
    """Arreglo periodico de circulos diluido -> formula de Maxwell/Hashin."""
    N = (256, 256)
    km, ki = 1.0, 0.0 + 1e-6  # inclusiones casi impermeables
    f_target = 0.04
    x = (np.arange(N[0]) + 0.5) / N[0] - 0.5
    y = (np.arange(N[1]) + 0.5) / N[1] - 0.5
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(f_target / np.pi)
    mask = X ** 2 + Y ** 2 < R ** 2
    f = mask.mean()
    k = np.where(mask, ki, km)
    r = homogenize(k, tol=1e-11, maxiter=20000, ktol=1e-11)
    ref = maxwell_2d(km, ki, f)
    Kiso = 0.5 * np.trace(r.K)
    rel = abs(Kiso - ref) / ref
    assert rel < 5e-3, (Kiso, ref, rel)
    # isotropia del arreglo cuadrado diluido
    assert abs(r.K[0, 0] - r.K[1, 1]) / Kiso < 1e-6
    yield_report("Maxwell diluido (f=%.4f)" % f, N,
                 f"K={Kiso:.6f} ref={ref:.6f} rel={rel:.2e}")


def _campo_lognormal(rng, N, corr_px, sigma_ln, aspect=1.0):
    """Campo lognormal periodico con correlacion gaussiana (anisotropa si aspect!=1).

    aspect = longitud de correlacion en x dividida por la de y.
    """
    from scipy.ndimage import gaussian_filter
    g = rng.standard_normal(N)
    sig = (corr_px * aspect, corr_px)
    g = gaussian_filter(g, sig, mode="wrap")
    g = (g - g.mean()) / g.std()
    return np.exp(sigma_ln * g)


def test_lognormal_media_geometrica():
    """Campo lognormal isotropo en 2D -> K_eff = media geometrica (EXACTO).

    Resultado clasico (Matheron; demostrado para campos lognormales isotropos
    estadisticamente en 2D): K_eff = exp(<ln k>), independiente de la varianza.
    Es la prueba mas exigente de la suite porque fija un NUMERO, no una cota, y
    porque el campo es continuo -- no hay pixelado que enmascare el error del
    esquema. Ademas es el caso de referencia que usa la propia literatura de
    yacimientos, asi que verifica el codigo en su lenguaje.
    """
    rng = np.random.default_rng(2024)
    N = (192, 192)
    for sigma_ln in (0.5, 1.5):
        M = 10
        Ks, geos = [], []
        for _ in range(M):
            k = _campo_lognormal(rng, N, 4.0, sigma_ln)
            r = homogenize(k, tol=1e-11, maxiter=3000, ktol=1e-11, dual=False)
            Ks.append(r.K_primal)
            geos.append(np.exp(np.log(k).mean()))
        Ks = np.array(Ks)
        Kbar = Ks.mean(axis=0)
        Kiso = 0.5 * np.trace(Kbar)
        geo = np.mean(geos)
        # error estandar del estimador de ensamble
        se = (0.5 * (Ks[:, 0, 0] + Ks[:, 1, 1])).std(ddof=1) / np.sqrt(M)
        rel = abs(Kiso - geo) / geo
        aniso = abs(Kbar[0, 0] - Kbar[1, 1]) / Kiso
        shear = abs(Kbar[0, 1]) / Kiso
        assert rel < 0.02, (sigma_ln, Kiso, geo, rel)
        assert abs(Kiso - geo) < 3 * se + 0.01 * geo, (sigma_ln, Kiso, geo, se)
        assert aniso < 0.04 and shear < 0.04, (sigma_ln, aniso, shear)
        yield_report(f"lognormal 2D, media geometrica (s={sigma_ln})", N,
                     f"K={Kiso:.5f} geo={geo:.5f} rel={rel:.2e} "
                     f"(+-{3*se/geo:.1e} 3se) aniso={aniso:.3f} cizalla={shear:.3f}")


def test_anisotropia_estadistica_ordenada():
    """Correlacion alargada en x -> K_xx > K_yy, monotono en la razon de aspecto.

    Comprueba que el codigo captura anisotropia INDUCIDA POR LA ESTRUCTURA con
    el signo correcto, que es justo lo que el barrido tiene que medir.
    """
    rng = np.random.default_rng(5)
    N = (192, 192)
    ratios = []
    for aspect in (1.0, 2.0, 4.0, 8.0):
        acc = np.zeros((2, 2))
        M = 5
        for _ in range(M):
            k = _campo_lognormal(rng, N, 3.0, 1.2, aspect=aspect)
            r = homogenize(k, tol=1e-11, maxiter=3000, ktol=1e-11, dual=False)
            acc += r.K_primal
        K = acc / M
        ratios.append(K[0, 0] / K[1, 1])
    assert ratios[0] == max(min(ratios[0], 1.05), 0.95), ratios
    assert all(b > a for a, b in zip(ratios, ratios[1:])), ratios
    yield_report("anisotropia inducida (Kxx/Kyy)", N,
                 " ".join(f"a={a}:{r:.3f}" for a, r in
                          zip((1, 2, 4, 8), ratios)))


def test_anisotropia_local_tensorial():
    """k local tensorial constante y rotado -> K = k exactamente."""
    N = (32, 32)
    th = np.deg2rad(30.0)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    kloc = Rm @ np.diag([5.0, 0.5]) @ Rm.T
    k = np.zeros((2, 2) + N)
    k[:, :] = kloc[:, :, None, None]
    r = homogenize(k, tol=1e-12, ktol=1e-13)
    err = np.abs(r.K_primal - kloc).max() / np.abs(kloc).max()
    assert err < 1e-11, (r.K_primal, kloc)
    yield_report("k tensorial uniforme rotado", N, err)


def test_consistencia_primal_dual():
    """Primal y dual son duales algebraicos exactos: deben COINCIDIR.

    No es una cota: es un test de correccion del codigo. Rompe si hay un error
    en un proyector, en la inversion de k o en el ensamblaje de K.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    N = (96, 96)
    for contrast in (1e2, 1e4, 1e6):
        g = gaussian_filter(rng.standard_normal(N), 3.0, mode="wrap")
        k = np.where(g > 0, 1.0, 1.0 / contrast)
        r = homogenize(k, tol=1e-12, maxiter=4000, ktol=1e-12)
        assert r.gap < 1e-8, (contrast, r.gap)
        yield_report(f"consistencia primal-dual (c={contrast:.0e})", N,
                     f"gap={r.gap:.2e}")


def test_convergencia_de_malla():
    """K debe converger al refinar, con orden ~1 (frontera escalonada).

    La microestructura se define de forma CONTINUA (un circulo), de modo que
    refinar la malla refina tambien la geometria; el error observado es el de
    discretizacion real, no el de un pixelado fijo.
    """
    vals = {}
    for scheme in ("rotated", "fd"):
        Ks = []
        for n in (64, 128, 256, 512):
            x = (np.arange(n) + 0.5) / n - 0.5
            X, Y = np.meshgrid(x, x, indexing="ij")
            k = np.where(X ** 2 + Y ** 2 < 0.3 ** 2, 1e-6, 1.0)
            r = homogenize(k, tol=1e-11, maxiter=3000, dual=False,
                           scheme=scheme, ktol=1e-11)
            Ks.append(0.5 * np.trace(r.K_primal))
        vals[scheme] = Ks
        # extrapolacion de Richardson asumiendo orden 1
        ext = Ks[-1] + (Ks[-1] - Ks[-2])
        errs = [abs(K - ext) / ext for K in Ks]
        assert errs[1] < errs[0] and errs[2] < errs[1] and errs[3] < errs[2], (scheme, Ks)
        ratios = [errs[i] / errs[i + 1] for i in range(2)]
        yield_report(f"convergencia de malla ({scheme})", "64->512",
                     "K=" + " ".join(f"{K:.6f}" for K in Ks)
                     + f" | razon de error ~{np.mean(ratios):.2f} (orden 1 => 2)")
    # los dos esquemas deben converger al MISMO limite continuo
    d = abs(vals["rotated"][-1] - vals["fd"][-1]) / vals["rotated"][-1]
    assert d < 5e-3, (vals, d)
    yield_report("acuerdo entre esquemas (n=512)", 512, f"dif rel = {d:.2e}")


# ---------------------------------------------------------------------------

_REPORT = []


def yield_report(name, N, value):
    _REPORT.append((name, str(N), str(value)))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            res = t()
            if hasattr(res, "__iter__"):
                list(res)
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print("\n--- detalle ---")
    for name, N, value in _REPORT:
        print(f"  {name:38s} N={N:12s} {value}")
    print(f"\n{len(tests)-failed}/{len(tests)} tests OK")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
