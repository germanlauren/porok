"""
Verificacion del solver de elasticidad FFT.

Misma disciplina que el resto: cada test contrasta contra un resultado exacto o
una cota rigurosa, nunca contra otra corrida del propio codigo.
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fftgk.elasticity import (  # noqa: E402
    ElasticProjector, effective_stiffness, solve_elasticity,
    apply_C, lame_from_E_nu,
)

_REPORT = []


def rep(name, value):
    _REPORT.append((name, str(value)))


def C_iso_voigt(lam, mu, d=2):
    """Rigidez isotropa en base de Mandel."""
    s2 = np.sqrt(2.0)
    if d == 2:
        C = np.array([[lam + 2 * mu, lam, 0.0],
                      [lam, lam + 2 * mu, 0.0],
                      [0.0, 0.0, 2 * mu]])
        return C
    raise NotImplementedError


def test_proyector_es_idempotente():
    """Proj debe ser un proyector ortogonal: Proj^2 = Proj y simetrico."""
    rng = np.random.default_rng(1)
    for N in [(32, 32), (12, 12, 12)]:
        d = len(N)
        P = ElasticProjector(N)
        a = rng.standard_normal((d, d) + N)
        a = 0.5 * (a + np.swapaxes(a, 0, 1))     # simetrizar
        p1 = P.project(a)
        p2 = P.project(p1)
        idem = np.abs(p2 - p1).max() / np.abs(p1).max()
        # simetria: <Pa, b> = <a, Pb>
        b = rng.standard_normal((d, d) + N)
        b = 0.5 * (b + np.swapaxes(b, 0, 1))
        s1 = float(np.vdot(P.project(a), b).real)
        s2 = float(np.vdot(a, P.project(b)).real)
        sim = abs(s1 - s2) / max(abs(s1), 1e-300)
        assert idem < 1e-12, (N, idem)
        assert sim < 1e-12, (N, sim)
        rep(f"proyector idempotente y simetrico {N}",
            f"Proj^2-Proj = {idem:.2e}, asimetria = {sim:.2e}")


def test_medio_homogeneo():
    """Material uniforme -> C_eff = C exactamente, y campo local uniforme."""
    N = (32, 32)
    lam, mu = lame_from_E_nu(10.0, 0.25)
    C = effective_stiffness(np.full(N, lam), np.full(N, mu), tol=1e-13, stol=1e-14)
    ref = C_iso_voigt(lam, mu)
    err = np.abs(C - ref).max() / np.abs(ref).max()
    assert err < 1e-12, (C, ref, err)
    # el campo de deformacion debe ser exactamente el macroscopico
    E = np.array([[1.0, 0.0], [0.0, 0.0]])
    eps, sig, info = solve_elasticity(np.full(N, lam), np.full(N, mu), E,
                                      tol=1e-13, stol=1e-14)
    fluct = np.abs(eps - E.reshape(2, 2, 1, 1)).max()
    assert fluct < 1e-12, fluct
    rep("medio homogeneo", f"err C = {err:.2e}, fluctuacion = {fluct:.2e}")


def test_laminado_backus():
    """Laminado de dos capas isotropas: promedio exacto de Backus.

    Con capas perpendiculares a x, son continuas sigma_xx, sigma_xy y eps_yy.
    De ahi salen las formulas exactas (Backus 1962 / Postma):

        C11* = <1/C11>^-1
        C12* = <1/C11>^-1 <C12/C11>
        C22* = <C22 - C12^2/C11> + <1/C11>^-1 <C12/C11>^2
        C66* = <1/C66>^-1

    Es el test mas exigente de esta suite: fija cuatro numeros exactos.
    """
    N = (64, 64)
    n1 = 24
    f1 = n1 / N[0]
    f2 = 1 - f1
    lam1, mu1 = lame_from_E_nu(1.0, 0.30)
    lam2, mu2 = lame_from_E_nu(50.0, 0.20)

    lam = np.empty(N); lam[:n1] = lam1; lam[n1:] = lam2
    mu = np.empty(N); mu[:n1] = mu1; mu[n1:] = mu2

    C11 = np.array([lam1 + 2 * mu1, lam2 + 2 * mu2])
    C12 = np.array([lam1, lam2])
    C22 = C11.copy()
    C66 = np.array([mu1, mu2])
    f = np.array([f1, f2])

    inv11 = 1.0 / np.sum(f / C11)
    C11s = inv11
    C12s = inv11 * np.sum(f * C12 / C11)
    C22s = np.sum(f * (C22 - C12 ** 2 / C11)) + inv11 * np.sum(f * C12 / C11) ** 2
    C66s = 1.0 / np.sum(f / C66)

    C = effective_stiffness(lam, mu, tol=1e-13, maxiter=30000, stol=1e-14)
    # base de Mandel: la componente de cizalla es 2*C66
    obt = np.array([C[0, 0], C[0, 1], C[1, 1], C[2, 2] / 2.0])
    ref = np.array([C11s, C12s, C22s, C66s])
    rel = np.abs(obt - ref) / np.abs(ref)
    assert rel.max() < 1e-9, (obt, ref, rel)
    rep("laminado de Backus (C11,C12,C22,C66)",
        "exacto=" + np.array2string(ref, precision=5) +
        f" err max={rel.max():.2e}")


def test_cotas_hashin_shtrikman_2d():
    """Microestructura aleatoria de dos fases: K y G dentro de las cotas HS 2D.

    En deformacion plana 2D las cotas HS para el modulo de compresion en el
    plano K y el de cizalla G son:
        K^HS(K0,G0) = <1/(K + G0)>^-1 - G0
    con (K0,G0) los de la fase mas rigida (cota superior) o mas blanda (inferior).
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(4)
    N = (128, 128)
    lam1, mu1 = lame_from_E_nu(1.0, 0.25)
    lam2, mu2 = lame_from_E_nu(20.0, 0.25)
    K1, K2 = lam1 + mu1, lam2 + mu2          # modulo de area 2D: K = lam + mu

    g = gaussian_filter(rng.standard_normal(N), 3.0, mode="wrap")
    masc = g > np.quantile(g, 0.5)
    fr = masc.mean()
    lam = np.where(masc, lam2, lam1)
    mu = np.where(masc, mu2, mu1)

    C = effective_stiffness(lam, mu, tol=1e-11, maxiter=8000, stol=1e-11)
    # invariantes isotropos desde C (proyeccion sobre la parte isotropa)
    K_eff = 0.25 * (C[0, 0] + C[1, 1] + 2 * C[0, 1])
    G_eff = 0.25 * (C[0, 0] + C[1, 1] - 2 * C[0, 1]) / 2 + C[2, 2] / 4

    def hs_K(G0):
        return 1.0 / ((1 - fr) / (K1 + G0) + fr / (K2 + G0)) - G0

    lo, hi = hs_K(mu1), hs_K(mu2)
    assert lo - 1e-9 <= K_eff <= hi + 1e-9, (lo, K_eff, hi)
    assert min(mu1, mu2) <= G_eff <= max(mu1, mu2), (mu1, G_eff, mu2)
    rep("cotas HS 2D (modulo de area)",
        f"{lo:.4f} <= {K_eff:.4f} <= {hi:.4f}   G_eff={G_eff:.4f}")


def test_isotropia_de_C_en_medio_isotropo():
    """C_eff de un medio homogeneo debe cumplir exactamente C11-C12 = 2 C66."""
    N = (24, 24)
    lam, mu = lame_from_E_nu(7.0, 0.33)
    C = effective_stiffness(np.full(N, lam), np.full(N, mu), tol=1e-13, stol=1e-14)
    res = abs((C[0, 0] - C[0, 1]) - C[2, 2]) / C[0, 0]
    assert res < 1e-12, (C, res)
    rep("identidad isotropa C11-C12=2C66", f"residuo = {res:.2e}")


def test_contraste_alto_grieta():
    """Franja casi sin rigidez (proto-grieta): convergencia y sentido fisico.

    Una grieta en campo de fase es una zona de rigidez casi nula, es decir
    contraste enorme. El solver tiene que aguantarlo, y el resultado tiene que
    cumplir el limite de laminado: C11* -> media armonica, que con una capa casi
    nula tiende a cero proporcionalmente a su rigidez.
    """
    N = (128, 128)
    lam0, mu0 = lame_from_E_nu(1.0, 0.25)
    ancho = 4
    f_g = ancho / N[0]
    filas = []
    for eta in (1e-2, 1e-4, 1e-6):
        lam = np.full(N, lam0); mu = np.full(N, mu0)
        lam[:ancho] *= eta; mu[:ancho] *= eta
        C = effective_stiffness(lam, mu, tol=1e-11, maxiter=8000, stol=1e-11)
        C11_1, C11_2 = eta * (lam0 + 2 * mu0), lam0 + 2 * mu0
        exacto = 1.0 / (f_g / C11_1 + (1 - f_g) / C11_2)
        rel = abs(C[0, 0] - exacto) / exacto
        filas.append((eta, C[0, 0], exacto, rel))
        assert rel < 1e-8, (eta, C[0, 0], exacto, rel)
    rep("franja de rigidez casi nula (limite laminado)",
        " | ".join(f"eta={e:.0e}: C11={c:.5f} exacto={x:.5f} err={r:.1e}"
                   for e, c, x, r in filas))


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
        print(f"  {name:42s} {value}")
    print(f"\n{len(tests)-failed}/{len(tests)} tests OK")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
