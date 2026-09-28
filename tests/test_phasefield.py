"""
Verificacion del solver de campo de fase AT2.

Aqui no hay un "resultado exacto de la naturaleza" contra el cual comparar: lo
que se verifica es que el SOLVER resuelve el MODELO que dice resolver. Para eso
se usan soluciones cerradas del propio modelo AT2, que existen y son exigentes.
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fftgk.phasefield import PhaseFieldSolver, psi_split, phi_homogeneo  # noqa: E402
from fftgk.elasticity import lame_from_E_nu  # noqa: E402
from fftgk.damage import crack_density  # noqa: E402

_REPORT = []


def rep(name, value):
    _REPORT.append((name, str(value)))


def solver(N=(64, 64), E=1.0, nu=0.2, Gc=1e-3, ell_px=6.0, L=(1.0, 1.0),
           Gc_field=None):
    lam, mu = lame_from_E_nu(E, nu)
    ell = ell_px * L[0] / N[0]
    gc = Gc if Gc_field is None else Gc_field
    return PhaseFieldSolver(N, lam, mu, gc, ell, L=L)


def test_escalado_cuadratico_de_psi():
    """psi+(lam E) = lam^2 psi+(E) exactamente, a phi fijo.

    Es la propiedad estructural sobre la que descansa el control por longitud
    de grieta: permite resolver la ELASTICIDAD UNA SOLA VEZ por iteracion
    escalonada y buscar el factor de carga resolviendo solo el campo de fase,
    que es practicamente gratis. Si no fuera exacta, el metodo seria invalido.

    Se cumple porque (a) el problema elastico es lineal en la deformacion
    macroscopica impuesta a moduli fijos, y (b) el signo de tr(eps) -- y por
    tanto la separacion de Amor -- no cambia al escalar por lam > 0.
    """
    from fftgk.phasefield import psi_split
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(1)
    N = (48, 48)
    Gc = 2e-3 * np.exp(0.5 * gaussian_filter(rng.standard_normal(N), 4.0,
                                             mode="wrap"))
    S = solver(N=N, ell_px=4.0, Gc_field=Gc)
    # danar un poco para que phi no sea trivial y los moduli sean heterogeneos
    S.step(np.array([[0.0, 0.0], [0.0, 0.03]]), tol_stag=1e-8, max_stag=20)
    E_dir = np.array([[0.0, 0.2], [0.2, 1.0]])
    S._resolver_elasticidad(0.02 * E_dir, tol=1e-12)
    p1, _, _ = psi_split(S.lam0, S.mu0, S.eps, 2)
    S._resolver_elasticidad(0.02 * 2.5 * E_dir, tol=1e-12)
    p2, _, _ = psi_split(S.lam0, S.mu0, S.eps, 2)
    err = np.abs(p2 - 2.5 ** 2 * p1).max() / p2.max()
    assert err < 1e-10, err
    rep("escalado cuadratico psi+(lam E) = lam^2 psi+(E)", f"error = {err:.2e}")


def test_operador_gradiente_simetrico():
    """El termino de gradiente debe ser SIMETRICO tambien con Gc variable.

    Test de regresion de un bug real. La primera version escribio la derivada
    variacional de Gc (l/2)|grad phi|^2 como  -Gc l Laplaciano(phi)  en vez de
    -l div(Gc grad phi). Coinciden solo si Gc es uniforme; si Gc varia, el
    operador deja de ser simetrico y el gradiente conjugado diverge.

    Sintoma que tuvo: con un campo de Gc con microfisuras, el primer solve de
    phi agotaba 4000 iteraciones y devolvia phi = 1 en el 65% del dominio con
    una energia motriz de 9e-6. Fractura instantanea de la nada.

    Lo grave es que los tests de entonces NO lo detectaron: los que usaban Gc
    uniforme no podian verlo, y el de localizacion usaba un Gc por tramos con
    contraste suave donde CG todavia convergia a algo plausible. De ahi este
    test, que ataca la propiedad directamente.
    """
    from fftgk.phasefield import PhaseFieldSolver
    rng = np.random.default_rng(0)
    N = (48, 48)
    from scipy.ndimage import gaussian_filter
    Gc = 2e-3 * np.exp(0.8 * gaussian_filter(rng.standard_normal(N), 4.0,
                                             mode="wrap"))
    S = solver(N=N, ell_px=4.0, Gc_field=Gc)
    a = rng.standard_normal(N)
    b = rng.standard_normal(N)
    s1 = float(np.vdot(S._op_gradiente(a), b))
    s2 = float(np.vdot(a, S._op_gradiente(b)))
    asim = abs(s1 - s2) / abs(s1)
    assert asim < 1e-12, asim
    # semidefinido positivo
    assert float(np.vdot(a, S._op_gradiente(a))) > 0
    # y con Gc uniforme debe reducirse a Gc * |D|^2
    S2 = solver(N=N, Gc=2e-3, ell_px=4.0)
    lap = np.real(np.fft.ifftn(S2._D2 * np.fft.fftn(a)))
    red = np.abs(S2._op_gradiente(a) - 2e-3 * lap).max() / np.abs(2e-3 * lap).max()
    assert red < 1e-12, red
    rep("operador de gradiente simetrico (Gc variable)",
        f"asimetria = {asim:.2e}; reduce a Gc*Lap con Gc uniforme a {red:.2e}")


def test_separacion_amor():
    """psi+ + psi- debe reproducir la energia elastica total, y psi+ anularse
    en compresion volumetrica pura."""
    rng = np.random.default_rng(0)
    N = (16, 16)
    lam, mu = lame_from_E_nu(3.0, 0.25)
    lam_f, mu_f = np.full(N, lam), np.full(N, mu)
    eps = rng.standard_normal((2, 2) + N)
    eps = 0.5 * (eps + np.swapaxes(eps, 0, 1))
    pp, pm, tr = psi_split(lam_f, mu_f, eps, 2)
    total = 0.5 * lam * tr ** 2 + mu * np.einsum("ij...,ij...->...", eps, eps)
    err = np.abs(pp + pm - total).max() / total.max()
    assert err < 1e-12, err

    # compresion volumetrica pura: psi+ = 0
    e2 = np.zeros((2, 2) + N)
    e2[0, 0] = e2[1, 1] = -0.01
    pp2, pm2, _ = psi_split(lam_f, mu_f, e2, 2)
    assert np.abs(pp2).max() < 1e-14, pp2.max()
    assert pm2.min() > 0, pm2.min()
    rep("separacion de Amor", f"psi+ + psi- = psi a {err:.2e}; "
                              f"compresion pura da psi+ = {np.abs(pp2).max():.1e}")


def test_solucion_homogenea_exacta():
    """Celda homogenea bajo traccion volumetrica: phi debe seguir la formula
    cerrada phi = 2H/(Gc/l + 2H), sin gradientes.

    Es el test mas directo de la ecuacion del campo de fase.
    """
    S = solver(N=(48, 48), Gc=2e-3, ell_px=6.0)
    lam, mu = S.lam0[0, 0], S.mu0[0, 0]
    Kd = lam + mu                      # d = 2
    filas = []
    for e0 in (0.005, 0.01, 0.02, 0.04):
        S.phi[:] = 0.0
        S.H[:] = 0.0
        E = np.array([[e0, 0.0], [0.0, e0]])
        S.step(E, tol_stag=1e-10, max_stag=40)
        psi_p = 0.5 * Kd * (2 * e0) ** 2
        exacto = phi_homogeneo(psi_p, S.Gc[0, 0], S.ell)
        obt = float(S.phi.mean())
        disp = float(S.phi.std())
        rel = abs(obt - exacto) / exacto
        filas.append((e0, obt, exacto, rel))
        assert disp < 1e-10, ("phi debe ser uniforme", disp)
        assert rel < 1e-8, (e0, obt, exacto, rel)
    rep("solucion homogenea exacta",
        " | ".join(f"e={e:.3f}: phi={o:.5f} exacto={x:.5f} err={r:.1e}"
                   for e, o, x, r in filas))


def test_pico_de_tension_at2():
    """Curva tension-deformacion homogenea: el pico cae donde predice AT2.

    Carga biaxial eps = e*I en 2D:  tr eps = 2e,  eps_dev = 0, luego
        psi+ = (Kd/2)(2e)^2 = 2 Kd e^2
        phi  = 2 psi+ / (a + 2 psi+),   a = Gc/l
        sigma = (1-phi)^2 Kd (2e)

    Maximizando sigma sobre psi+:  a = 6 psi+, es decir  2 psi+ = a/3.
    De ahi salen TRES predicciones independientes y cerradas:

        e_pico   = sqrt( a / (12 Kd) )
        phi_pico = 1/4                        exactamente
        sigma_pico / sigma_elastica = 9/16     exactamente
    """
    S = solver(N=(32, 32), Gc=2e-3, ell_px=6.0)
    Kd = S.lam0[0, 0] + S.mu0[0, 0]
    a = S.Gc[0, 0] / S.ell
    e_pico = np.sqrt(a / (12.0 * Kd))
    es = np.linspace(0.2 * e_pico, 2.5 * e_pico, 40)
    sig, phis = [], []
    for e0 in es:
        S.phi[:] = 0.0
        S.H[:] = 0.0
        S.step(np.array([[e0, 0.0], [0.0, e0]]), tol_stag=1e-10, max_stag=40)
        sig.append(float(np.trace(S.tension_media())) / 2.0)
        phis.append(float(S.phi.mean()))
    sig = np.array(sig)
    i = int(np.argmax(sig))
    e_med = es[i]
    phi_pico = phis[i]
    # el maximo de la malla de puntos debe caer junto al predicho
    assert abs(e_med - e_pico) / e_pico < 0.05, (e_med, e_pico)
    assert abs(phi_pico - 0.25) < 0.02, phi_pico
    # y la caida respecto a la respuesta elastica debe ser el factor 9/16
    sig_el = Kd * 2 * e_med
    razon = sig[i] / sig_el
    assert abs(razon - 9 / 16) < 0.02, (razon, 9 / 16)
    rep("pico de la curva AT2",
        f"e_pico medido={e_med:.5f} predicho={e_pico:.5f}; "
        f"phi en el pico={phi_pico:.4f} (exacto 0.25); "
        f"sigma/sigma_el={razon:.4f} (exacto {9/16:.4f})")


def test_irreversibilidad():
    """Al descargar, phi no debe disminuir."""
    S = solver(N=(48, 48), Gc=2e-3, ell_px=6.0)
    S.step(np.array([[0.03, 0.0], [0.0, 0.03]]), tol_stag=1e-9, max_stag=40)
    phi_carga = S.phi.copy()
    assert phi_carga.mean() > 0.05, phi_carga.mean()
    for e0 in (0.02, 0.01, 0.0, -0.01):
        S.step(np.array([[e0, 0.0], [0.0, e0]]), tol_stag=1e-9, max_stag=40)
        caida = float((phi_carga - S.phi).max())
        assert caida < 1e-12, (e0, caida)
    rep("irreversibilidad al descargar",
        f"phi tras carga = {phi_carga.mean():.5f}; "
        f"caida maxima al descargar hasta e=-0.01: {caida:.1e}")


def test_compresion_no_dana():
    """Compresion volumetrica pura no debe generar dano (separacion de Amor).

    Sin separacion, este test falla: es el control de que la separacion esta
    realmente activa en el lazo acoplado, no solo en la formula.
    """
    S = solver(N=(48, 48), Gc=2e-3, ell_px=6.0)
    S.step(np.array([[-0.05, 0.0], [0.0, -0.05]]), tol_stag=1e-10, max_stag=40)
    assert S.phi.max() < 1e-12, S.phi.max()
    # pero cizalla pura de magnitud comparable SI debe danar
    S2 = solver(N=(48, 48), Gc=2e-3, ell_px=6.0)
    S2.step(np.array([[0.0, 0.05], [0.05, 0.0]]), tol_stag=1e-10, max_stag=40)
    assert S2.phi.mean() > 0.05, S2.phi.mean()
    rep("compresion volumetrica no dana",
        f"phi(compresion)={S.phi.max():.1e}  vs  phi(cizalla)={S2.phi.mean():.4f}")


def test_localizacion_en_zona_debil():
    """Con una franja de Gc reducido, el dano debe localizar ahi.

    Verifica que el acoplamiento elasticidad-dano y el termino de gradiente
    funcionan: sin el gradiente el dano no se organizaria en una banda.

    Dos decisiones de diseno que importan:

    * Traccion UNIAXIAL en y, no volumetrica. Bajo carga volumetrica el dano
      crece de forma estable y phi tiende a 1 solo asintoticamente: nunca se
      forma una grieta propiamente dicha. La traccion perpendicular a la franja
      si produce localizacion, porque al danarse la banda el resto descarga y
      la deformacion se concentra ahi.

    * La franja se hace mas ancha que l, para que el contraste medido no este
      dominado por el suavizado del termino de gradiente.

    Se carga de forma adaptativa hasta que la grieta esta REALMENTE formada
    (phi > 0.9), no solo insinuada.
    """
    N = (64, 64)
    j0, j1 = 29, 35                 # franja de 6 px, l = 4 px
    Gc = np.full(N, 4e-3)
    Gc[:, j0:j1] = 1.0e-3           # franja horizontal debil (constante en x)
    S = solver(N=N, Gc=2e-3, ell_px=4.0, Gc_field=Gc)

    e0, pasos = 0.004, 0
    while S.phi.max() < 0.92 and e0 < 0.40 and pasos < 60:
        S.step(np.array([[0.0, 0.0], [0.0, e0]]), tol_stag=1e-6, max_stag=30)
        e0 *= 1.18
        pasos += 1

    perfil = S.phi.mean(axis=0)     # promedio en x -> perfil en y
    j = int(np.argmax(perfil))
    assert S.phi.max() > 0.9, ("la grieta no se formo", S.phi.max(), e0)
    assert j0 - 2 <= j < j1 + 2, (j, perfil.max())
    contraste = perfil.max() / max(perfil[5], 1e-12)
    assert contraste > 3.0, (contraste, perfil.max(), perfil[5])
    # y el dano debe estar concentrado, no repartido
    frac_en_franja = S.phi[:, j0 - 3:j1 + 3].sum() / S.phi.sum()
    assert frac_en_franja > 0.5, frac_en_franja
    rep("localizacion en franja debil",
        f"grieta formada en {pasos} pasos (e={e0:.4f}), phi_max={S.phi.max():.3f}, "
        f"centro en j={j} (franja {j0}-{j1-1}), contraste={contraste:.1f}x, "
        f"{frac_en_franja*100:.0f}% del dano en la franja")


def test_balance_energetico():
    """La energia de fractura debe ser Gc por la longitud de grieta medida.

    Cierra el lazo con damage.py: la densidad de grieta que mide el estimador
    verificado en el doc 04 tiene que ser consistente con el funcional de
    energia que usa el solver. Si difieren, una de las dos esta mal.
    """
    S = solver(N=(96, 96), Gc=2e-3, ell_px=6.0)
    for e0 in np.linspace(0.005, 0.04, 10):
        S.step(np.array([[e0, 0.0], [0.0, e0]]), tol_stag=1e-7, max_stag=30)
    Ef = S.energia_fractura()
    dens = crack_density(S.phi, S.ell, L=S.L, estimator="sum")
    esperado = S.Gc[0, 0] * dens * float(np.prod(S.L))
    rel = abs(Ef - esperado) / esperado
    assert rel < 1e-10, (Ef, esperado, rel)
    rep("balance energia de fractura vs densidad de grieta",
        f"Ef={Ef:.6e}  Gc*Gamma={esperado:.6e}  rel={rel:.1e}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}", flush=True)
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}", flush=True)
    print("\n--- detalle ---")
    for name, value in _REPORT:
        print(f"  {name:44s} {value}")
    print(f"\n{len(tests)-failed}/{len(tests)} tests OK")
    return failed



def test_sin_modos_nulos_en_3d():
    """El operador del campo de fase no puede tener modos sin gradiente.

    El esquema rotated de Willot los tiene: su factor (1 + exp(i xi_m h))/2 se
    anula en Nyquist, y en 3D eso deja lineas enteras de la red de Fourier con
    |D|^2 = 0 (142 modos a 48^3, 286 a 96^3). Como el gradiente es el UNICO
    termino que suaviza phi, esos modos quedan sin penalizacion y el campo se
    fragmenta en motas por debajo de l. Los dos pilotos 3D fallaron asi (doc 18).

    Prueba: una sola resolucion de phi con energia motriz ruidosa. Con el
    operador correcto sale UN cumulo; con rotated salian miles.
    """
    from scipy import ndimage
    rng = np.random.default_rng(7)
    N = (32, 32, 32); ell = 4.0 / 32; Gc = 2e-3
    H0 = 0.45 / 0.55 * Gc / ell / 2          # lleva phi a ~0.45
    S = PhaseFieldSolver(N, 1.0, 1.0, Gc, ell, L=(1.0,) * 3)
    D2 = np.einsum("i...,i...->...", np.conj(S.base_phi.D), S.base_phi.D).real
    nulos = int((D2 < 1e-10 * D2.max()).sum()) - 1        # menos k = 0
    assert nulos == 0, ("el operador de phi tiene modos nulos", nulos)
    S.H = H0 * np.exp(0.8 * rng.standard_normal(N))
    phi, _ = S._resolver_phi(tol=1e-10)
    _, nc = ndimage.label(phi > 0.5)
    assert nc <= 3, ("phi fragmentado: el ruido paso sin suavizar", nc)
    rep("sin modos nulos en 3D",
        f"0 modos con |D|^2=0 (rotated tendria {int(_nulos_rotated(N))}); "
        f"{nc} cumulo(s) con H ruidoso")


def _nulos_rotated(N):
    from fftgk.homogenize import FourierProjector
    P = FourierProjector(N, (1.0,) * len(N), "rotated")
    D2 = np.einsum("i...,i...->...", np.conj(P.D), P.D).real
    return (D2 < 1e-10 * D2.max()).sum() - 1

if __name__ == "__main__":
    raise SystemExit(main())
