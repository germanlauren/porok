"""
Fractura por campo de fase AT2, resuelta por FFT sobre la misma grilla que la
elasticidad, la homogeneizacion de permeabilidad y el tensor de dano.

Energia
-------
    E[u, phi] = INT [ g(phi) psi+(eps) + psi-(eps) ] dV
              + Gc INT [ phi^2/(2 l) + (l/2)|grad phi|^2 ] dV

    g(phi) = (1 - phi)^2 + eta_res        degradacion
    eta_res                                rigidez residual, evita singularidad

Separacion traccion-compresion (Amor, volumetrica-desviatorica)
---------------------------------------------------------------
    psi+ = (Kd/2) <tr eps>_+^2  +  mu  eps_dev : eps_dev
    psi- = (Kd/2) <tr eps>_-^2

con Kd = lambda + 2 mu / d el modulo volumetrico en d dimensiones (en 2D
deformacion plana, Kd = lambda + mu).

Es IMPRESCINDIBLE para este proyecto: sin separacion, las grietas tambien
nuclean bajo compresion, y todo el estudio trata justamente sobre trayectorias
de carga distintas -- muchas de ellas con componentes compresivas.

La ley resultante es no lineal por el corchete de Macaulay. Se linealiza
congelando el SIGNO de tr(eps) del iterado anterior, con lo que el material
queda isotropo local con campos efectivos:

    mu_eff = g mu                     siempre
    K_eff  = g Kd   si tr eps >= 0
           = Kd     si tr eps <  0    (la compresion volumetrica no se degrada)
    lambda_eff = K_eff - 2 mu_eff / d

y se itera hasta que el campo de signos deja de cambiar. Asi se reutiliza tal
cual el solver de elasticidad ya verificado.

Irreversibilidad
----------------
Campo de historia de Miehe:  H(x,t) = max_{s<=t} psi+(eps(x,s)).
Garantiza que el dano no se cure al descargar, sin restricciones desigualdad.

Ecuacion del campo de fase
--------------------------
    dE/dphi = 0   =>   (Gc/l + 2H) phi - Gc l Laplaciano(phi) = 2H

Eliptica lineal con coeficiente variable: el laplaciano es diagonal en Fourier
(simbolo -|D|^2, el MISMO D del resto del codigo) y el termino 2H phi es
diagonal en espacio real. Se resuelve con gradiente conjugado. Como H >= 0, el
principio del maximo da 0 <= phi <= 1 sin imponerlo.

Esquema escalonado
------------------
Por paso de carga: (elasticidad <-> signo de tr eps) hasta convergencia, luego
actualizar H, luego resolver phi, y repetir hasta que phi se estabilice.
"""

from __future__ import annotations

import numpy as np

from .fft import fftn as _fftn, ifftn as _ifftn

from .homogenize import FourierProjector, _cg
from .elasticity import ElasticProjector, apply_C

__all__ = ["PhaseFieldSolver", "psi_split", "Historia"]


def psi_split(lam, mu, eps, d=None):
    """Densidades de energia psi+ y psi- segun la separacion de Amor.

    Devuelve (psi_mas, psi_menos, tr).
    """
    d = d if d is not None else eps.shape[0]
    tr = np.einsum("ii...->...", eps)
    Kd = lam + 2.0 * mu / d
    eye = np.eye(d).reshape((d, d) + (1,) * (eps.ndim - 2))
    dev = eps - (tr / d)[None, None, ...] * eye
    dev2 = np.einsum("ij...,ij...->...", dev, dev)
    trp = np.maximum(tr, 0.0)
    trm = np.minimum(tr, 0.0)
    psi_p = 0.5 * Kd * trp ** 2 + mu * dev2
    psi_m = 0.5 * Kd * trm ** 2
    return psi_p, psi_m, tr


class PhaseFieldSolver:
    """Fractura por campo de fase AT2 en celda periodica, por FFT.

    Parameters
    ----------
    N : tuple
        Malla.
    lam0, mu0 : ndarray (*N) | float
        Constantes de Lame del material INTACTO. Pueden ser campos, para
        representar heterogeneidad de la microestructura porosa.
    Gc : ndarray (*N) | float
        Tenacidad. Campo, para permitir zonas debiles.
    ell : float
        Longitud de regularizacion, en unidades de L.
    L : tuple | None
        Tamano fisico de la celda. Por defecto N (voxel unitario).
    eta_res : float
        Rigidez residual. 1e-6 es lo tipico; el solver de elasticidad esta
        verificado hasta ahi (ver doc 06).
    """

    def __init__(self, N, lam0, mu0, Gc, ell, L=None, eta_res=1e-6,
                 scheme="rotated", esquema_phi="fd"):
        self.N = tuple(N)
        self.d = len(self.N)
        self.L = tuple(float(x) for x in (L if L is not None else self.N))
        self.ell = float(ell)
        self.eta_res = float(eta_res)

        ones = np.ones(self.N)
        self.lam0 = np.asarray(lam0, float) * ones
        self.mu0 = np.asarray(mu0, float) * ones
        self.Gc = np.asarray(Gc, float) * ones

        self.base = FourierProjector(self.N, self.L, scheme)
        self.P = ElasticProjector(self.N, self.L, scheme, base=self.base)

        # OPERADOR DEL CAMPO DE FASE: diferencias adelantadas, NO "rotated".
        #
        # El esquema rotated de Willot multiplica cada derivada por
        # (1 + exp(i xi_m h))/2 en las otras direcciones, y ese factor se anula
        # en Nyquist. Resultado: modos con |D|^2 = 0 aunque k != 0. Para la
        # elasticidad y para Darcy eso es inocuo -- esos modos simplemente se
        # proyectan fuera -- pero aqui el gradiente es el UNICO termino que
        # suaviza phi, y un modo sin gradiente queda sin ninguna penalizacion
        # espacial: solo lo frenan los terminos locales (Gc/l + 2H) phi, que
        # no suavizan nada.
        #
        # En 2D hay UN modo asi, (pi, pi), y fue inofensivo. En 3D hay 286 a
        # 96^3: tres lineas enteras de la red de Fourier. El piloto 3D se
        # fragmento en miles de motas por debajo de l exactamente por ahi
        # (doc 18). Las diferencias adelantadas tienen |D_j|^2 = 4/h^2 sin^2
        # (xi_j h/2), que solo se anula en k = 0: ningun modo espurio.
        self.esquema_phi = esquema_phi
        self.base_phi = (self.base if esquema_phi == scheme
                         else FourierProjector(self.N, self.L, esquema_phi))
        # simbolo del laplaciano del campo de fase: -|D|^2
        self._D2 = np.einsum("i...,i...->...", np.conj(self.base_phi.D),
                             self.base_phi.D).real

        self.phi = np.zeros(self.N)
        self.H = np.zeros(self.N)
        self.eps = np.zeros((self.d, self.d) + self.N)
        self._tr_pos = np.ones(self.N, dtype=bool)   # signo congelado
        self._et = None                              # arranque en caliente del CG

    # ------------------------------------------------------------------ #
    def g(self, phi=None):
        phi = self.phi if phi is None else phi
        return (1.0 - phi) ** 2 + self.eta_res

    def _moduli_efectivos(self):
        """(lambda_eff, mu_eff) con el signo de tr(eps) congelado."""
        d = self.d
        g = self.g()
        Kd = self.lam0 + 2.0 * self.mu0 / d
        mu_eff = g * self.mu0
        K_eff = np.where(self._tr_pos, g * Kd, Kd)
        lam_eff = K_eff - 2.0 * mu_eff / d
        return lam_eff, mu_eff

    # ------------------------------------------------------------------ #
    def _resolver_elasticidad(self, E_macro, tol=1e-10, maxiter=4000,
                              max_signo=8, qoi_cg=1e-9):
        """Elasticidad con el corchete de Macaulay tratado por iteracion de signo.

        ARRANQUE EN CALIENTE. Entre iteraciones escalonadas -- y entre pasos de
        carga -- la fluctuacion de deformacion cambia poco, asi que se reutiliza
        como punto de partida del CG. Sin esto, un paso de carga puede llegar a
        encadenar cientos de solves desde cero y el experimento se vuelve
        inviable: medido, es la diferencia entre minutos y segundos por paso.
        """
        d = self.d
        E = np.zeros((d, d) + self.N)
        E[:, :] = np.asarray(E_macro, float).reshape((d, d) + (1,) * d)
        if self._et is None:
            self._et = np.zeros((d, d) + self.N)
        it_signo = 0
        info = {"iters": 0, "converged": True, "stop": "sin-iterar"}
        for it_signo in range(1, max_signo + 1):
            lam_e, mu_e = self._moduli_efectivos()
            load = apply_C(lam_e, mu_e, E)
            b = -self.P.project(load)
            escala = np.sqrt(float(np.vdot(load, load).real))
            # Criterio por cantidad de interes tambien aqui: dentro de un lazo
            # escalonado no tiene sentido resolver la elasticidad a 1e-10, y el
            # residuo converge mucho mas lento que la tension media, que es lo
            # que alimenta psi+ y por tanto el campo de fase.
            def _qoi(e):
                s = apply_C(lam_e, mu_e, E + e)
                return s.reshape(d * d, -1).mean(axis=1)

            et, info = _cg(lambda e: self.P.project(apply_C(lam_e, mu_e, e)),
                           b, x0=self._et, tol=tol, maxiter=maxiter,
                           scale=escala, qoi=_qoi, qoi_tol=qoi_cg)
            self._et = et
            eps = E + et
            nuevo = np.einsum("ii...->...", eps) >= 0.0
            cambio = int(np.sum(nuevo != self._tr_pos))
            self._tr_pos = nuevo
            self.eps = eps
            if cambio == 0:
                break
        return it_signo, info

    def _op_gradiente(self, p):
        """G*( Gc G p ) = -div( Gc grad p ), en forma de divergencia.

        IMPORTANTE. Con Gc variable en el espacio, la derivada variacional de
        Gc (l/2)|grad phi|^2 es  -l div(Gc grad phi),  NO  -Gc l Laplaciano(phi).
        Las dos coinciden solo si Gc es uniforme.

        La diferencia no es cosmetica: escrito como Gc l Lap(phi) el operador NO
        ES SIMETRICO cuando Gc varia, y el gradiente conjugado sobre un operador
        no simetrico diverge. Sintoma observado: con un campo de Gc con
        microfisuras, el primer solve de phi agotaba 4000 iteraciones y devolvia
        phi = 1 en el 65% del dominio con una energia motriz de 9e-6. Es decir,
        fractura instantanea de la nada.

        En forma de divergencia el operador es G*(Gc G .), que es simetrico por
        construccion -- <q, G*(Gc G p)> = <G q, Gc G p> -- y semidefinido
        positivo porque Gc > 0. CG vuelve a ser aplicable.
        """
        d = self.d
        axes = tuple(range(1, d + 1))
        phat = _fftn(p)
        D = self.base_phi.D
        g = np.real(_ifftn(D * phat[None, ...], axes=axes))
        gq = _fftn(self.Gc[None, ...] * g, axes=axes)
        return np.real(_ifftn(
            np.einsum("i...,i...->...", np.conj(D), gq)))

    def _resolver_phi(self, tol=1e-10, maxiter=4000, x0=None):
        """(Gc/l) phi + 2H phi - l div(Gc grad phi) = 2H, por CG.

        `x0` es el arranque en caliente. Por defecto se parte del phi actual,
        que es lo correcto en un paso de carga; dentro de la busqueda de lam
        conviene pasar el phi de la evaluacion ANTERIOR, que esta mucho mas
        cerca (ver paso_longitud).
        """
        l = self.ell
        diag = self.Gc / l + 2.0 * self.H

        def A(p):
            return diag * p + l * self._op_gradiente(p)

        b = 2.0 * self.H
        escala = np.sqrt(float(np.vdot(b, b).real))
        if escala == 0.0:
            return np.zeros(self.N), {"iters": 0, "converged": True}
        arranque = self.phi.copy() if x0 is None else x0.copy()
        p, info = _cg(A, b, x0=arranque, tol=tol, maxiter=maxiter,
                      scale=escala)
        return np.clip(p, 0.0, 1.0), info

    # ------------------------------------------------------------------ #
    def step(self, E_macro, tol_stag=1e-4, max_stag=50, tol=1e-10,
             congelar_H=False, qoi_tol=1e-6, qoi_cg=1e-9):
        """Un paso de carga a deformacion macroscopica E_macro.

        congelar_H=True impide actualizar el campo de historia: sirve para
        medir la respuesta elastica a dano fijo (por ejemplo al descargar).

        DOS CRITERIOS DE PARADA. El clasico es el cambio PUNTUAL maximo de phi.
        Pasado el snap fragil ese criterio se estanca: en la grieta abierta
        cualquier cambio minusculo de phi mueve mucho la deformacion local y la
        realimentacion no deja bajar el maximo puntual, asi que el lazo quema
        todas las iteraciones sin que el resultado cambie. Medido: el paso del
        snap y los siguientes costaban ~90 s frente a ~1 s de los demas.

        Pero lo que este proyecto MIDE son medias volumetricas -- densidad de
        grieta, D_ij, K -- y esas convergen mucho antes que el maximo puntual.
        Asi que se corta tambien cuando la densidad de grieta se estabiliza en
        dos chequeos consecutivos. Es el mismo argumento que llevo a cortar el
        CG por la cantidad de interes en vez de por el residuo (homogenize.py).

        El diagnostico devuelve `stop` para saber cual de los dos actuo: si casi
        todos los pasos cortan por "qoi", conviene revisar que no se este
        perdiendo fisica.
        """
        it = 0
        dphi = np.inf
        dens_prev = None
        estables = 0
        motivo = "maxiter"
        info_e = {}
        it_signo = 0
        hist = []          # historial de dphi, para detectar estancamiento
        for it in range(1, max_stag + 1):
            it_signo, info_e = self._resolver_elasticidad(E_macro, tol=tol,
                                                          qoi_cg=qoi_cg)
            psi_p, psi_m, _ = psi_split(self.lam0, self.mu0, self.eps, self.d)
            if not congelar_H:
                self.H = np.maximum(self.H, psi_p)
            phi_new, info_p = self._resolver_phi(tol=tol)
            dphi = float(np.abs(phi_new - self.phi).max())
            self.phi = np.maximum(self.phi, phi_new)   # irreversibilidad dura
            if dphi < tol_stag:
                motivo = "dphi"
                break
            # Deteccion temprana de estancamiento. Cuando el incremento de carga
            # es demasiado grande para la rama, el lazo oscila en vez de
            # converger y gastaria las max_stag iteraciones completas antes de
            # que `cargar_hasta` pueda partir el paso. Como cada intento fallido
            # cuesta lo mismo que uno exitoso, detectarlo a las ~12 iteraciones
            # en vez de a las 40 es un factor 3 en el tramo caro del barrido.
            hist.append(dphi)
            if it >= 12 and dphi > 0.7 * hist[-9]:
                motivo = "estancado"
                break
            # criterio por cantidad de interes: densidad de grieta = <phi^2>/l
            dens = float(np.mean(self.phi ** 2) / self.ell)
            if dens_prev is not None and dens > 0:
                if abs(dens - dens_prev) / dens < qoi_tol:
                    estables += 1
                    if estables >= 2:
                        motivo = "qoi"
                        break
                else:
                    estables = 0
            dens_prev = dens
        return {
            "stag_iters": it,
            "dphi": dphi,
            "signo_iters": it_signo,
            "stop": motivo,
            "convergido": motivo in ("dphi", "qoi"),
            "hist_dphi": hist[-3:],
        }

    # ------------------------------------------------------------------ #
    # Control por longitud de grieta
    # ------------------------------------------------------------------ #

    def longitud_grieta(self, phi=None):
        """Densidad de superficie de grieta, Gamma/|Y|.

        Se usa el estimador phi^2 y no el funcional AT2 completo por dos
        razones (doc 04 seccion 3): es mas preciso en la malla -- 0.26% frente
        a 10% del termino de gradiente con l/h = 8 -- y es mucho mas barato,
        una media en vez de dos FFT. Como aqui se evalua muchas veces dentro de
        una busqueda de raiz, lo segundo importa.
        """
        p = self.phi if phi is None else phi
        return float(np.mean(p ** 2) / self.ell)

    def paso_longitud(self, E_dir, dGamma, lam_ini=1.0, max_stag=40,
                      tol_stag=1e-5, tol=1e-10, tol_lam=1e-4,
                      max_biseccion=40, tol_busqueda=1e-7):
        """Un incremento de LONGITUD DE GRIETA, resolviendo el factor de carga.

        Por que hace falta. Bajo control de deformacion, en un punto limite
        ningun incremento de carga atraviesa la transicion: la rama de
        equilibrio gira y el esquema adaptativo se arrastra sin avanzar (doc 09
        seccion 3). Invirtiendo el control -- se impone cuanto crece la grieta y
        se resuelve la carga que lo produce -- la rama se vuelve monotona y
        recorrible, incluido el tramo donde la carga BAJA al crecer la grieta.

        La clave que lo hace barato. Se parametriza la deformacion macroscopica
        como E = lam * E_dir. Para phi fijo el problema elastico es lineal en la
        deformacion impuesta, luego eps(lam) = lam * eps_hat, y por tanto

            psi+(lam) = lam^2 * psi+(eps_hat)

        exactamente. Ademas el signo de tr(eps) no cambia al escalar por lam > 0,
        asi que la separacion de Amor tampoco. Consecuencia: se resuelve la
        ELASTICIDAD UNA SOLA VEZ por iteracion escalonada, y la busqueda de lam
        solo necesita resolver el campo de fase, que es practicamente gratis
        (0 iteraciones de CG en la practica, doc 09 seccion 5).

        Como Gamma crece de forma monotona con lam, se usa biseccion con
        expansion del intervalo: robusto, sin derivadas.

        Parameters
        ----------
        E_dir : array (d,d)
            Direccion de la deformacion macroscopica, normalizada como se
            quiera; lam absorbe la escala.
        dGamma : float
            Incremento de densidad de superficie de grieta pedido.
        lam_ini : float
            Punto de partida para lam. Usar el del paso anterior acelera.

        Returns
        -------
        dict con lam, Gamma alcanzada, iteraciones, y motivo de parada.
        """
        d = self.d
        E_dir = np.asarray(E_dir, float)
        H_ini = self.H.copy()
        phi_ini = self.phi.copy()
        objetivo = self.longitud_grieta() + dGamma

        # La busqueda de lam se resuelve DIEZ VECES mas fino que el criterio de
        # convergencia del lazo escalonado. Usar el mismo numero para las dos
        # cosas es un error sutil y caro: si la busqueda solo garantiza lam con
        # error ~tol_lam, entonces el dlam medido entre iteraciones tiene un
        # piso de ese mismo tamano y la condicion dlam < tol_lam se cumple por
        # casualidad o no se cumple nunca. Se veia como stag creciendo (5, 7, 8,
        # 9, 12...) sin que lam cambiara de verdad.
        # CIEN veces mas fino que el criterio de convergencia, no diez. El
        # resultado mas fino que reporta el articulo es un borrado de 0.018% en
        # K; con tol_biseccion = 0.1*tol_lam la deriva numerica llegaba a
        # 0.023%, es decir, se habria comido el resultado. El costo de apretar
        # es casi nulo porque Illinois converge superlinealmente.
        tol_biseccion = 0.01 * tol_lam
        # La pendiente dGamma/dlam se hereda del incremento ANTERIOR: cambia
        # poco de un incremento al siguiente, asi que el predictor ya sirve en
        # la PRIMERA iteracion escalonada, que era la mas cara (17-18 solves).
        # Es el paso de continuacion clasico.
        pendiente = getattr(self, "_pendiente_lam", None)

        lam = float(lam_ini)
        motivo = "maxiter"
        it = 0
        for it in range(1, max_stag + 1):
            # --- elasticidad UNA vez, con lam = 1 sobre la direccion dada ---
            self._resolver_elasticidad(E_dir, tol=tol)
            psi_hat, _, _ = psi_split(self.lam0, self.mu0, self.eps, d)

            # --- buscar lam tal que Gamma(lam) = objetivo -----------------
            #
            # COSTO. Cada evaluacion resuelve el campo de fase. En 2D a 96^2 eso
            # era casi gratis con arranque en caliente y la busqueda no se
            # notaba; en 3D a 128^3 NO lo es, y con ~15 evaluaciones por
            # iteracion escalonada la busqueda pasa a dominar el costo total
            # (medido: 2000 s por incremento, 35x lo que predecia la sonda).
            # Dos cambios lo devuelven a su sitio, sin tocar el resultado:
            #
            #   1. arrancar cada evaluacion del phi de la evaluacion ANTERIOR.
            #      La biseccion produce lam cada vez mas cercanos, asi que el
            #      arranque es cada vez mejor y el CG converge en pocas
            #      iteraciones. Partir siempre del mismo phi desperdicia eso.
            #   2. resolver FLOJO mientras se busca y APRETADO solo al final.
            #      Un lam de prueba que se va a descartar no necesita 1e-10;
            #      solo tiene que decir de que lado del objetivo cae.
            #
            # El phi que queda en el estado es siempre el de la resolucion
            # apretada, asi que la trayectoria no cambia (verificado contra el
            # barrido 2D: identico a 1e-12).
            caliente = [self.phi.copy()]

            def phi_de(lm, tol_local=None):
                self.H = np.maximum(H_ini, (lm ** 2) * psi_hat)
                p, _ = self._resolver_phi(tol=tol_local or tol, x0=caliente[0])
                caliente[0] = p
                return np.maximum(phi_ini, np.clip(p, 0.0, 1.0))

            def G_de(lm):
                return self.longitud_grieta(phi_de(lm, tol_busqueda))

            lo, hi = lam, lam
            g_lo = g_hi = G_de(lam)

            # SALIDA TEMPRANA. Pasada la primera iteracion escalonada, lam casi
            # no se mueve: rehacer la busqueda entera para volver al mismo
            # numero es el grueso del costo (medido: 7 a 9 solves por iteracion
            # ya convergida). Con la pendiente dGamma/dlam de la busqueda
            # anterior se traduce el error en Gamma a error en lam, y si ya
            # estamos dentro de tolerancia no se busca nada.
            if (pendiente is not None
                    and abs(g_hi - objetivo) <= pendiente * tol_biseccion * lam):
                lam_nuevo = lam
                phi_nuevo = phi_de(lam_nuevo)
                dphi = float(np.abs(phi_nuevo - self.phi).max())
                self.phi = phi_nuevo
                if dphi < tol_stag:
                    motivo = "convergido"
                    break
                continue

            # PREDICTOR. Si ya conocemos dGamma/dlam, un paso de Newton cae
            # practicamente encima de la raiz y el intervalo se abre en una
            # evaluacion en vez de en cinco. Expandir a ciegas desde el lam
            # viejo era lo que costaba 10-12 solves en las iteraciones
            # intermedias, donde phi todavia se mueve y lam con el.
            if pendiente is not None and pendiente > 0.0:
                lam_p = max(lam + (objetivo - g_lo) / pendiente, 1e-12)
                g_p = G_de(lam_p)
                if (g_lo - objetivo) * (g_p - objetivo) <= 0.0:
                    if lam_p > lam:
                        lo, hi, g_hi = lam, lam_p, g_p
                    else:
                        lo, hi, g_lo, g_hi = lam_p, lam, g_p, g_lo
                else:
                    lo = hi = lam_p           # mismo lado, pero mas cerca
                    g_lo = g_hi = g_p

            sube, baja = 1.3, 0.77
            n_exp = 0
            if g_hi < objetivo:
                while g_hi < objetivo and n_exp < 60:
                    lo, g_lo = hi, g_hi
                    hi *= sube
                    g_hi = G_de(hi)
                    n_exp += 1
            else:
                while g_lo > objetivo and n_exp < 60:
                    hi, g_hi = lo, g_lo
                    lo *= baja
                    if lo < 1e-12:
                        break
                    g_lo = G_de(lo)
                    n_exp += 1
            pendiente = abs((g_hi - g_lo) / max(hi - lo, 1e-30))
            if not (g_lo <= objetivo <= g_hi):
                # el objetivo no es alcanzable: la irreversibilidad ya lo supera
                motivo = "objetivo-inalcanzable"
                lam = hi
                self.phi = phi_de(lam)
                break

            # Falsa posicion con amortiguacion de Illinois, en vez de biseccion
            # pura. Gamma(lam) es suave y monotona, asi que interpolar entre los
            # extremos acierta mucho mejor que partir por la mitad: la biseccion
            # necesita log2(0.3/tol) ~ 12 evaluaciones para el mismo intervalo,
            # y cada evaluacion es un solve del campo de fase, que en 3D no es
            # gratis. Illinois conserva el bracket -- no puede divergir -- y
            # ademas evita el estancamiento clasico de la falsa posicion, en el
            # que un extremo se queda fijo y la convergencia se vuelve lineal.
            f_lo, f_hi = g_lo - objetivo, g_hi - objetivo
            lam_nuevo = 0.5 * (lo + hi)
            for _ in range(max_biseccion):
                if f_hi != f_lo:
                    cand = hi - f_hi * (hi - lo) / (f_hi - f_lo)
                else:
                    cand = 0.5 * (lo + hi)
                # no dejar que el candidato se pegue a un extremo
                margen = 0.05 * (hi - lo)
                lam_nuevo = min(max(cand, lo + margen), hi - margen)
                g = G_de(lam_nuevo)
                f = g - objetivo
                if f == 0.0:
                    break
                if f < 0.0:
                    lo, f_lo = lam_nuevo, f
                    f_hi *= 0.5                      # Illinois
                else:
                    hi, f_hi = lam_nuevo, f
                    f_lo *= 0.5
                if (hi - lo) <= tol_biseccion * max(hi, 1e-30):
                    break

            phi_nuevo = phi_de(lam_nuevo)
            dphi = float(np.abs(phi_nuevo - self.phi).max())
            self.phi = phi_nuevo
            dlam = abs(lam_nuevo - lam) / max(abs(lam_nuevo), 1e-30)
            lam = lam_nuevo
            if dphi < tol_stag and dlam < tol_lam:
                motivo = "convergido"
                break

        # dejar el estado consistente con el lam final
        self._pendiente_lam = pendiente
        self.H = np.maximum(H_ini, (lam ** 2) * psi_hat)
        self._resolver_elasticidad(lam * E_dir, tol=tol)
        return {
            "lam": lam,
            "Gamma": self.longitud_grieta(),
            "Gamma_objetivo": objetivo,
            "stag_iters": it,
            "dphi": dphi if it else np.inf,
            "stop": motivo,
            "E": lam * E_dir,
        }

    # ------------------------------------------------------------------ #
    def estado(self):
        """Copia del estado completo, para poder deshacer un paso fallido."""
        return {
            "phi": self.phi.copy(),
            "H": self.H.copy(),
            "eps": self.eps.copy(),
            "tr_pos": self._tr_pos.copy(),
            "et": None if self._et is None else self._et.copy(),
        }

    def restaurar(self, st):
        self.phi = st["phi"].copy()
        self.H = st["H"].copy()
        self.eps = st["eps"].copy()
        self._tr_pos = st["tr_pos"].copy()
        self._et = None if st["et"] is None else st["et"].copy()

    def cargar_hasta(self, E_ini, E_fin, n_sub_max=32, tol_stag=1e-5,
                     max_stag=40, qoi_tol=1e-6, callback=None, dt_ini=0.1, dt_min=1e-4):
        """Avanza de E_ini a E_fin con SUBPASOS ADAPTATIVOS.

        Motivacion, medida. Cerca del punto critico el lazo escalonado necesita
        cada vez mas iteraciones (15 -> 20 -> 34 al acercarse), y si el
        incremento de carga es demasiado grande deja de converger del todo: se
        queda oscilando, `dphi` no baja monotonamente, y el estado que devuelve
        depende de donde se corto el lazo. Eso seria fatal aqui, porque las
        cantidades que no convergen son justamente las topologicas -- numero de
        cumulos y razon de localizacion R -- que son el corazon de la
        afirmacion del proyecto. K, en cambio, apenas se mueve (0.6% entre 5 y
        20 iteraciones), asi que truncar "parece" inocuo y no lo es.

        Comprobado que la rama es ESTABLE (con pasos pequenos converge a
        dphi ~ 8e-6), no hace falta control por longitud de grieta: basta
        partir el incremento a la mitad cada vez que el lazo no converge, y
        deshacer el paso fallido. Si algun dia hiciera falta bajar de
        n_sub_max subdivisiones, eso SI indicaria un punto limite verdadero y
        entonces si tocaria el esquema de Aranda & Segurado.

        Devuelve lista de dicts, uno por subpaso aceptado.
        """
        E_ini = np.asarray(E_ini, float)
        E_fin = np.asarray(E_fin, float)
        hechos = []
        t = 0.0
        dt = float(dt_ini)
        n_divisiones = 0
        while t < 1.0 - 1e-12:
            dt = min(dt, 1.0 - t)
            E = E_ini + (E_fin - E_ini) * (t + dt)
            antes = self.estado()
            info = self.step(E, tol_stag=tol_stag, max_stag=max_stag,
                             qoi_tol=qoi_tol)
            if info["stop"] in ("maxiter", "estancado"):
                self.restaurar(antes)
                dt *= 0.5
                n_divisiones += 1
                if n_divisiones > n_sub_max:
                    info["stop"] = "punto-limite"
                    info["E"] = E
                    hechos.append(info)
                    break
                continue
            # Piso para el subpaso. En un PUNTO LIMITE verdadero ningun
            # incremento de carga converge: el esquema adaptativo reduce dt sin
            # fin y avanza de forma microscopica, dando la falsa impresion de
            # progreso. Con piso, se detecta y se reporta en vez de gastar horas
            # arrastrandose. Es la senal de que hace falta control por longitud
            # de grieta (Aranda & Segurado): imponer el avance de la grieta y
            # resolver la carga, en vez de al reves.
            if dt < dt_min:
                self.restaurar(antes)
                info["stop"] = "punto-limite"
                info["E"] = E
                hechos.append(info)
                break
            t += dt
            n_divisiones = 0        # el contador es por tramo, no acumulado
            info["t"] = t
            info["E"] = E
            info["divisiones"] = n_divisiones
            hechos.append(info)
            if callback is not None:
                callback(self, info)
            # El conteo de iteraciones es la senal de cercania al punto critico:
            # crece de forma sistematica al acercarse (medido 15 -> 20 -> 34).
            # Usarlo para ajustar el paso evita gastar intentos fallidos, que
            # cuestan lo mismo que un paso exitoso.
            frac = info["stag_iters"] / max_stag
            if frac > 0.6:
                dt *= 0.6
            elif frac < 0.25:
                dt *= 1.5
            dt = min(dt, 1.0 - t) if t < 1.0 else dt
        return hechos

    # ------------------------------------------------------------------ #
    def energia_elastica(self):
        psi_p, psi_m, _ = psi_split(self.lam0, self.mu0, self.eps, self.d)
        vol = float(np.prod(self.L))
        return float(np.mean(self.g() * psi_p + psi_m)) * vol

    def energia_fractura(self):
        """Gc INT [phi^2/(2l) + (l/2)|grad phi|^2].

        Se evalua con el MISMO simbolo D que todo lo demas, de modo que sea
        consistente con el estimador de densidad de grieta de damage.py.
        """
        from .damage import gradient
        g = gradient(self.phi, L=self.L, projector=self.base_phi)
        dens = (self.phi ** 2 / (2 * self.ell)
                + 0.5 * self.ell * (g ** 2).sum(axis=0))
        vol = float(np.prod(self.L))
        return float(np.mean(self.Gc * dens)) * vol

    def tension_media(self):
        lam_e, mu_e = self._moduli_efectivos()
        sig = apply_C(lam_e, mu_e, self.eps)
        return sig.reshape(self.d, self.d, -1).mean(axis=2)


# ---------------------------------------------------------------------- #
# Solucion homogenea exacta, para verificacion
# ---------------------------------------------------------------------- #

def phi_homogeneo(psi_plus, Gc, ell):
    """Solucion exacta del campo de fase cuando no hay gradientes.

    De (Gc/l + 2H) phi = 2H con Lap(phi) = 0:

        phi = 2 H / (Gc/l + 2 H)

    Es la respuesta antes de que el dano localice, y da un test exacto de la
    ecuacion del campo de fase independiente del solver elastico.
    """
    return 2.0 * psi_plus / (Gc / ell + 2.0 * psi_plus)
