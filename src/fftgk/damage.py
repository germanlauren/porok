"""
Extraccion del tensor de dano D_ij desde el campo de fase.

De esta definicion depende la afirmacion central del proyecto, asi que se
construye desde la energia del propio campo de fase en vez de postularse.

-------------------------------------------------------------------------
1. Densidad de superficie de grieta (AT2)
-------------------------------------------------------------------------

El modelo de campo de fase AT2 regulariza la grieta con la densidad

    gamma(phi, grad phi) = phi^2/(2 l) + (l/2) |grad phi|^2

cuya integral converge (Gamma-convergencia) a la longitud de grieta en 2D
(area en 3D) cuando l -> 0. El perfil optimo en 1D es phi(s) = exp(-|s|/l), y
para ese perfil cada uno de los dos terminos integra exactamente 1/2 en la
direccion transversal: hay equiparticion, y la suma da 1 por unidad de
longitud de grieta. Es decir, la normalizacion es exacta, sin constante libre.

-------------------------------------------------------------------------
2. Generalizacion tensorial, y por que la version obvia esta sesgada
-------------------------------------------------------------------------

El termino de gradiente contiene la informacion direccional: grad phi es
perpendicular al plano de grieta. La generalizacion obvia es reemplazar
|grad phi|^2 por el producto diadico:

    D^raw_ij = (1/|Y|) integral_Y  l * d_i(phi) d_j(phi)  dV        (SESGADO)

En el continuo es exacta: tr D^raw = Gamma/|Y| por equiparticion, sin
constante ajustable. **En la malla no.** El perfil AT2 tiene un pico (derivada
discontinua) en el nucleo de la grieta, y diferenciarlo numericamente subestima
la integral. Medido, con grieta alineada con la malla:

    l/h      2       4       8      16      24
    tr/exacto  0.648   0.812   0.903   0.951   0.967

Converge a orden 1 en h/l: con l = 8 voxeles, que es lo tipico en simulaciones
de campo de fase, el error es del 10%.

Y hay algo peor que un sesgo global. **El sesgo depende de la orientacion de la
grieta respecto a la malla.** Con l/h = 8:

    grieta alineada (1,0):   0.903 x el valor exacto
    grieta a 45 grados (1,1): 1.049 x el valor exacto

Una dispersion del 16% puramente numerica. En un trabajo cuya afirmacion
central es sobre la ANISOTROPIA del tensor de dano, eso es fatal: el artefacto
de malla se leeria como anisotropia fisica.

-------------------------------------------------------------------------
3. El estimador corregido
-------------------------------------------------------------------------

La solucion sale de medir por separado los dos terminos de la densidad AT2:

    A = int phi^2/(2l) dV        B = int (l/2)|grad phi|^2 dV

En el continuo A = B = Gamma/2. En la malla, A es casi exacto (no deriva nada,
solo muestrea el perfil) y B carga todo el error:

                        l/h=4    l/h=8
    2A / exacto         0.9897   0.9974      <- fiable
    2B / exacto         0.8122   0.9033      <- sesgado
    dispersion de 2A por orientacion (l/h=8):  0.5%
    dispersion de 2B por orientacion (l/h=8):  16%

Ademas, aunque la MAGNITUD de D^raw esta sesgada, su FORMA no: el cociente
entre autovalores da 1e-3 o mejor, y el autovector principal se alinea con la
normal a 1e-5. El sesgo es un factor escalar comun que se cancela al normalizar.

De ahi el estimador que se usa por defecto:

    D_ij = Gamma_A  *  N_ij

    Gamma_A = (1/|Y|) int phi^2 / l dV                    (magnitud, fiable)
    N_ij    = int d_i(phi) d_j(phi) / int |grad phi|^2     (direccion, traza 1)

Magnitud del termino que la malla resuelve bien; direccion del termino que la
resuelve bien. Ambos errores se evitan.

Propiedades (verificadas en tests/test_damage.py):

  * traza = densidad de superficie de grieta, a 0.3% con l/h = 8
  * sin sesgo apreciable por orientacion respecto a la malla
  * grieta plana unica con normal n:  D = (Gamma/|Y|) n (x) n, rango 1
  * objetividad: D(R x) = R D R^T
  * aditividad sobre familias de grietas bien separadas

-------------------------------------------------------------------------
4. La razon de localizacion como bandera de validez
-------------------------------------------------------------------------

En el continuo A = B. En la malla la razon

    R = B / A

mide dos cosas a la vez: resolucion (R -> 1 al refinar) y, mas importante,
**si el dano se localizo o no**. Una grieta bien formada sigue el perfil optimo
y da R cercano a 1 tras corregir por resolucion. Dano difuso repartido por todo
el dominio no sigue ese perfil y R se aparta.

Esto importa porque **la nocion misma de "densidad de grieta" no esta definida
para dano difuso**. R queda como bandera automatica: cuando se aparta de 1, el
tensor de dano de ese punto del barrido no es una medida confiable y hay que
marcarlo, no promediarlo con los demas.

Esto es exactamente el **tensor de fabrica de Oda** usado en mecanica de rocas
(F_ij = (1/V) suma_k A_k n_i n_j, orientacion pesada por area) y el tensor de
densidad de grietas de Kachanov. Es decir: la definicion no es un invento
nuestro conveniente, es la traduccion al lenguaje del campo de fase de la
medida que la comunidad de yacimientos ya usa. Ese es justamente el punto que
hay que poder defender ante un revisor.

-------------------------------------------------------------------------
3. La jerarquia de momentos: por que un solo tensor no basta
-------------------------------------------------------------------------

D_ij tal como se define arriba **ignora la apertura**. La mecanica de dano usa
justamente eso (densidad de grietas), pero el flujo pesa la apertura al cubo
(ley cubica). Para que la comparacion sea honesta -- y para que nadie pueda
decir "usaron una version debil del modelo de dano" -- se define la familia

    D^(m)_ij = (1/|Y|) integral_Y  l * h(x)^m * d_i(phi) d_j(phi)  dV

    m = 0 : densidad de grieta pura (Oda / Kachanov, mecanica de dano)
    m = 1 : pesado por apertura (transmisividad tipo Darcy)
    m = 3 : pesado por apertura al cubo (ley cubica)

La pregunta central del proyecto se vuelve entonces precisa y falsable:

    ¿Es K funcion de D^(0)? ¿De (D^(0), D^(3))? ¿De ALGUNA familia finita de
    momentos?

La hipotesis es que no, porque la conectividad de la red de grietas no es
ningun momento de la distribucion de orientaciones: dos poblaciones con
momentos identicos a todo orden pueden percolar o no percolar. Formularlo como
jerarquia, y no como "el tensor de dano falla", es lo que hace el argumento
resistente a revision.
"""

from __future__ import annotations

import numpy as np

from .fft import fftn as _fftn, ifftn as _ifftn

from .homogenize import FourierProjector

__all__ = ["gradient", "damage_tensor", "crack_density", "at2_profile"]


def gradient(phi, L=None, scheme="spectral", projector=None):
    """Gradiente de un campo escalar periodico. Devuelve (d, *N).

    Usa el mismo simbolo discreto que el homogeneizador, de modo que se puede
    pedir consistencia con el esquema del solver de flujo. Por defecto
    "spectral": el campo de fase es suave por construccion en la escala l, y
    ahi la diferenciacion espectral es la mas precisa. Pero el perfil AT2 tiene
    un pico en el nucleo de la grieta, asi que conviene comparar esquemas --
    ver test_damage.test_esquemas_de_gradiente.
    """
    phi = np.asarray(phi, dtype=np.float64)
    N = phi.shape
    d = phi.ndim
    P = projector if projector is not None else FourierProjector(N, L, scheme)
    phihat = _fftn(phi)
    g = _ifftn(P.D * phihat[None, ...], axes=tuple(range(1, d + 1)))
    return np.real(g)


def damage_tensor(phi, ell, L=None, weight=None, m=0, scheme="spectral",
                  projector=None, method="normalized"):
    """Tensor de dano D^(m)_ij desde el campo de fase.

    method = "normalized"  (por defecto, RECOMENDADO)
        D^(m) = Gamma_A^(m) * N^(m),  con la magnitud tomada del termino phi^2
        de la densidad AT2 y la direccion del termino de gradiente normalizado.
        Evita el sesgo por resolucion y el sesgo por orientacion respecto a la
        malla. Ver seccion 3 del docstring del modulo.

    method = "raw"
        D^(m) = (1/|Y|) int l w^m grad(phi) (x) grad(phi).  La version obvia,
        sesgada. Se conserva solo para poder mostrar la diferencia.

    Parameters
    ----------
    phi : ndarray (*N)
        Campo de fase, 0 (intacto) a 1 (grieta).
    ell : float
        Longitud de regularizacion, en las MISMAS unidades que L.
    L : tuple | None
        Longitudes fisicas de la celda. Por defecto N (voxel = 1).
    weight : ndarray (*N) | None
        Campo de apertura h(x). Necesario solo si m != 0.
    m : int
        Momento de apertura.
          0 = densidad de grieta pura (Oda / Kachanov, mecanica de dano)
          1 = pesado por apertura (transmisividad)
          3 = pesado por apertura al cubo (ley cubica)

    Returns
    -------
    D : ndarray (d, d) simetrico semidefinido positivo.
    """
    phi = np.asarray(phi, dtype=np.float64)
    d = phi.ndim
    g = gradient(phi, L=L, scheme=scheme, projector=projector)

    if m == 0:
        w = np.ones_like(phi)
    else:
        if weight is None:
            raise ValueError("m != 0 requiere el campo de apertura `weight`")
        w = np.asarray(weight, dtype=np.float64) ** m

    # tensor direccional sin normalizar: int w * grad(phi) (x) grad(phi)
    T = np.empty((d, d))
    for i in range(d):
        for j in range(i, d):
            T[i, j] = T[j, i] = np.mean(w * g[i] * g[j])

    if method == "raw":
        return ell * T
    if method != "normalized":
        raise ValueError(f"method desconocido: {method}")

    trT = np.trace(T)
    if trT <= 0.0:
        return np.zeros((d, d))
    gamma_A = np.mean(w * phi ** 2) / ell    # = 2A, la densidad de grieta
    return gamma_A * (T / trT)


def crack_density(phi, ell, L=None, weight=None, m=0, estimator="phi2"):
    """Densidad de superficie de grieta Gamma/|Y|.

    estimator = "phi2"  : 2A = int w phi^2 / l        (fiable, por defecto)
    estimator = "grad"  : 2B = int w l |grad phi|^2   (sesgado en la malla)
    estimator = "sum"   : A + B                       (la densidad AT2 literal)
    """
    phi = np.asarray(phi, dtype=np.float64)
    w = (np.ones_like(phi) if m == 0
         else np.asarray(weight, dtype=np.float64) ** m)
    if estimator == "phi2":
        return float(np.mean(w * phi ** 2) / ell)
    g = gradient(phi, L=L)
    B2 = float(np.mean(w * ell * (g ** 2).sum(axis=0)))
    if estimator == "grad":
        return B2
    if estimator == "sum":
        return 0.5 * (float(np.mean(w * phi ** 2) / ell) + B2)
    raise ValueError(f"estimator desconocido: {estimator}")


def localization_ratio(phi, ell, L=None):
    """R = B / A, razon de equiparticion de la densidad AT2.

    En el continuo R = 1. Se aparta por dos causas:
      * resolucion insuficiente (R < 1, converge a 1 al refinar)
      * dano NO localizado (dano difuso no sigue el perfil optimo)

    Sirve como bandera de validez: si R se aleja de 1 mas alla de lo que
    explica la resolucion, la "densidad de grieta" de ese punto del barrido no
    es una medida confiable y hay que marcarlo en vez de promediarlo.
    """
    phi = np.asarray(phi, dtype=np.float64)
    A = float(np.mean(phi ** 2) / (2 * ell))
    g = gradient(phi, L=L)
    B = float(np.mean(ell / 2 * (g ** 2).sum(axis=0)))
    return B / A if A > 0 else np.nan


def at2_profile(coords, ell):
    """Perfil optimo AT2: phi(s) = exp(-|s|/l).

    `coords` es la distancia con signo al plano de grieta. Util para construir
    casos de prueba con respuesta analitica conocida.
    """
    return np.exp(-np.abs(np.asarray(coords, dtype=np.float64)) / ell)
