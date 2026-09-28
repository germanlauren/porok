# Correr en el HPC

Todo el código es **numpy + scipy puros**. No hace falta FEniCSx, ni MPI, ni
compilar nada. Si el entorno tiene `h5py` la salida va a HDF5; si no, cae a
`.npz` sin romperse.

## ⚠ Antes que nada: el operador del campo de fase cambió (doc 18)

Los dos pilotos 3D se fragmentaron en miles de motas por debajo de ℓ. La causa
no era la siembra (eso era un problema real, pero distinto): el término que
suaviza φ usaba el gradiente `rotated`, que en 3D tiene **286 modos con
|D|² = 0 a 96³** — tres líneas enteras de la red de Fourier. Esos modos quedan
sin ninguna penalización espacial. En 2D hay uno solo y fue inofensivo.

Prueba directa, misma energía motriz ruidosa en 48³: `rotated` da **7067
cúmulos**, diferencias adelantadas (`fd`) da **1**.

Ahora el campo de fase usa `fd` por defecto; elasticidad y Darcy siguen con
`rotated`, donde esos modos simplemente se proyectan fuera. Consecuencias
prácticas:

- **Los archivos llevan el esquema en el nombre** (`..._s20260914_fd.h5`). Los
  checkpoints de los pilotos viejos NO se reanudan por accidente.
- **El barrido 2D publicado se hizo con `rotated`.** Se conserva intacto en
  `resultados_final.jsonl`; el nuevo va a `resultados_fd.jsonl`. Hay que
  relanzarlo para que 2D y 3D usen el mismo operador:

  ```bash
  POROK_ESQUEMA_PHI=fd python3 -u scripts/experimento_final.py 2>&1 | tee barrido2d_fd.log
  ```

  En los primeros incrementos el cambio mueve K un 0.4–0.6%, muy por debajo de
  todos los efectos reportados, pero el snap es justo donde el operador más
  importa y hay que verlo, no suponerlo.

## 0. scipy es OBLIGATORIO, no opcional

No es solo por velocidad. Cuatro partes lo necesitan de verdad:

| módulo | usa | para qué |
|---|---|---|
| `topology.py` | `ndimage.label` | cúmulos y percolación — el descriptor central |
| `produccion.py` | `ndimage.gaussian_filter` | generar la microestructura |
| `apertura.py` | `ndimage.gaussian_filter` | suavizado del campo de apertura |
| `cierres.py` | `optimize.minimize_scalar` | calibrar los modelos de la industria |
| `fft.py` | `scipy.fft` | 3–10× más rápido que numpy (medido, §0) |

Preferir conda-forge sobre pip en un entorno con FEniCSx: mezclar los dos puede
romper dolfinx.

**Medido en el nodo (32 núcleos, 31.2 GB, python 3.14.6 / numpy 2.5.3 /
scipy 1.18.1):**

| forma | numpy.fft | scipy 1 hilo | scipy 32 hilos | ganancia |
|---|---|---|---|---|
| 512² | 2.5 ms | 1.3 | 0.5 | 5.5× |
| 1024² | 11.3 ms | 5.1 | 1.1 | 9.9× |
| 64³ | 2.1 ms | 1.0 | 0.7 | 3.1× |
| 128³ | 34.8 ms | 13.6 | 3.7 | 9.4× |

Unos 2.5× vienen de la planificación (columna de un hilo) y ~3.7× del
paralelismo. **La FFT satura hacia los 8 hilos**, y eso decide la estrategia de
lanzamiento (ver §3). En las mallas chicas (64³) casi toda la ganancia es de
planificación, no de hilos: no vale la pena darles muchos núcleos.

### h5py también hace falta

La sonda lo reporta ausente. Sin él la salida cae a un `.npz` por incremento,
que **corre igual pero no se puede leer** con `leer_escalares` / `leer_campo` /
`listar`, que son las funciones con las que se analiza el barrido. En 3D, con
campos de 0.6 GB por instantánea, tampoco conviene el formato suelto.

```bash
conda install -c conda-forge scipy h5py
```

```bash
tar xzf porok.tar.gz && cd porok
python3 -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

---

## 1. Primero medir, después lanzar

```bash
python3 scripts/sondeo_hpc.py --max-3d 192
```

Reporta núcleos reales (por afinidad, no `cpu_count`), memoria, y sobre todo
**cuánto cuesta un paso a cada tamaño** en tiempo y en memoria, con una
estimación de horas por trayectoria. Con eso se dimensiona el barrido en vez de
adivinarlo.

También compara `numpy.fft` contra `scipy.fft` multihilo. El código ya usa
scipy por defecto (`src/fftgk/fft.py`); el sondeo confirma cuánto rinde en tu
nodo — aquí dio de 3.1× a 9.9× según la forma (§0).

**Control de hilos.** Por defecto se usan los núcleos que la afinidad del
proceso permite, que es lo que respeta la asignación de SLURM. Para fijarlo:

```bash
export FFTGK_WORKERS=16
```

Pedir 8 CPUs y lanzar 64 hilos degrada el rendimiento propio y molesta a los
demás trabajos del nodo.

## 2. Lanzar una trayectoria

```bash
python3 scripts/produccion.py --trayectoria P1_uniaxial --n 256 --dim 2 \
    --gamma-fin 0.8 --salida /scratch/$USER/corridas
```

3D, que es donde la percolación es realmente distinta de 2D:

```bash
python3 scripts/produccion.py --trayectoria P1_uniaxial --n 128 --dim 3 \
    --gamma-fin 0.8 --campos-cada 5 --salida /scratch/$USER/corridas
```

Trayectorias disponibles: `P1_uniaxial`, `P2_cizalla`, `P3_confinada`,
`P4_transversal` (las cuatro proporcionales) y las dos **no proporcionales**,
que rotan 90° la dirección de carga a mitad de camino:

- `P4_no_proporcional` — rota en 0.45 · `gamma_fin`. Excursión **subdominante**:
  el material la borra.
- `P5_excursion_grande` — rota en 0.80 · `gamma_fin`. Excursión **dominante**:
  no la borra.

Son las que sostienen el resultado de memoria del artículo (doc 14). El punto
de rotación se puede mover con `--switch-frac`. La corrida imprime
`>> rotacion de carga en Gamma=...` cuando ocurre; si ese mensaje no aparece,
la trayectoria corrió como proporcional y el resultado no sirve.

### Presupuesto medido para el barrido 3D

De la sonda en este nodo, con los 32 hilos:

| malla | campos [GB] | paso PF [s] | homog [s] | 40 incr [h] |
|---|---|---|---|---|
| 96³ | 0.25 | 10.0 | 0.95 | 0.34 |
| 128³ | 0.59 | 18.6 | 2.59 | **0.65** |
| 160³ | 1.16 | 37.4 | 5.04 | 1.30 |
| 192³ | 2.00 | 62.8 | 7.51 | 2.18 |

**Esa columna resultó ser una subestimación grave, y el piloto lo destapó.**
Supone tres pasos de campo de fase por incremento. El control por longitud de
grieta no hace tres: hace **entre 40 y 70**, porque cada evaluación de la
bisección sobre λ resuelve el campo de fase entero. En 2D a 96² eso no se
notaba; en 3D a 128³ el piloto midió **1777, 2180, 2408, 2613 y 3370 s** por
incremento — unas 35× la predicción, y creciendo.

Arreglado en `paso_longitud` (doc 16), con tres cambios que no alteran el
resultado: arranque en caliente entre evaluaciones de λ, falsa posición de
Illinois en vez de bisección pura, y un predictor de Newton que hereda la
pendiente `dΓ/dλ` del incremento anterior. Medido en 2D: **de 361 a 164 solves
y de 11.6 a 3.9 s por incremento, ~3×**, y de paso 50× más preciso.

### Presupuesto, ya con el arreglo

| malla | ℓ/L | por incremento | trayectoria (53 incr) | 15 corridas, 8 procesos |
|---|---|---|---|---|
| **96³** | 1/24 | ~230 s | ~3.4 h | **~7 h** |
| 128³ | 1/32 | ~650 s | ~9.6 h | ~19 h |

**Usar 96³ para el barrido.** Además de costar tres veces menos, deja `ℓ/L =
1/24`, que es exactamente el del barrido 2D publicado: la comparación 2D↔3D
queda a igual regularización en vez de mezclar dos. 128³ se reserva para una
sola trayectoria, como chequeo de resolución.

Memoria a 96³: 8 × 0.25 = 2.0 GB de los 28.9. Sobra.

Primero el piloto, para confirmar el costo real con el código arreglado:

```bash
FFTGK_WORKERS=8 python3 -u scripts/produccion.py \
    --trayectoria P1_uniaxial --n 96 --dim 3 --gamma-fin 0.80 \
    --campos-cada 5 --salida $HOME/porok_corridas 2>&1 | tee piloto96.log
```

### El piloto 96³ falló, y por qué importa

La primera corrida 96³ terminó completa (54 incrementos, 18.3 h) y **no sirvió**.
El diagnóstico está en el doc 17; el resumen:

| | 2D (96²) | 3D (96³), primer intento |
|---|---|---|
| `stag` mediana | 8–12 | 45–80 |
| `n_clusters` máx | 7 | 4269 |
| `R` espectral máx | 1.48 | 9.28 |
| cúmulos por debajo de ℓ | — | **99.9%** |
| cúmulo mayor | celda entera | 88 de 884736 vóxeles |
| R espectral / R dif. finitas | 2.0 (constante) | **36** |

El daño nunca localizó: quedó difuso y con ruido de escala de malla. Causa:
sembrar el **mismo número** de microfisuras en 2D y en 3D. La separación media
va como `(|Y|/n)^(1/d)`, así que con n = 40 las semillas casi se tocan en 2D
(largo/separación = 0.63) y quedan a triple distancia de su tamaño en 3D
(0.34). Cada disco quedó aislado y no hubo con qué enlazar.

Arreglado en `campo_Gc`: ahora se fija la **cobertura** (largo/separación), no
el número, y n sale de la dimensión. Da 40 en 2D —idéntico bit a bit al
barrido publicado— y **253 en 3D**. Generar el campo cuesta 5 s a 96³.

Antes de volver a lanzar las quince, una corrida corta de verificación:

```bash
FFTGK_WORKERS=8 python3 -u scripts/produccion.py \
    --trayectoria P1_uniaxial --n 96 --dim 3 --gamma-fin 0.30 \
    --campos-cada 3 --salida $HOME/porok_verif 2>&1 | tee verif96.log
```

Son ~20 incrementos, algo más de una hora. Lo que tiene que pasar, y lo que
el primer intento NO hacía:

- `cum` (número de cúmulos) debe quedarse **bajo**, decenas y no miles
- el cúmulo mayor debe crecer hasta ser una fracción apreciable de la fase
  dañada (`f_mayor` subiendo hacia 1), no 88 vóxeles sueltos
- `stag` debe parecerse al 2D, del orden de 10 y no pegado al tope

Con el diagnóstico (doc 17, script `diagnostico3d.py`) se confirma en segundos:
la columna `bajo_l` tiene que caer muy por debajo del 99.9%.

Lo que hay que mirar en ese piloto es **si percola** (columna `perc`). En 2D no
percoló ningún estado hasta `tr D = 0.51`; en 3D el umbral es mucho más bajo y
es justo el régimen donde se decide si el tensor de daño sigue siendo
suficiente. Si a `gamma_fin=0.80` no percola, subir `gamma_fin` antes de lanzar
el barrido, no la malla.

**Es reanudable.** Guarda un checkpoint tras cada incremento; relanzar el mismo
comando continúa donde iba. Pensado para colas con límite de tiempo: si el
trabajo muere a las 24 h, se vuelve a encolar y sigue.

## 3. Lanzar el barrido: muchos procesos, pocos hilos cada uno

```bash
bash scripts/lanzar_barrido.sh
NPROC=8 HILOS=4 MALLA=96 DIM=3 bash scripts/lanzar_barrido.sh
bash scripts/lanzar_barrido.sh --seguir        # ver el avance
```

**Por qué no una sola corrida con los 32 hilos.** La tabla del §0 muestra que la
FFT **satura**: de 1 a 32 hilos gana ~3.7×, no 32×. Pero un barrido es
vergonzosamente paralelo — trayectorias y semillas son independientes — así que
varios procesos de pocos hilos rinden mucho más:

| esquema | rendimiento agregado (128³) |
|---|---|
| 1 proceso × 32 hilos | 1 / 3.7 ms = 0.27 FFT/ms |
| **8 procesos × 4 hilos** | **8 / ~5 ms = 1.6 FFT/ms** — ~6× más |

Y como **una sola semilla no basta para publicar** (la celda finita tiene
fluctuación estadística propia, doc 01 §4), de todas formas hacen falta varias
corridas. El paralelismo sale gratis.

Memoria: 8 procesos × 128³ = 4.7 GB de los 28.9 disponibles. Holgado. A 192³
serían 16 GB, todavía viable pero sin margen para otra cosa.

**Siempre** con un hilo de BLAS por proceso (el script ya lo hace):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FFTGK_WORKERS=4
```

Si OpenBLAS y `scipy.fft` se pelean por los mismos núcleos, el conjunto rinde
menos que con un solo hilo.

Hay también `scripts/trabajo.slurm` por si en algún momento esto va a un
clúster con planificador.

## 3.5 Correr en un clúster con cola

Un solo comando sirve para las tres situaciones —SLURM, PBS o máquina pelada—
porque detecta cuál hay:

```bash
bash scripts/preflight.sh          # QUE HAY, antes de encolar nada
bash scripts/lanzar.sh --ensayo    # qué se enviaría, sin enviarlo
bash scripts/lanzar.sh             # una semilla, 5 corridas (~12 h)
bash scripts/lanzar.sh --completo  # tres semillas, 15 corridas
bash scripts/lanzar.sh --estado    # avance de cada corrida
```

**Correr `preflight.sh` primero, siempre.** Reporta planificador y particiones,
núcleos y memoria, versiones de numpy/scipy/h5py, dónde hay disco, y —lo que
más importa— **ejecuta una corrida de prueba 16³ completa**, incluida una
trayectoria no proporcional. Un trabajo que muere en el primer segundo por un
`import` que falta puede haber esperado seis horas en cola, y el error no se ve
hasta abrir el `.err`.

### Qué se pide, y por qué no 32 núcleos

`--cpus-per-task=6`, no 32. La FFT **satura** hacia los 8 hilos: de 1 a 32 gana
~3.7×, no 32× (§0). Pedir el nodo entero para una tarea desperdicia cinco
sextos y no la acelera. Cinco tareas de 6 hilos usan el nodo de verdad.

`--mem=8G`. Una corrida 96³ usa ~0.25 GB de campos más temporales; 8 GB sobra y
evita que el planificador la mate por un pico.

`--array=0-4` son las cinco trayectorias con una semilla. **Conviene correr una
semilla primero** (~12 h) y agregar las otras dos cuando el resultado esté
confirmado: si algo está mal se pierden doce horas y no treinta.

### El walltime no tiene que alcanzar

Una trayectoria a 96³ lleva del orden de 11 a 15 h según cuánto cueste el
entorno del snap, y eso no se sabe de antemano. En vez de pedir de más —que
alarga la espera en cola— el trabajo **se reencola solo**:

- **SLURM**: `--signal=B:USR1@900` avisa quince minutos antes del corte; el
  manejador llama a `scontrol requeue` y la corrida arranca desde su último
  checkpoint. Con `--open-mode=append` el log no se pisa.
- **PBS**: al terminar el tramo, `scripts/falta.py` mira el último Γ guardado y
  hace `qsub` de otro tramo si falta recorrido.

`falta.py` además detecta una corrida **atascada** (dos Γ iguales seguidos) y
en ese caso **no** reencola: insistir sobre algo que no avanza consume la cuota
sin producir nada.

Consecuencia práctica: si la cola tiene límite de 4 h, poner 4 y el trabajo da
las vueltas que haga falta.

### Lo único que hay que editar a mano

En `scripts/trabajo.slurm` (o `.pbs`), las tres líneas comentadas con `###`:
partición, cuenta y tiempo. El resto se adapta solo — `scripts/entorno.sh`
prueba el python activo, después conda (`fenicsx-env`, `porok`, `base`),
después `module load`, y se queda con el primero que importe numpy, scipy y
h5py.

## 4. Los archivos .h5

Un archivo por trayectoria. La estructura separa escalares de campos a
propósito: uno analiza los escalares cientos de veces y los campos dos, así que
leer la serie completa es instantáneo aunque el archivo pese gigas.

```
/meta                    atributos: malla, l, contraste, semilla
/escalares/Gamma         (n,)        densidad de grieta
/escalares/lam           (n,)        factor de carga  <- el snap-back se ve aquí
/escalares/D             (n,d,d)     tensor de daño
/escalares/K             (n,d,d)     permeabilidad efectiva
/escalares/R             (n,)        razón de localización (>1 = grieta formada)
/escalares/n_clusters    (n,)
/escalares/f_mayor       (n,)        fracción del cúmulo mayor
/escalares/percola       (n,d)
/campos/0000/phi         float32 comprimido
/campos/0000/H
/campos/0000/eps
```

```python
from fftgk.almacen import leer_escalares, leer_campo, listar
print(listar("corridas/P1_uniaxial_n256_d2_s20260914.h5"))
e = leer_escalares(".../P1....h5")        # toda la serie, instantáneo
phi = leer_campo(".../P1....h5", 20)      # un campo concreto
```

Los campos van en float32 comprimido: el análisis no necesita doble precisión,
la corrida sí la usa internamente. Reduce ~6× frente a float64 sin comprimir.

## 5. Qué vigilar en la salida

| señal | qué significa |
|---|---|
| `lam` deja de crecer y **baja** | se cruzó el punto límite; el control por longitud de grieta está funcionando (doc 10) |
| `stag` cerca del tope | el lazo escalonado no converge; subir `--max-stag` y comprobar que `lam` no cambia |
| `stop = punto-limite` | ni el control por longitud atraviesa: sería un punto límite de segundo orden, avisar |
| `R` cruza 1.0 | el daño pasó de difuso a grieta formada. **Antes de eso `D_ij` no es densidad de grieta** (doc 04 §5) |
| `percola` pasa a 1 | la región interesante: ahí las diferencias pasan de porcentajes a factores (doc 05) |

## 6. Costo esperado

De la sonda en 2 núcleos, escalando: el costo lo domina la vecindad de las
transiciones, no el paso típico. Tres observaciones independientes apuntan a lo
mismo (docs 05, 08, 09), así que **presupuestar por el peor caso**, no por la
media: el paso crítico llegó a costar 90× el típico antes de las mejoras, y
sigue costando varias veces más.

## 7. Julia y PicoGK

Ninguno hace falta para correr esto. Dónde sí aportarían:

- **PicoGK** para generar microestructuras donde porosidad, anisotropía y
  conectividad estén **desacopladas**, cosa que una microestructura aleatoria no
  permite — y la conectividad es justo la variable oculta del proyecto
  (doc 04, anexo). Exportar vóxeles a `.npy` y cargarlos como campo `Gc`.
- **Julia** si en algún momento el cuello pasa a ser el lazo escalonado: ahí un
  solver monolítico con diferenciación automática sería más rápido que el
  escalonado. Hoy el cuello son las FFT, así que no rendiría.
