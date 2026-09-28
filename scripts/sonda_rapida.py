"""
Sonda rapida, SIN dependencias del proyecto. Solo numpy (scipy si esta).

Pensada para pegarse en una terminal remota y correr en menos de un minuto.
Responde lo que un benchmark tradicional no responde:

  * cuantos nucleos puede usar ESTE proceso (afinidad), que puede ser mucho
    menos que los del nodo si hay SLURM o cgroups de por medio
  * que backend de FFT hay y cuanto gana con varios hilos -- la FFT es la
    operacion dominante del solver, asi que ese factor se multiplica por todo
  * cuanta memoria pide cada malla, y cual es la mas grande que cabe

La version completa (scripts/sondeo_hpc.py) ademas cronometra un paso real del
campo de fase, pero necesita el paquete.
"""
import os
import sys
import time
import numpy as np

print("=" * 66)
print("1. NUCLEOS Y MEMORIA")
print("=" * 66)
print(f"  nucleos del nodo (cpu_count)     : {os.cpu_count()}")
try:
    afin = len(os.sched_getaffinity(0))
    print(f"  nucleos de ESTE proceso          : {afin}")
    if afin != os.cpu_count():
        print("     ^ menos que los del nodo: hay cgroup o SLURM limitando")
except AttributeError:
    afin = os.cpu_count()
for v in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE",
          "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    if os.environ.get(v):
        print(f"  {v:32s} : {os.environ[v]}")
try:
    with open("/proc/meminfo") as f:
        mi = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in f}
    tot, dis = mi["MemTotal"] / 2**30, mi.get("MemAvailable", 0) / 2**30
    print(f"  memoria total / disponible       : {tot:.1f} / {dis:.1f} GB")
except Exception:
    tot = dis = 0.0
print(f"  python {sys.version.split()[0]}  numpy {np.__version__}")
try:
    import scipy
    import scipy.fft as sfft
    print(f"  scipy {scipy.__version__}  -> FFT multihilo disponible")
except ImportError:
    sfft = None
    print("  scipy NO esta -> hace falta instalarlo")
try:
    import h5py
    print(f"  h5py {h5py.__version__}")
except ImportError:
    print("  h5py no esta (la salida caeria a .npz, funciona igual)")

print()
print("=" * 66)
print("2. FFT: LO QUE UN BENCHMARK TRADICIONAL NO MIDE")
print("=" * 66)
print("  numpy.fft es de UN SOLO HILO. En un nodo de 64 nucleos eso significa")
print("  usar 1/64 de la maquina sin que nproc lo delate. scipy.fft si escala.")
print()
print(f"  {'forma':>18} {'numpy [ms]':>11} {'scipy 1h':>10} {'scipy Nh':>10} {'ganancia':>9}")
for forma in [(1024, 1024), (2048, 2048), (128, 128, 128), (192, 192, 192)]:
    nbytes = np.prod(forma) * 16 * 3 / 2**30      # complejo + temporales
    if dis and nbytes > 0.25 * dis:
        print(f"  {str(forma):>18}   (se omite: pediria {nbytes:.1f} GB)")
        continue
    x = np.random.default_rng(0).standard_normal(forma)
    t = time.time()
    for _ in range(3):
        np.fft.fftn(x)
    tn = (time.time() - t) / 3 * 1e3
    if sfft is None:
        print(f"  {str(forma):>18} {tn:11.1f}          -          -         -")
        continue
    t = time.time()
    for _ in range(3):
        sfft.fftn(x, workers=1)
    t1 = (time.time() - t) / 3 * 1e3
    t = time.time()
    for _ in range(3):
        sfft.fftn(x, workers=-1)
    tN = (time.time() - t) / 3 * 1e3
    print(f"  {str(forma):>18} {tn:11.1f} {t1:10.1f} {tN:10.1f} {tn/tN:8.2f}x")
    del x
print(f"\n  (scipy Nh usa hasta {afin} hilos)")

print()
print("=" * 66)
print("3. QUE MALLA CABE")
print("=" * 66)
print("  El solver guarda, por malla: phi y H (2 escalares), eps y el arranque")
print("  en caliente (2 tensores d x d) y ~6 campos vectoriales temporales del")
print("  gradiente conjugado. Todo en float64.")
print()
print(f"  {'malla':>16} {'memoria [GB]':>13} {'% disponible':>13}")
for n, d in [(512, 2), (1024, 2), (2048, 2),
             (96, 3), (128, 3), (160, 3), (192, 3), (256, 3), (320, 3)]:
    celdas = n ** d
    gbm = (2 + 2 * d * d + 6 * d) * celdas * 8 / 2**30
    pct = 100 * gbm / dis if dis else float("nan")
    marca = "  <- holgado" if pct < 25 else ("  <- justo" if pct < 60 else "  <- NO cabe")
    print(f"  {f'{n}^{d}':>16} {gbm:13.2f} {pct:12.0f}%{marca}")

print()
print("=" * 66)
print("COMO LEER ESTO")
print("=" * 66)
print("""  - Si 'nucleos de ESTE proceso' < los del nodo, pedir mas CPUs al
    planificador o el codigo solo usara esos.
  - Si la ganancia de scipy multihilo es > 1.5x, el backend importa. El
    codigo del proyecto ya usa scipy; esto confirma cuanto rinde aqui.
  - Apuntar a mallas que queden bajo el 25% de la memoria disponible: el
    solver pide picos transitorios por encima de la cuenta de arriba.
  - Para el costo en TIEMPO de un paso real hace falta la sonda completa
    (scripts/sondeo_hpc.py), que necesita el paquete.""")
