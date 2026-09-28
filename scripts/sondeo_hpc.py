"""
Sondeo del HPC: que tamano aguanta, cuanto cuesta, y si conviene FFT multihilo.

Correr ANTES de lanzar el barrido. Responde tres preguntas con numeros, no con
suposiciones:

  1. Cuantos nucleos y cuanta memoria hay realmente disponibles.
  2. Si scipy.fft con varios hilos es mas rapido que numpy.fft (de un solo
     hilo). En una maquina con muchos nucleos esto puede ser el factor mas
     grande de todos, y el codigo usa numpy.fft por defecto.
  3. Cuanto cuesta un paso del campo de fase y una homogeneizacion a cada
     tamano, en tiempo y en memoria, para poder dimensionar el barrido.

Uso:   python3 scripts/sondeo_hpc.py
       python3 scripts/sondeo_hpc.py --max-3d 192     (si hay memoria de sobra)
"""
import sys
import os
import time
import argparse
import pathlib
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[0] / "src"))


def seccion(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72, flush=True)


def gb(x):
    return x / 1024 ** 3


def memoria_disponible():
    try:
        with open("/proc/meminfo") as f:
            d = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in f}
        return d.get("MemAvailable", d.get("MemFree", 0)), d.get("MemTotal", 0)
    except Exception:
        return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-3d", type=int, default=128)
    a = ap.parse_args()

    seccion("1. Hardware y entorno")
    disp, total = memoria_disponible()
    print(f"  nucleos logicos (os.cpu_count) : {os.cpu_count()}")
    try:
        print(f"  nucleos utilizables (afinidad) : {len(os.sched_getaffinity(0))}")
    except Exception:
        pass
    for v in ("SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE", "SLURM_MEM_PER_NODE",
              "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(v):
            print(f"  {v:26s} : {os.environ[v]}")
    print(f"  memoria total / disponible     : {gb(total):.1f} / {gb(disp):.1f} GB")
    print(f"  python {sys.version.split()[0]}   numpy {np.__version__}")
    try:
        import scipy
        print(f"  scipy {scipy.__version__}")
    except ImportError:
        print("  scipy NO disponible -- hace falta")
    try:
        import h5py
        print(f"  h5py {h5py.__version__}")
    except ImportError:
        print("  h5py no disponible: la salida caera a .npz (funciona igual)")
    cfg = getattr(np, "show_config", None)
    if cfg:
        try:
            info = np.show_config(mode="dicts")
            bl = info.get("Build Dependencies", {}).get("blas", {})
            print(f"  BLAS: {bl.get('name','?')} {bl.get('version','')}")
        except Exception:
            pass

    seccion("2. FFT: numpy (1 hilo) contra scipy multihilo")
    print("  El solver usa numpy.fft. Si scipy con N hilos gana por un factor")
    print("  apreciable, conviene cambiarlo: es la operacion dominante.\n")
    try:
        import scipy.fft as sfft
    except ImportError:
        sfft = None
    nh = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"  {'forma':>16} {'numpy [ms]':>12} {'scipy 1h':>10} "
          f"{'scipy Nh':>10} {'ganancia':>9}")
    for forma in [(512, 512), (1024, 1024), (64, 64, 64), (128, 128, 128)]:
        if len(forma) == 3 and forma[0] > a.max_3d:
            continue
        x = np.random.default_rng(0).standard_normal(forma)
        t = time.time()
        for _ in range(3):
            np.fft.fftn(x)
        tn = (time.time() - t) / 3 * 1e3
        t1 = tN = float("nan")
        if sfft is not None:
            t = time.time()
            for _ in range(3):
                sfft.fftn(x, workers=1)
            t1 = (time.time() - t) / 3 * 1e3
            t = time.time()
            for _ in range(3):
                sfft.fftn(x, workers=-1)
            tN = (time.time() - t) / 3 * 1e3
        g = tn / tN if tN == tN and tN > 0 else float("nan")
        print(f"  {str(forma):>16} {tn:12.1f} {t1:10.1f} {tN:10.1f} {g:8.2f}x")
    print(f"\n  (scipy Nh usa hasta {nh} hilos)")

    seccion("3. Costo de un paso, por tamano")
    from fftgk.phasefield import PhaseFieldSolver
    from fftgk.elasticity import lame_from_E_nu
    from fftgk.homogenize import homogenize
    from scipy.ndimage import gaussian_filter

    lam, mu = lame_from_E_nu(1.0, 0.2)
    print(f"  {'malla':>14} {'campos [GB]':>12} {'paso PF [s]':>12} "
          f"{'homog [s]':>11} {'40 incr [h]':>12}")
    casos = [(256, 2), (512, 2)]
    casos += [(n, 3) for n in (64, 96, 128, 160, 192, 256) if n <= a.max_3d]
    for n, d in casos:
        N = (n,) * d
        # memoria: el solver guarda phi, H (escalares) y eps (d*d), mas el
        # arranque en caliente et (d*d) y temporales del CG (~4 campos vectoriales)
        campos_float64 = (2 + 2 * d * d + 6 * d) * np.prod(N) * 8
        if campos_float64 > 0.35 * (disp or 1e18):
            print(f"  {str(N):>14} {gb(campos_float64):12.2f}   "
                  f"(se omite: pasaria del 35% de la memoria disponible)")
            continue
        rng = np.random.default_rng(0)
        Gc = 2e-3 * np.exp(0.2 * gaussian_filter(rng.standard_normal(N), 4.0,
                                                 mode="wrap"))
        ell = 4.0 / n
        S = PhaseFieldSolver(N, lam, mu, Gc, ell, L=(1.0,) * d)
        E = np.zeros((d, d))
        E[-1, -1] = 0.04
        t = time.time()
        S.step(E, tol_stag=1e-5, max_stag=12)
        tpf = time.time() - t
        k = 1.0 + 1e4 * np.clip(S.phi, 0, 1) ** 2
        t = time.time()
        homogenize(k, L=(1.0,) * d, tol=1e-10, maxiter=1500, dual=False,
                   ktol=1e-9)
        th = time.time() - t
        # un incremento de longitud de grieta ~ 3x un paso (bisección + lazo)
        horas = 40 * (3 * tpf + th) / 3600
        print(f"  {str(N):>14} {gb(campos_float64):12.2f} {tpf:12.2f} "
              f"{th:11.2f} {horas:12.2f}")
        del S

    seccion("4. Lectura")
    print("""  - Si la ganancia de scipy multihilo es > 1.5x, vale la pena cambiar el
    backend de FFT (una linea en homogenize.py y damage.py).
  - '40 incr [h]' es el costo estimado de UNA trayectoria completa. Multiplicar
    por el numero de trayectorias y de realizaciones de microestructura.
  - El costo cerca del punto limite es varias veces el del paso tipico
    (doc 09), asi que la estimacion es un piso, no una media.
  - En 3D la memoria la domina el campo de deformacion (d*d componentes).
    Si el tamano que interesa no cabe, la salida es paralelizar por dominio,
    no subir la malla.""")


if __name__ == "__main__":
    main()
