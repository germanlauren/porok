# Detecta un python utilizable y fija los hilos. Se SOURCEA, no se ejecuta.
#
#   source scripts/entorno.sh
#
# Deja definidas:
#   PY              interprete a usar
#   FFTGK_WORKERS   hilos de FFT por proceso
#   ENTORNO_OK      1 si numpy, scipy y h5py estan; 0 si falta alguno
#
# Por que asi. En un cluster ajeno uno no sabe de antemano si hay conda, si hay
# modulos, o si el python del sistema ya sirve. Adivinar mal cuesta un trabajo
# encolado que muere en el primer segundo y una hora de cola perdida, asi que
# se PRUEBA en orden y se usa el primero que funcione.

_tiene_todo() {
    "$1" -c 'import numpy, scipy, h5py' >/dev/null 2>&1
}

_tiene_minimo() {
    "$1" -c 'import numpy, scipy' >/dev/null 2>&1
}

PY=""
ENTORNO_OK=0

# 1. el python que ya esta activo
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && _tiene_todo "$cand"; then
        PY=$(command -v "$cand"); ENTORNO_OK=1; break
    fi
done

# 2. conda, con el entorno del proyecto
if [[ -z "$PY" ]]; then
    for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" \
                "/opt/conda" "${CONDA_PREFIX%/envs/*}"; do
        [[ -f "$base/etc/profile.d/conda.sh" ]] || continue
        # shellcheck disable=SC1091
        source "$base/etc/profile.d/conda.sh"
        for env in fenicsx-env porok base; do
            conda activate "$env" >/dev/null 2>&1 || continue
            if _tiene_todo python3; then
                PY=$(command -v python3); ENTORNO_OK=1
                echo "[entorno] conda $env en $base"
                break 2
            fi
        done
    done
fi

# 3. modulos, que es lo habitual en clusters universitarios
if [[ -z "$PY" ]] && command -v module >/dev/null 2>&1; then
    for m in python scipy-stack anaconda miniforge; do
        module load "$m" >/dev/null 2>&1 || continue
        if _tiene_todo python3; then
            PY=$(command -v python3); ENTORNO_OK=1
            echo "[entorno] module load $m"
            break
        fi
    done
fi

# 4. ultimo recurso: algo que al menos tenga numpy y scipy. La salida cae a
#    .npz, que corre igual, pero despues no se puede analizar con leer_*.
if [[ -z "$PY" ]]; then
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && _tiene_minimo "$cand"; then
            PY=$(command -v "$cand")
            echo "[entorno] AVISO: falta h5py, la salida caera a .npz"
            break
        fi
    done
fi

# Un hilo de BLAS por proceso. Si OpenBLAS y scipy.fft se pelean por los
# mismos nucleos, el conjunto rinde MENOS que con un solo hilo.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Hilos de FFT: los que asigne el planificador, o los del nodo si no hay.
if [[ -z "${FFTGK_WORKERS:-}" ]]; then
    if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
        export FFTGK_WORKERS="$SLURM_CPUS_PER_TASK"
    elif [[ -n "${NCPUS:-}" ]]; then
        export FFTGK_WORKERS="$NCPUS"
    else
        export FFTGK_WORKERS=6
    fi
fi

export PY ENTORNO_OK
