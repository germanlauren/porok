#!/bin/bash
# Un solo comando que funciona en cualquiera de las tres situaciones:
# SLURM, PBS, o una maquina pelada. Detecta cual es y hace lo que corresponde.
#
#   bash scripts/lanzar.sh                 # una semilla, 5 trayectorias
#   bash scripts/lanzar.sh --completo      # tres semillas, 15 corridas
#   bash scripts/lanzar.sh --estado        # ver como va
#   bash scripts/lanzar.sh --ensayo        # imprime lo que haria, sin hacerlo
#
# Variables que se pueden fijar antes: MALLA, DIM, GAMMA, SALIDA, NPROC, HILOS.

set -uo pipefail
cd "$(dirname "$0")/.."

MALLA=${MALLA:-96}
DIM=${DIM:-3}
GAMMA=${GAMMA:-0.80}
SALIDA=${SALIDA:-${SCRATCH:-$HOME}/porok_corridas}
COMPLETO=0
ENSAYO=0

for arg in "$@"; do
    case "$arg" in
        --completo) COMPLETO=1 ;;
        --ensayo)   ENSAYO=1 ;;
        --estado)   MODO=estado ;;
        *) echo "argumento desconocido: $arg"; exit 2 ;;
    esac
done

# ------------------------------------------------------------------ estado --
if [[ "${MODO:-}" == "estado" ]]; then
    echo "Salida: $SALIDA"
    if command -v squeue >/dev/null 2>&1; then
        echo; echo "En cola:"; squeue -u "$(whoami)" -n porok 2>/dev/null
    elif command -v qstat >/dev/null 2>&1; then
        echo; echo "En cola:"; qstat -u "$(whoami)" 2>/dev/null
    else
        echo; echo "Procesos:"; pgrep -af produccion.py || echo "  ninguno"
    fi
    echo; echo "Avance de cada corrida:"
    # shellcheck disable=SC1091
    source scripts/entorno.sh
    shopt -s nullglob
    for f in "$SALIDA"/*.h5; do
        [[ -e "$f" ]] || continue
        "$PY" - "$f" <<'PY'
import sys, pathlib
try:
    import h5py, numpy as np
    with h5py.File(sys.argv[1], "r") as h:
        g = h["escalares/Gamma"][...]; R = h["escalares/R"][...]
        nc = h["escalares/n_clusters"][...]; pc = h["escalares/percola"][...]
    perc = "SI" if np.any(pc[-1]) else "no"
    print(f"  {pathlib.Path(sys.argv[1]).stem:38s} "
          f"{len(g):3d} incr  Gamma={g[-1]:.4f}  R={R[-1]:6.3f}  "
          f"cum={int(nc[-1]):5d}  percola={perc}")
except Exception as e:
    print(f"  {pathlib.Path(sys.argv[1]).stem:38s} (ilegible: {e})")
PY
    done
    exit 0
fi

# -------------------------------------------------------------- lanzamiento --
if (( COMPLETO )); then
    ARRAY="0-14"; SEM_TXT="tres semillas (15 corridas)"
else
    ARRAY="0-4";  SEM_TXT="una semilla (5 corridas)"
    export SEMILLAS_POROK="20260914"
fi

echo "=================================================================="
echo " malla ${MALLA}^${DIM}   gamma_fin=$GAMMA   $SEM_TXT"
echo " salida: $SALIDA"
echo "=================================================================="

correr() { if (( ENSAYO )); then echo "  [ensayo] $*"; else "$@"; fi; }

if command -v sbatch >/dev/null 2>&1; then
    echo " planificador: SLURM"
    correr sbatch --array="$ARRAY" \
        --export="ALL,MALLA=$MALLA,DIM=$DIM,GAMMA=$GAMMA,SALIDA=$SALIDA" \
        scripts/trabajo.slurm
    echo
    echo " Seguimiento:  bash scripts/lanzar.sh --estado"
    echo " Cancelar:     scancel -u \$(whoami) -n porok"

elif command -v qsub >/dev/null 2>&1; then
    echo " planificador: PBS"
    correr qsub -J "$ARRAY" \
        -v "SEMILLAS_POROK=${SEMILLAS_POROK:-},MALLA=$MALLA,DIM=$DIM,GAMMA=$GAMMA,SALIDA=$SALIDA" \
        scripts/trabajo.pbs
    echo
    echo " Seguimiento:  bash scripts/lanzar.sh --estado"

else
    echo " sin planificador: se lanza directo"
    # Cinco trayectorias a la vez, seis hilos cada una, es lo que llena un nodo
    # de 32 sin que la FFT se estorbe a si misma (satura hacia los 8 hilos).
    #
    # Se EXPORTAN en vez de prefijar la llamada: un prefijo `VAR=x funcion` en
    # bash deja la variable pegada al shell despues de la llamada, que es una
    # de esas rarezas que muerden mas tarde.
    export NPROC=${NPROC:-5} HILOS=${HILOS:-6}
    export MALLA DIM GAMMA SALIDA
    echo "   NPROC=$NPROC HILOS=$HILOS MALLA=$MALLA DIM=$DIM GAMMA=$GAMMA"
    correr bash scripts/lanzar_barrido.sh
fi
