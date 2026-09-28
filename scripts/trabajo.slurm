#!/bin/bash
#SBATCH --job-name=porok
#SBATCH --output=logs/porok_%A_%a.out
#SBATCH --error=logs/porok_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
###SBATCH --partition=CAMBIAR
###SBATCH --account=CAMBIAR
#
# ============================================================================
# Barrido 3D del articulo. Una tarea del array = una trayectoria.
# ============================================================================
#
# QUE SE PIDE Y POR QUE
# ---------------------
# cpus-per-task=6, no 32. La FFT SATURA hacia los 8 hilos: de 1 a 32 gana ~3.7x,
# no 32x (medido, HPC.md seccion 0). Pedir 32 nucleos para una tarea desperdicia
# cinco sextos del nodo y no la acelera. Con 6 hilos por tarea y cinco tareas en
# paralelo se usa el nodo entero de verdad.
#
# mem=8G. Una corrida 96^3 usa ~0.25 GB de campos mas temporales; 8 GB sobra y
# evita que el planificador la mate por un pico transitorio.
#
# array=0-4: las CINCO trayectorias con una semilla. Para las tres semillas del
# barrido completo, cambiar a 0-14. Se recomienda correr primero una semilla
# (~12 h) y agregar las otras dos cuando el resultado este confirmado: si algo
# esta mal, se pierden doce horas y no treinta.
#
# REENCOLADO AUTOMATICO
# ---------------------
# --time=24:00:00 es una apuesta: una trayectoria a 96^3 lleva del orden de 11 a
# 15 h segun cuanto cueste el entorno del snap, y eso no se sabe de antemano.
# En vez de pedir de mas --que alarga la espera en cola-- el trabajo se REENCOLA
# solo: SLURM manda USR1 quince minutos antes del corte, el manejador pide
# `scontrol requeue` y la corrida arranca de nuevo desde su ultimo checkpoint.
# Con --open-mode=append el log no se pisa, asi que queda la historia completa.
#
# Consecuencia practica: el --time NO tiene que alcanzar para la trayectoria
# entera. Si la cola tiene limite de 4 h, poner 4 y el trabajo dara las vueltas
# que haga falta.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# --------------------------------------------------------------- reencolado --
_reencolar() {
    echo "[$(date +%H:%M:%S)] USR1: la cola va a cortar. Reencolando ${SLURM_JOB_ID}."
    echo "                    Se continua desde el ultimo checkpoint."
    scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null \
        || echo "                    (scontrol requeue fallo; relanzar a mano)"
    exit 0
}
trap _reencolar USR1

# ----------------------------------------------------------------- entorno --
# shellcheck disable=SC1091
source scripts/entorno.sh
if [[ -z "${PY:-}" ]]; then
    echo "ERROR: no hay python con numpy y scipy. Correr scripts/preflight.sh"
    exit 1
fi

# ------------------------------------------------------------- que trayectoria --
TRAYECTORIAS=(P1_uniaxial P2_cizalla P3_confinada
              P4_no_proporcional P5_excursion_grande)
SEMILLAS=(${SEMILLAS_POROK:-20260914 20260915 20260916})

i=${SLURM_ARRAY_TASK_ID:-0}
n=${#TRAYECTORIAS[@]}
TRAY=${TRAYECTORIAS[$(( i % n ))]}
SEM=${SEMILLAS[$(( (i / n) % ${#SEMILLAS[@]} ))]}

MALLA=${MALLA:-96}
DIM=${DIM:-3}
GAMMA=${GAMMA:-0.80}
ESQUEMA_PHI=${ESQUEMA_PHI:-fd}
SALIDA=${SALIDA:-${SCRATCH:-$HOME}/porok_corridas}
mkdir -p "$SALIDA"

echo "=================================================================="
echo " trabajo  : ${SLURM_JOB_ID:-local}  tarea ${i}  (reintento ${SLURM_RESTART_COUNT:-0})"
echo " nodo     : $(hostname)"
echo " trayect. : $TRAY   semilla=$SEM"
echo " malla    : ${MALLA}^${DIM}   gamma_fin=$GAMMA"
echo " hilos    : FFTGK_WORKERS=$FFTGK_WORKERS  (cpus-per-task=${SLURM_CPUS_PER_TASK:-?})"
echo " salida   : $SALIDA"
echo " python   : $PY"
echo "=================================================================="

# En segundo plano + wait, que es lo que permite que el trap de USR1 dispare:
# con el python en primer plano la senal quedaria encolada hasta que terminara.
"$PY" -u scripts/produccion.py \
    --trayectoria "$TRAY" --semilla "$SEM" \
    --n "$MALLA" --dim "$DIM" --gamma-fin "$GAMMA" \
    --max-stag 120 --campos-cada 5 --esquema-phi "$ESQUEMA_PHI" \
    --salida "$SALIDA" &
PID=$!
wait $PID
CODIGO=$?

echo "[$(date +%H:%M:%S)] produccion.py termino con codigo $CODIGO"
exit $CODIGO
