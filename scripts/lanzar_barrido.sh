#!/bin/bash
# Lanzador de barrido para UNA maquina con muchos nucleos, sin planificador.
#
# Por que asi y no una sola corrida con todos los hilos
# -----------------------------------------------------
# Medido en el nodo (32 nucleos, scipy.fft):
#
#     forma        numpy    scipy 1 hilo   scipy 32 hilos
#     128^3       35.9 ms      13.4 ms        2.7 ms
#
# La FFT satura: de 1 a 32 hilos gana ~5x, no 32x. Pero un barrido es
# vergonzosamente paralelo -- trayectorias y semillas son independientes -- asi
# que N procesos de pocos hilos rinden mucho mas que un proceso de muchos:
#
#     1 proceso  x 32 hilos :  1 / 2.7 ms   = 0.37 FFT/ms
#     8 procesos x  4 hilos :  8 / ~4.5 ms  = 1.78 FFT/ms    <- ~5x mas
#
# Y como una sola semilla no basta para publicar (la celda finita tiene
# fluctuacion propia, doc 01 seccion 4), de todas formas hacen falta varias
# corridas. Sale gratis.
#
# Uso:
#   bash scripts/lanzar_barrido.sh                  # valores por defecto
#   NPROC=6 HILOS=5 MALLA=160 DIM=3 bash scripts/lanzar_barrido.sh
#   bash scripts/lanzar_barrido.sh --seguir         # ver el avance

set -uo pipefail
cd "$(dirname "$0")/.."

NPROC=${NPROC:-8}          # procesos simultaneos
HILOS=${HILOS:-4}          # hilos de FFT por proceso
MALLA=${MALLA:-96}
DIM=${DIM:-3}
GAMMA=${GAMMA:-0.80}
# Punto de rotacion de P4/P5 = fraccion * GAMMA_REF. Al extender una corrida
# ya hecha a un GAMMA mayor, dejar GAMMA_REF en el GAMMA original.
GAMMA_REF=${GAMMA_REF:-$GAMMA}
# Argumentos extra para produccion.py, p.ej. EXTRA="--apertura" (doc 22).
EXTRA=${EXTRA:-}
SALIDA=${SALIDA:-$HOME/porok_corridas}
LOGS=${LOGS:-$SALIDA/logs}

# Las CINCO del articulo. P4 y P5 son no proporcionales: rotan la direccion de
# carga a mitad de camino, y son las que sostienen el resultado de memoria.
# Sin ellas el barrido 3D no reproduce el hallazgo central.
TRAYECTORIAS=(P1_uniaxial P2_cizalla P3_confinada P4_no_proporcional P5_excursion_grande)
# Una semilla por defecto. Las tres del barrido completo se piden con
# SEMILLAS_POROK, que es la misma variable que usan trabajo.slurm y trabajo.pbs,
# para que las tres rutas de lanzamiento se controlen igual.
SEMILLAS=(${SEMILLAS_POROK:-20260914 20260915 20260916})

if [[ "${1:-}" == "--seguir" ]]; then
    echo "Avance (Ctrl-C para salir):"
    tail -n 3 -f "$LOGS"/*.log
    exit 0
fi

mkdir -p "$SALIDA" "$LOGS"

# Un hilo de BLAS por proceso: si OpenBLAS y scipy.fft se pelean por los mismos
# nucleos, el conjunto rinde MENOS que con un solo hilo.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export FFTGK_WORKERS=$HILOS

echo "=================================================================="
echo " barrido: ${#TRAYECTORIAS[@]} trayectorias x ${#SEMILLAS[@]} semillas"
echo " malla ${MALLA}^${DIM}   ${NPROC} procesos x ${HILOS} hilos"
echo " salida: $SALIDA"
echo "=================================================================="

cola=()
for s in "${SEMILLAS[@]}"; do
  for t in "${TRAYECTORIAS[@]}"; do
    cola+=("$t:$s")
  done
done

lanzados=0
for item in "${cola[@]}"; do
    tray=${item%%:*}
    sem=${item##*:}
    # esperar a que haya cupo
    while (( $(jobs -rp | wc -l) >= NPROC )); do sleep 5; done
    log="$LOGS/${tray}_s${sem}.log"
    echo "[$(date +%H:%M:%S)] lanzando $tray semilla=$sem -> $log"
    nohup python3 -u scripts/produccion.py \
        --trayectoria "$tray" --semilla "$sem" \
        --n "$MALLA" --dim "$DIM" --gamma-fin "$GAMMA" --gamma-ref "$GAMMA_REF" \
        --max-stag 120 --campos-cada 5 --max-incrementos 400 $EXTRA --esquema-phi "${ESQUEMA_PHI:-fd}" \
        --salida "$SALIDA" >> "$log" 2>&1 &
    lanzados=$((lanzados+1))
    sleep 2
done

echo "$lanzados corridas lanzadas. Esperando..."
wait
echo "[$(date +%H:%M:%S)] barrido terminado. Resultados en $SALIDA"
echo "Todas son REANUDABLES: si algo murio, volver a correr este mismo script."
