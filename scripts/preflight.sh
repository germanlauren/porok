#!/bin/bash
# Que hay en este cluster, ANTES de encolar nada.
#
#   bash scripts/preflight.sh
#
# No envia trabajos, no escribe en scratch, no instala nada. Solo mira y
# reporta, y al final imprime el comando de lanzamiento que corresponde.
#
# Existe porque encolar a ciegas en una maquina ajena sale caro: un trabajo que
# muere en el primer segundo por un import que falta puede haber esperado seis
# horas en cola, y el error no se ve hasta que se mira el .err.

set -uo pipefail
cd "$(dirname "$0")/.."

linea() { printf '%s\n' "======================================================================"; }

linea; echo " 1. QUIEN Y DONDE"; linea
echo "  usuario        : $(whoami)"
echo "  maquina        : $(hostname)"
echo "  sistema        : $(uname -sr)"
[[ -f /etc/os-release ]] && echo "  distro         : $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "  directorio     : $(pwd)"

linea; echo " 2. PLANIFICADOR"; linea
PLAN=""
if command -v sbatch >/dev/null 2>&1; then
    PLAN=slurm
    echo "  SLURM          : $(sbatch --version 2>/dev/null | head -1)"
    echo
    echo "  particiones disponibles:"
    sinfo -o "  %-18P %-6a %-12l %-6D %-10t %N" 2>/dev/null | head -20 \
        || echo "    (sinfo no responde)"
    echo
    echo "  tus trabajos en cola:"
    squeue -u "$(whoami)" 2>/dev/null | head -10 || echo "    (ninguno)"
    echo
    lim=$(sacctmgr -n show qos format=name,maxwall 2>/dev/null | head -5)
    [[ -n "$lim" ]] && { echo "  limites de QOS:"; echo "$lim" | sed 's/^/    /'; }
elif command -v qsub >/dev/null 2>&1; then
    PLAN=pbs
    echo "  PBS/Torque     : $(qstat --version 2>&1 | head -1)"
    echo
    echo "  colas:"
    qstat -Q 2>/dev/null | head -15 || echo "    (qstat -Q no responde)"
else
    PLAN=ninguno
    echo "  NO hay planificador: se lanza directo con nohup."
fi

linea; echo " 3. NUCLEOS Y MEMORIA DE ESTE NODO"; linea
echo "  cpu_count      : $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')"
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' 2>/dev/null
import os
try:
    print(f"  afinidad       : {len(os.sched_getaffinity(0))}  (lo que ESTE proceso puede usar)")
except Exception:
    pass
try:
    with open("/proc/meminfo") as f:
        mi = {l.split(':')[0]: int(l.split()[1]) * 1024 for l in f}
    print(f"  memoria        : {mi['MemTotal']/2**30:.1f} GB total,"
          f" {mi.get('MemAvailable',0)/2**30:.1f} GB disponible")
except Exception:
    pass
PY
fi
echo
echo "  OJO: el nodo de login NO es el nodo de computo. Estos numeros valen"
echo "  para el login; los del nodo real salen al correr sondeo_hpc.py dentro"
echo "  de un trabajo."

linea; echo " 4. ENTORNO DE PYTHON"; linea
# shellcheck disable=SC1091
source scripts/entorno.sh
if [[ -z "${PY:-}" ]]; then
    echo "  NO se encontro un python con numpy y scipy."
    echo "  Instalar con:  conda install -c conda-forge numpy scipy h5py"
    ENTORNO_OK=0
else
    echo "  interprete     : $PY"
    "$PY" - <<'PY'
import importlib, sys
print(f"  python         : {sys.version.split()[0]}")
for m in ("numpy", "scipy", "h5py"):
    try:
        print(f"  {m:14s} : {importlib.import_module(m).__version__}")
    except ImportError:
        print(f"  {m:14s} : FALTA")
PY
fi

linea; echo " 5. EL CODIGO CORRE"; linea
if [[ -n "${PY:-}" ]]; then
    if "$PY" -c "
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np
from produccion import campo_Gc, direccion, CAMBIO
from fftgk.phasefield import PhaseFieldSolver
from fftgk.elasticity import lame_from_E_nu
Gc = campo_Gc((16,16,16), 1)
lam, mu = lame_from_E_nu(1.0, 0.2)
S = PhaseFieldSolver((16,)*3, lam, mu, Gc, 4.0/16, L=(1.0,)*3)
S.paso_longitud(direccion('P1_uniaxial', 3), 0.02, lam_ini=0.05, max_stag=5)
assert 'P5_excursion_grande' in CAMBIO
print('  prueba 16^3    : OK (incluye las trayectorias no proporcionales)')
" 2>&1 | tail -3; then :; else
        echo "  la prueba FALLO -- no encolar hasta resolverlo"
    fi
fi

linea; echo " 6. DONDE ESCRIBIR"; linea
for d in "${SCRATCH:-}" "/scratch/$(whoami)" "/lustre/$(whoami)" "$HOME"; do
    [[ -z "$d" ]] && continue
    if [[ -d "$d" && -w "$d" ]]; then
        libre=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -d ' G')
        echo "  $d  (libre: ${libre:-?} GB)"
    fi
done
echo
echo "  El barrido completo son 15 archivos .h5. A 96^3 con campos cada 5"
echo "  incrementos, cada uno pesa del orden de 1 a 2 GB: presupuestar 30 GB."

linea; echo " 7. QUE LANZAR"; linea
case "$PLAN" in
  slurm)
    echo "  bash scripts/lanzar.sh              # una semilla, 5 corridas"
    echo "  bash scripts/lanzar.sh --completo   # tres semillas, 15 corridas"
    echo "  bash scripts/lanzar.sh --ensayo     # ver el comando sin enviarlo"
    echo
    echo "  (equivale a sbatch scripts/trabajo.slurm con el array correcto)"
    echo "  Ajustar antes, arriba de trabajo.slurm: --partition, --account, --time."
    echo "  El trabajo se REENCOLA solo si la cola lo corta, asi que el --time"
    echo "  no tiene que alcanzar para la trayectoria entera."
    ;;
  pbs)
    echo "  bash scripts/lanzar.sh              # una semilla, 5 corridas"
    echo "  bash scripts/lanzar.sh --completo   # tres semillas, 15 corridas"
    echo
    echo "  (equivale a qsub scripts/trabajo.pbs)"
    echo "  Ajustar antes, arriba de trabajo.pbs: -q (cola), -l walltime, -A."
    ;;
  *)
    echo "  bash scripts/lanzar.sh              # 5 corridas, 6 hilos cada una"
    echo "  bash scripts/lanzar.sh --completo   # las 15"
    ;;
esac
echo
echo "  Todo es REANUDABLE: si algo muere, el mismo comando continua desde el"
echo "  ultimo checkpoint. No hay que borrar nada."
linea
