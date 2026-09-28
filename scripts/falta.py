"""
¿Le falta recorrido a una corrida? Imprime 1 si si, 0 si ya llego.

    python3 scripts/falta.py <salida> <etiqueta> <gamma_fin>

Se usa para decidir si hay que encolar otro tramo en colas que no reencolan
solas (PBS). Mira el ULTIMO Gamma guardado en el archivo de resultados, que es
la unica fuente fiable: el checkpoint guarda el estado del solver pero no la
longitud de grieta, y recalcularla exigiria reconstruirlo entero.

Criterios de "ya llego", en orden:
  * no hay archivo           -> falta (arrancar de cero)
  * el ultimo Gamma >= meta  -> listo
  * los dos ultimos Gamma son iguales -> atascada, NO reencolar (reencolar una
    corrida atascada la deja dando vueltas hasta agotar la cuota)
  * cualquier otro caso      -> falta
"""
import pathlib
import sys


def falta(salida, etiqueta, gamma_fin, tol=1e-9):
    base = pathlib.Path(salida).expanduser()
    h5 = base / f"{etiqueta}.h5"
    if h5.exists():
        try:
            import h5py
            with h5py.File(h5, "r") as f:
                g = f["escalares/Gamma"][...]
        except Exception:
            return True
    else:
        npz = base / etiqueta
        if not npz.is_dir():
            return True
        import numpy as np
        archivos = sorted(npz.glob("inc_*.npz"))
        if not archivos:
            return True
        g = [float(np.load(a)["Gamma"]) for a in archivos[-2:]]

    if len(g) == 0:
        return True
    if float(g[-1]) >= float(gamma_fin) - tol:
        return False
    if len(g) >= 2 and abs(float(g[-1]) - float(g[-2])) < tol:
        return False          # atascada: no insistir
    return True


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("uso: falta.py <salida> <etiqueta> <gamma_fin>")
    print("1" if falta(sys.argv[1], sys.argv[2], float(sys.argv[3])) else "0")
