"""
¿Colapsa K sobre una sola funcion del tensor de dano?

Lee el JSONL del experimento y responde la pregunta central de forma
cuantitativa, no visual.

Metodo. Un cierre basado en estado afirma K = f(D). Si eso fuera cierto, al
interpolar cada trayectoria a los MISMOS valores de tr D las permeabilidades
tendrian que coincidir. La dispersion entre trayectorias a estado de dano
igualado es, por tanto, una medida directa del error irreducible que ningun
cierre de ese tipo puede evitar.

Se comparan tres niveles de "estado", cada vez mas informativo:

  1. tr D             -- escalar, densidad de grieta (lo que usa un modelo de
                         dano isotropo)
  2. (tr D, Dyy/tr D) -- el tensor completo: magnitud y forma. Es lo que usa un
                         cierre de tensor de dano, que es lo implementado en los
                         simuladores de yacimientos.
  3. + descriptor topologico -- si al anadirlo la dispersion colapsa, ese es el
                         ingrediente que falta, y es el resultado constructivo.

Reportar la dispersion en los tres niveles es lo que separa "el tensor de dano
falla" (facil de atacar) de "ninguna familia finita de momentos basta, y aqui
esta lo que si funciona" (defendible).
"""
import sys
import json
import pathlib
import numpy as np

AQUI = pathlib.Path(__file__).resolve().parent
RUTA = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI.parent / "resultados_final.jsonl"


def cargar(ruta):
    filas = [json.loads(l) for l in open(ruta) if l.strip()]
    por = {}
    for f in filas:
        por.setdefault(f["trayectoria"], []).append(f)
    for k in por:
        por[k].sort(key=lambda r: r["trD"])
    return por


def interp(tray, campo, xs):
    x = np.array([r["trD"] for r in tray])
    y = np.array([r[campo] for r in tray])
    ok = (xs >= x.min()) & (xs <= x.max())
    out = np.full(len(xs), np.nan)
    out[ok] = np.interp(xs[ok], x, y)
    return out


def main():
    por = cargar(RUTA)
    nombres = sorted(por)
    print(f"Trayectorias: {', '.join(f'{n} ({len(por[n])} puntos)' for n in nombres)}")
    if len(nombres) < 2:
        print("\nHace falta mas de una trayectoria para responder la pregunta.")
        return

    lo = max(min(r["trD"] for r in por[n]) for n in nombres)
    hi = min(max(r["trD"] for r in por[n]) for n in nombres)
    if not (hi > lo):
        print("\nLas trayectorias no comparten rango de tr D todavia.")
        return
    xs = np.linspace(lo, hi, 12)

    print(f"\nRango comun de tr D: [{lo:.4f}, {hi:.4f}]\n")
    print("[1] K a MISMO tr D  (si un modelo de dano escalar bastara, las")
    print("    columnas serian iguales)\n")
    cab = f"{'tr D':>8} " + " ".join(f"{n[:12]:>13}" for n in nombres)
    print(cab + f" {'disp Kyy':>10} {'razon':>8}")
    Ks = {n: interp(por[n], "Kyy", xs) for n in nombres}
    forma = {n: interp(por[n], "Dyy", xs) / np.maximum(xs, 1e-30) for n in nombres}
    disp = []
    for i, x in enumerate(xs):
        vals = np.array([Ks[n][i] for n in nombres])
        if np.any(np.isnan(vals)):
            continue
        razon = vals.max() / max(vals.min(), 1e-30)
        rel = (vals.max() - vals.min()) / np.mean(vals)
        disp.append((x, rel, razon))
        print(f"{x:8.4f} " + " ".join(f"{v:13.3f}" for v in vals)
              + f" {rel*100:9.1f}% {razon:8.2f}x")

    if disp:
        rels = np.array([d[1] for d in disp])
        razones = np.array([d[2] for d in disp])
        print(f"\n    dispersion mediana {np.median(rels)*100:.1f}%  "
              f"maxima {rels.max()*100:.1f}%  "
              f"razon maxima {razones.max():.2f}x")

    print("\n[2] Forma del tensor a mismo tr D  (Dyy/trD; si difiere, parte de")
    print("    la dispersion anterior la explicaria el tensor completo)\n")
    print(f"{'tr D':>8} " + " ".join(f"{n[:12]:>13}" for n in nombres))
    for i, x in enumerate(xs):
        vals = [forma[n][i] for n in nombres]
        if np.any(np.isnan(vals)):
            continue
        print(f"{x:8.4f} " + " ".join(f"{v:13.4f}" for v in vals))

    print("\n[3] Topologia a mismo tr D\n")
    for campo, etiqueta in (("chi", "caracteristica de Euler"),
                            ("n_clusters", "numero de cumulos"),
                            ("f_mayor", "fraccion del cumulo mayor")):
        vals = {n: interp(por[n], campo, xs) for n in nombres}
        print(f"  {etiqueta}")
        print(f"{'tr D':>10} " + " ".join(f"{n[:12]:>13}" for n in nombres))
        for i, x in enumerate(xs):
            v = [vals[n][i] for n in nombres]
            if np.any(np.isnan(v)):
                continue
            print(f"{x:10.4f} " + " ".join(f"{q:13.3f}" for q in v))
        print()

    print("[4] Percolacion y regimen\n")
    for n in nombres:
        perc = [r for r in por[n] if r["percola_y"] or r["percola_x"]]
        loc = [r for r in por[n] if r["R"] > 1.0]
        print(f"  {n:16s} percola desde trD="
              f"{(min(r['trD'] for r in perc) if perc else float('nan')):.4f}"
              f"   R>1 desde trD="
              f"{(min(r['trD'] for r in loc) if loc else float('nan')):.4f}"
              f"   Kyy final={por[n][-1]['Kyy']:.1f}")


if __name__ == "__main__":
    main()
