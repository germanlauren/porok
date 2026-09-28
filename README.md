# porok — damage-induced permeability from phase-field fracture and FFT homogenization

Code and data behind *A Second-Order Damage Tensor Is Sufficient for
Damage-Induced Permeability; Scalar Crack Density Is Not*.

German Orlando Romero Suárez¹, Diego Fernando Villegas Bermúdez¹,
Wilmer Velilla Díaz²

¹ Universidad Industrial de Santander, Bucaramanga, Colombia
² Universidad de La Serena, La Serena, Chile

Contact: german2218069@correo.uis.edu.co

## What this is

A periodic-cell reference that grows cracks mechanically and measures the
permeability of the resulting network on the same grid, so that damage state
and permeability are observed at the same instant and the loading path is a
free variable. It is used to bound two families of damage–permeability
closure without fitting anything, and to place the aperture-based closures
used in reservoir practice.

Three results, all reproducible from this archive:

1. Permeability spreads by up to 43.5% among states sharing one scalar crack
   density — the irreducible error of every scalar closure.
2. Matching the full second-order damage tensor collapses that spread to
   3.0% in two dimensions and 0.61% in three, with no residual floor.
3. Aperture closures (cubic law, Oda–Snow) are undefined in 63% of the sweep
   and overpredict by 10² to 3×10⁶ where they are defined, because a moment
   of aperture cannot see connectivity.

## Layout

    src/fftgk/        solver: elasticity, phase field, Darcy homogenization,
                      damage tensor, aperture, closures, HDF5 storage
    scripts/          sweeps, analysis, figures, manuscript generators
    tests/            unit and verification tests
    datos/            state tables (one JSON object per damage state)
    figuras/          manuscript figures, PDF and PNG
    figuras_ecuaciones/  typeset equations
    HPC.md            how the sweeps were run on the cluster

## Data files

| file | contents |
|---|---|
| `datos/resultados_fd.jsonl` | 2D sweep, 210 states: damage tensor, permeability tensor, topology |
| `datos/resultados_kb2d.jsonl` | 2D sweep with aperture, 212 states: adds crack volume, mean aperture, D⁽¹⁾, D⁽³⁾, Oda–Snow, cubic law and the aperture-consistent reference |
| `datos/resultados_3d.jsonl` | 3D sweep at 96³, 410 states |
| `datos/apertura_3d.jsonl` | aperture moments reconstructed from the stored 3D fields |
| `datos/cierres_error_fd.json`, `cierres_error_3d.json` | closure error summaries |
| `datos/verif_apertura.json` | Sneddon verification of the aperture extraction |

## Reproducing

Requires Python 3.11 with NumPy, SciPy, h5py, matplotlib. No compilation.

    python3 scripts/experimento_final.py          # 2D sweep
    POROK_APERTURA=1 python3 scripts/barrido_kb2d.py 3600   # 2D with aperture
    bash scripts/lanzar.sh --completo             # 3D sweep on a cluster
    python3 scripts/cierres_error.py              # closure bounds, 2D
    python3 scripts/cierres_error3d.py            # closure bounds, 3D
    python3 scripts/verif_apertura.py             # Sneddon verification
    python3 scripts/figuras.py && python3 scripts/figura3d.py
    node scripts/manuscrito.js                    # rebuild the manuscript

Every figure in the paper is regenerated from the archived state tables by
those scripts; nothing is drawn by hand.

## License

Code: MIT. Data and figures: CC BY 4.0.
