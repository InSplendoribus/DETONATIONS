
# deTONAtions

Dusty Endevours at TONAntzintla: Turbulent Interplay Of Nebulae and Shocks

Modular Eulerian Python project for 3-D hydrodynamical simulations.

## Layout
(see tree inside archive)

## Setup
```
pip install numpy scipy matplotlib
```

## Run (single process)
```
python -m detonations.main detonations.par
```

## Run (parallel/multiple)
Launch separate processes as needed.

## Parameter reference
See `detonations.par` for all keys grouped by sections with Python-literal values.

## Outputs
Figures are written to files by the plotting module; ASCII slices optional.
