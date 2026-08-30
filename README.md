# Adaptive Higher-Order Spreading

Code for paper "Control of Harmful Information Spreading on Adaptive Higher-Order Networks via Group Dissolution", arxiv: https://arxiv.org/abs/2608.07874.

## Files

- `adaptive_hypergraph_simulation.py`: Gillespie simulation and parallel repetitions.
- `mean_field_solver.py`: mean-field steady-state solver.
- `outbreak_threshold.py`: outbreak thresholds for the 3-uniform model.

## Requirements

Python 3.10+ with NumPy, SciPy, and tqdm:

```bash
python -m pip install numpy scipy tqdm
```

## Main APIs

```python
from adaptive_hypergraph_simulation import FullSplitHOSIS_Gillespie, FullSplitParams
from mean_field_solver import Params, solve_equilibrium
from outbreak_threshold import beta_c, mean_hyperdegree
```

Hypergraph node IDs must be integers in `0, ..., N - 1`. Use explicit `seed` and
`master_seed` values for reproducible simulations.
