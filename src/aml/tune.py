"""Grid search with k-fold CV over any Config field.

Every combination in GRID overrides CONFIG. An empty GRID just scores CONFIG,
and a list of dicts searches several separate grids.
Copy the best values into config.py.
"""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from sklearn.model_selection import ParameterGrid

from aml.config import CONFIG
from aml.data import load_data
from aml.pipeline import cross_validate


GRID = {
    "contamination": [0.03, 0.045, 0.06],
    "svr_c": [30, 52, 100],
}


def main() -> None:
    X, y, _ = load_data(CONFIG.data_dir)
    rows = []
    for params in ParameterGrid(GRID):
        scores = cross_validate(replace(CONFIG, **params), X, y)
        rows.append({**params, "r2_mean": scores.mean(), "r2_std": scores.std()})
        print(f"{params} -> R² {scores.mean():.4f} ± {scores.std():.4f}")

    results = pd.DataFrame(rows).sort_values("r2_mean", ascending=False)
    out = Path(CONFIG.output_dir) / "tuning.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nBest first (saved to {out}):\n{results.to_string(index=False)}")


if __name__ == "__main__":
    main()
