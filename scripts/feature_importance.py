"""Score every feature with two methods: random forest and mutual information.

Data: the cleaned features (notebook steps 1-6), training rows without the
flagged outlier rows, plus 20 shadow features (shuffled copies) that every
method scores too, as its noise level.

Methods:
  rf     random forest, out-of-fold permutation importance
  mi     mutual information with y, per feature on the rows where it is present

Gradient boosting, SVR and Lasso stability selection were tried too and dropped:
they separate the correlated block from the noise features much worse (AUC
0.70, 0.79, 0.37 vs 0.94 for RF and 0.85 for MI).

Run on the HPC: uv run python scripts/feature_importance.py [method ...]
Writes outputs/importance_<method>.csv; methods with an existing file are skipped.
"""

import sys
import time

from sklearn.ensemble import RandomForestRegressor

from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_clean_data
from aml.features import add_shadows
from aml.features import mutual_information
from aml.features import permutation_importance_cv


METHODS = {
    "rf": lambda X, y: permutation_importance_cv(
        RandomForestRegressor(
            n_estimators=1000, max_features="sqrt", random_state=SEED, n_jobs=-1
        ),
        X,
        y,
        seed=SEED,
    ),
    "mi": lambda X, y: mutual_information(X, y, seed=SEED),
}


def main() -> None:
    methods = sys.argv[1:] or list(METHODS)
    OUTPUT_DIR.mkdir(exist_ok=True)

    X, y, _, inlier = load_clean_data()
    X = add_shadows(X[inlier], seed=SEED)
    y = y[inlier]
    print(f"{X.shape[1]} features (incl. 20 shadows), {len(X)} rows", flush=True)

    for name in methods:
        path = OUTPUT_DIR / f"importance_{name}.csv"
        if path.exists():
            print(f"=== {name}: {path} exists, skipping ===", flush=True)
            continue
        print(f"=== {name} ===", flush=True)
        start = time.time()
        imp = METHODS[name](X, y)
        imp.to_csv(path)
        is_shadow = imp.index.str.startswith("shadow_")
        n_beat = (imp.loc[~is_shadow, "mean"] > imp.loc[is_shadow, "mean"].max()).sum()
        print(
            f"=== {name}: {n_beat} features beat the best shadow, "
            f"{(time.time() - start) / 60:.1f} min, wrote {path} ===\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
