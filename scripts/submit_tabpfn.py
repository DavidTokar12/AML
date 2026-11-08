"""Submission: TabPFN with the best settings of notebooks/tabpfn.ipynb.

- features: one set of aml.features.tabpfn_feature_sets (default "block", 185),
  raw (missing values, no transform)
- all training rows: outlier removal hurt TabPFN (inlier R² 0.767 with all
  rows vs 0.756 without the 4.5% flagged rows)
- n_estimators=32 (+0.002), 5 seeds averaged (+0.003), feature values softly
  clipped at 4 std (+0.001)

Run on the HPC (GPU): uv run python scripts/submit_tabpfn.py [set]
Writes outputs/submission_tabpfn_<set>_best.csv (id, y).
"""

import sys
import time

import numpy as np
import pandas as pd

from tabpfn import TabPFNRegressor

from aml.config import OUTPUT_DIR
from aml.data import load_clean_data
from aml.features import correlated_block
from aml.features import tabpfn_feature_sets


N_ESTIMATORS = 32
SEEDS = range(5)
CLIP_STD = 4.0


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "block"
    X, y, X_test, _ = load_clean_data()  # the outlier mask is not used
    block = set(correlated_block(pd.concat([X, X_test], ignore_index=True)))
    features = tabpfn_feature_sets(list(X.columns), block)[name]
    print(
        f"TabPFN on '{name}': {len(features)} features, all {len(X)} rows, "
        f"{N_ESTIMATORS} estimators, {len(SEEDS)} seeds",
        flush=True,
    )

    preds = []
    for seed in SEEDS:
        start = time.time()
        model = TabPFNRegressor(
            device="auto",
            random_state=seed,
            n_estimators=N_ESTIMATORS,
            inference_config={"OUTLIER_REMOVAL_STD": CLIP_STD},
        ).fit(X[features], y)
        preds.append(model.predict(X_test[features]))
        print(f"  seed {seed}: {(time.time() - start) / 60:.1f} min", flush=True)

    pred = np.mean(preds, axis=0)
    path = OUTPUT_DIR / f"submission_tabpfn_{name}_best.csv"
    pd.DataFrame({"id": X_test.index.astype(int), "y": pred}).to_csv(path, index=False)
    print(
        f"Wrote {path}: mean {pred.mean():.1f}, seed spread {np.std(preds, axis=0).mean():.2f}"
    )


if __name__ == "__main__":
    main()
