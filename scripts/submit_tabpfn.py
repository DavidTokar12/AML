"""Submission: TabPFN on one feature set, fitted on all inlier training rows.

Sets: aml.features.tabpfn_feature_sets (default "significant", the 76 features
with significant TabPFN permutation importance). Raw features (missing values,
no transform), from all 726 cleaned features, so sets outside the block work.

Run on the HPC (GPU): uv run python scripts/submit_tabpfn.py [set]
Writes outputs/submission_tabpfn_<set>.csv (id, y).
"""

import sys
import time

import pandas as pd

from aml.config import OUTPUT_DIR
from aml.data import load_clean_data
from aml.features import correlated_block
from aml.features import tabpfn_feature_sets
from aml.models import make_model


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "significant"
    X, y, X_test, inlier = load_clean_data()
    block = set(correlated_block(pd.concat([X, X_test], ignore_index=True)))
    features = tabpfn_feature_sets(list(X.columns), block)[name]
    print(
        f"TabPFN on '{name}': {len(features)} features, {inlier.sum()} rows", flush=True
    )

    start = time.time()
    model = make_model("tabpfn", len(features)).fit(X.loc[inlier, features], y[inlier])
    pred = model.predict(X_test[features])
    path = OUTPUT_DIR / f"submission_tabpfn_{name}.csv"
    pd.DataFrame({"id": X_test.index.astype(int), "y": pred}).to_csv(path, index=False)
    print(f"Wrote {path}: mean {pred.mean():.1f}, {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
