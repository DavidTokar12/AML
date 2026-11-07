"""Build the locked features once, for every model to reuse.

1. Clean (aml.features.clean_features) and flag the outlier training rows.
2. Keep the correlated block: features correlated with at least one other.
3. Quantile transform + iterative imputation (aml.models.make_preprocessor),
   fitted on train + test together (no y involved).

Run on the HPC: uv run python scripts/preprocess.py
Writes outputs/features/train_raw.csv / test_raw.csv (features before step 3,
with y and inlier), and train.csv / test.csv (after step 3) only if they don't
exist yet; delete them to recompute.
"""

import pandas as pd

from aml.config import FEATURES_DIR
from aml.data import load_clean_data
from aml.features import correlated_block
from aml.models import make_preprocessor


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    X, y, X_test, inlier = load_clean_data()

    # Train and test ids overlap (both start at 0), so split by position
    X_all = pd.concat([X, X_test], ignore_index=True)
    print("Finding the correlated block...", flush=True)
    block = set(correlated_block(X_all))  # computed once, not per column
    features = [c for c in X.columns if c in block]
    # Locked features for the GP/SVR (quantile transform + imputation): only
    # computed if they don't exist yet. Raw ones (missing values, no
    # transform) for the tree models and TabPFN.
    n = len(X)
    ids_train, ids_test = X.index.astype(int), X_test.index.astype(int)
    views = {"_raw": X_all[features]}
    if (FEATURES_DIR / "train.csv").exists() and (FEATURES_DIR / "test.csv").exists():
        print(
            "Locked features exist, skipping quantile transform + imputation",
            flush=True,
        )
    else:
        print(
            f"Quantile transform + imputation of {len(features)} features...",
            flush=True,
        )
        views[""] = make_preprocessor().fit_transform(X_all[features])
    for suffix, data in views.items():
        train = data.iloc[:n].set_axis(ids_train).assign(y=y.to_numpy(), inlier=inlier)
        train.to_csv(FEATURES_DIR / f"train{suffix}.csv", index_label="id")
        data.iloc[n:].set_axis(ids_test).to_csv(
            FEATURES_DIR / f"test{suffix}.csv", index_label="id"
        )
        print(f"Wrote train{suffix}.csv, test{suffix}.csv", flush=True)


if __name__ == "__main__":
    main()
