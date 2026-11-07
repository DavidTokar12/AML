"""Build the locked features once, for every model to reuse.

1. Clean (aml.features.clean_features) and flag the outlier training rows.
2. Keep the correlated block: features correlated with at least one other.
3. Quantile transform + iterative imputation (aml.models.make_preprocessor),
   fitted on train + test together (no y involved).

Run on the HPC: uv run python scripts/preprocess.py
Writes outputs/features/train.csv (features, y, inlier) and test.csv.
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
    features = [c for c in X.columns if c in set(correlated_block(X_all))]
    Z = make_preprocessor().fit_transform(X_all[features])

    n = len(X)
    train = (
        Z.iloc[:n].set_axis(X.index.astype(int)).assign(y=y.to_numpy(), inlier=inlier)
    )
    test = Z.iloc[n:].set_axis(X_test.index.astype(int))
    train.to_csv(FEATURES_DIR / "train.csv", index_label="id")
    test.to_csv(FEATURES_DIR / "test.csv", index_label="id")
    print(
        f"{len(features)} features, {len(train)} train rows ({inlier.sum()} inliers), "
        f"{len(test)} test rows, missing left: {int(Z.isna().sum().sum())}; "
        f"wrote {FEATURES_DIR}"
    )


if __name__ == "__main__":
    main()
