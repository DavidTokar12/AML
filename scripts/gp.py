"""Submission: fit the GP once on all inlier training rows, predict the test set.

Run on the HPC after scripts/preprocess.py: uv run python scripts/gp.py
Writes outputs/submission_gp.csv (id, y).
"""

import warnings

import pandas as pd

from sklearn.exceptions import ConvergenceWarning

from aml.config import OUTPUT_DIR
from aml.data import load_features
from aml.models import make_gp


def main() -> None:
    X, y, X_test, inlier = load_features()
    print(f"Fitting on {inlier.sum()} rows, {X.shape[1]} features", flush=True)
    with warnings.catch_warnings():
        # The noise level ends at its lower bound; harmless (see gp_kernel)
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp = make_gp(X.shape[1]).fit(X[inlier], y[inlier])

    pred = gp.predict(X_test)
    path = OUTPUT_DIR / "submission_gp.csv"
    pd.DataFrame({"id": X_test.index, "y": pred}).to_csv(path, index=False)
    print(f"Fitted kernel: {gp.kernel_}")
    print(f"Wrote {path}: {len(pred)} predictions, mean {pred.mean():.1f}")


if __name__ == "__main__":
    main()
