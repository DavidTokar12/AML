"""Submissions: fit each model once on all inlier training rows, predict the test set.

Configs: aml.models.make_model (the tuned ones). All use the locked features
except TabPFN, which gets the raw ones (aml.models.RAW_INPUT).

Run on the HPC after scripts/preprocess.py:
  uv run python scripts/submit.py [model ...]     (default: all)
Writes outputs/submission_<model>.csv (id, y).
"""

import sys
import time
import warnings

import pandas as pd

from sklearn.exceptions import ConvergenceWarning

from aml.config import OUTPUT_DIR
from aml.data import load_features
from aml.models import MODEL_NAMES
from aml.models import RAW_INPUT
from aml.models import make_model


def main() -> None:
    for name in sys.argv[1:] or MODEL_NAMES:
        X, y, X_test, inlier = load_features(raw=name in RAW_INPUT)
        print(
            f"{name}: fitting on {inlier.sum()} rows, {X.shape[1]} features "
            f"({'raw' if name in RAW_INPUT else 'locked'})",
            flush=True,
        )
        start = time.time()
        with warnings.catch_warnings():
            # GP noise level at its lower bound: harmless (see aml.models.gp_kernel)
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = make_model(name, X.shape[1]).fit(X[inlier], y[inlier])
        pred = model.predict(X_test)
        path = OUTPUT_DIR / f"submission_{name}.csv"
        pd.DataFrame({"id": X_test.index, "y": pred}).to_csv(path, index=False)
        print(
            f"  wrote {path}: mean {pred.mean():.1f}, "
            f"{(time.time() - start) / 60:.1f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
