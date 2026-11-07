"""Weighted ensemble: learn non-negative weights on the out-of-fold predictions,
apply them to the test predictions.

Weights: non-negative least squares on outputs/cv/<model>.csv (scripts/cv.py).
They are validated by fitting on 4 folds and scoring the 5th, then refitted on
all folds and applied to outputs/submission_<model>.csv (scripts/submit.py).

Run after cv.py and submit.py (seconds, fine on the laptop):
  uv run python scripts/ensemble.py [model ...]     (default: gp tabpfn)
Writes outputs/submission_ensemble.csv (id, y).
"""

import sys

import numpy as np
import pandas as pd

from scipy.optimize import nnls
from sklearn.metrics import r2_score

from aml.config import OUTPUT_DIR


def r2_per_fold(d: pd.DataFrame, pred: np.ndarray) -> float:
    return float(
        np.mean([r2_score(g["y"], pred[g.index]) for _, g in d.groupby("fold")])
    )


def main() -> None:
    models = sys.argv[1:] or ["gp", "tabpfn"]
    d = None
    for m in models:
        oof = pd.read_csv(OUTPUT_DIR / "cv" / f"{m}.csv")[["id", "fold", "y", "pred"]]
        oof = oof.rename(columns={"pred": m})
        d = oof if d is None else d.merge(oof, on=["id", "fold", "y"])
    d = d.reset_index(drop=True)

    pred = np.zeros(len(d))
    for fold in d["fold"].unique():
        train, val = d["fold"] != fold, d["fold"] == fold
        w, _ = nnls(d.loc[train, models].to_numpy(), d.loc[train, "y"].to_numpy())
        pred[val.to_numpy()] = d.loc[val, models].to_numpy() @ w
    for m in models:
        print(f"{m:12s} R² {r2_per_fold(d, d[m].to_numpy()):.4f}")
    print(f"{'ensemble':12s} R² {r2_per_fold(d, pred):.4f} (weights fitted on 4 folds)")

    w, _ = nnls(d[models].to_numpy(), d["y"].to_numpy())
    print("weights:", dict(zip(models, w.round(3), strict=True)))
    test = None
    for m in models:
        sub = pd.read_csv(OUTPUT_DIR / f"submission_{m}.csv").rename(columns={"y": m})
        test = sub if test is None else test.merge(sub, on="id")
    path = OUTPUT_DIR / "submission_ensemble.csv"
    pd.DataFrame({"id": test["id"], "y": test[models].to_numpy() @ w}).to_csv(
        path, index=False
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
