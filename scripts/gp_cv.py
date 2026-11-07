"""Cross-validate the GP (Matern 2.5, see aml.models.gp_kernel) on the
correlated block, saving every fit's predictions for later ensembling.

Preprocessing: quantile transform to normal, then KNNImputer (default).
CV: 5 folds x 3 repeats. Training folds exclude the outlier rows, R² is scored
on the inlier validation rows (predictions are saved for all rows). One more
fit on all inlier training rows predicts the test set.

Outputs in outputs/gp/:
  oof.csv   out-of-fold predictions: id, repeat, fold, y, pred, std, inlier
  test.csv  test predictions: id, pred, std
  fits.csv  per fit: repeat, fold (0 = full fit), r2, fitted kernel, minutes

Run on the HPC: uv run python scripts/gp_cv.py
"""

import time
import warnings

import numpy as np
import pandas as pd

from joblib import Parallel
from joblib import delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_clean_data
from aml.features import correlated_block
from aml.models import make_gp


N_REPEATS = 3
OUT = OUTPUT_DIR / "gp"


def fit_one(repeat, fold, train, val, X, y, X_test, inlier):
    """One fit. fold 0 = fit on all inlier rows and predict the test set."""
    start = time.time()
    with warnings.catch_warnings():
        # The noise level ends at its lower bound; harmless (see gp_kernel)
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = make_gp(X.shape[1])
        model.fit(X.iloc[train], y.iloc[train])

    if fold == 0:
        pred, std = model.predict(X_test, return_std=True)
        rows = pd.DataFrame({"id": X_test.index, "pred": pred, "std": std})
        r2 = np.nan
    else:
        pred, std = model.predict(X.iloc[val], return_std=True)
        rows = pd.DataFrame(
            {
                "id": X.index[val],
                "repeat": repeat,
                "fold": fold,
                "y": y.iloc[val].to_numpy(),
                "pred": pred,
                "std": std,
                "inlier": inlier[val],
            }
        )
        scored = rows[rows["inlier"]]
        r2 = r2_score(scored["y"], scored["pred"])

    fit = {
        "repeat": repeat,
        "fold": fold,
        "r2": r2,
        "kernel": str(model[-1].kernel_),
        "minutes": (time.time() - start) / 60,
    }
    print(f"  [r{repeat} f{fold}] R² {r2:.4f}, {fit['minutes']:.1f} min", flush=True)
    return fit, rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, X_test, inlier = load_clean_data()

    features = [
        c for c in X.columns if c in set(correlated_block(pd.concat([X, X_test])))
    ]
    X, X_test = X[features], X_test[features]
    print(
        f"{len(features)} features, {len(X)} rows ({inlier.sum()} inliers)", flush=True
    )

    folds = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    tasks = [
        delayed(fit_one)(
            i // CV_FOLDS + 1,
            i % CV_FOLDS + 1,
            train[inlier[train]],
            val,
            X,
            y,
            X_test,
            inlier,
        )
        for i, (train, val) in enumerate(folds.split(X))
    ]
    tasks.append(
        delayed(fit_one)(0, 0, np.where(inlier)[0], None, X, y, X_test, inlier)
    )
    results = Parallel(n_jobs=-1)(tasks)

    fits = pd.DataFrame([f for f, _ in results]).sort_values(["repeat", "fold"])
    fits.to_csv(OUT / "fits.csv", index=False)
    oof = [r for f, r in results if f["fold"] > 0]
    pd.concat(oof).sort_values(["repeat", "fold", "id"]).to_csv(
        OUT / "oof.csv", index=False
    )
    next(r for f, r in results if f["fold"] == 0).to_csv(OUT / "test.csv", index=False)
    cv = fits[fits["fold"] > 0]["r2"]
    print(f"R² {cv.mean():.4f} ± {cv.std():.4f}, wrote {OUT}")


if __name__ == "__main__":
    main()
