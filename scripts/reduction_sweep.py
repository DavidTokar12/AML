"""Does reducing the 185 locked features to a few components help the GP / SVR?

Reductions (fitted inside each training fold; PLS uses y):
  none  all 185 locked features (the reference)
  pls   PLS components, k = 3, 5, 8, 10, 15, 20, standardized
  pca   PCA components, k = 14 (above the shuffled-data level), 35, 60, 100
Models:
  gp      the GP (Matern 2.5, one length scale shared by all components)
  gp_ard  the same with one length scale per component (PLS only: with few
          components this no longer overfits like it did on 185 features)
  svr     SVR, C=100, epsilon=1, gamma="scale"
CV: 5 folds x 3 repeats (the usual folds), training folds without the outlier
rows, R² on the inlier validation rows.

Run on the HPC after scripts/preprocess.py: uv run python scripts/reduction_sweep.py
Writes outputs/reduction/scores.csv (one row per model, method, k, fold).
"""

import time
import warnings

import numpy as np
import pandas as pd

from joblib import Parallel
from joblib import delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_features
from aml.models import make_gp


N_REPEATS = 3
OUT = OUTPUT_DIR / "reduction"
REDUCTIONS = (
    [("none", 185)]
    + [("pls", k) for k in (3, 5, 8, 10, 15, 20)]
    + [("pca", k) for k in (14, 35, 60, 100)]
)


def reduce(method, k, X_train, y_train, X_val):
    """Fit the reduction on the training fold, apply it to both folds."""
    if method == "none":
        return X_train.to_numpy(), X_val.to_numpy()
    if method == "pca":
        pca = PCA(k, random_state=SEED).fit(X_train)
        return pca.transform(X_train), pca.transform(X_val)
    pls = PLSRegression(k, scale=False).fit(X_train, y_train)
    scaler = StandardScaler().fit(pls.transform(X_train))
    return (
        scaler.transform(pls.transform(X_train)),
        scaler.transform(pls.transform(X_val)),
    )


def make(model, d):
    if model == "gp":
        return make_gp(d)
    if model == "gp_ard":
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            np.full(d, np.sqrt(d)), (1e-2, 1e5), nu=2.5
        ) + WhiteKernel(0.3, (1e-5, 10.0))
        return GaussianProcessRegressor(
            kernel, normalize_y=True, n_restarts_optimizer=2, random_state=SEED
        )
    return SVR(kernel="rbf", C=100, epsilon=1, gamma="scale")


def run(model, method, k, repeat, fold, X, y, train, val):
    start = time.time()
    Z_train, Z_val = reduce(method, k, X.iloc[train], y.iloc[train], X.iloc[val])
    with warnings.catch_warnings():
        # GP parameters at their bounds (noise level, unused length scales)
        warnings.simplefilter("ignore", ConvergenceWarning)
        pred = make(model, Z_train.shape[1]).fit(Z_train, y.iloc[train]).predict(Z_val)
    return {
        "model": model,
        "method": method,
        "k": k,
        "repeat": repeat,
        "fold": fold,
        "r2": r2_score(y.iloc[val], pred),
        "minutes": (time.time() - start) / 60,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, _, inlier = load_features()
    folds = list(
        RepeatedKFold(n_splits=CV_FOLDS, n_repeats=N_REPEATS, random_state=SEED).split(
            X
        )
    )
    tasks = [
        delayed(run)(
            model,
            method,
            k,
            i // CV_FOLDS + 1,
            i % CV_FOLDS + 1,
            X,
            y,
            tr[inlier[tr]],
            va[inlier[va]],
        )
        for model in ("gp", "gp_ard", "svr")
        for method, k in REDUCTIONS
        if model != "gp_ard" or method == "pls"
        for i, (tr, va) in enumerate(folds)
    ]
    print(f"{len(tasks)} fits", flush=True)
    rows = []
    start = time.time()
    for done, row in enumerate(
        Parallel(n_jobs=-1, return_as="generator_unordered")(tasks), 1
    ):
        rows.append(row)
        if done % 50 == 0:
            print(
                f"  {done}/{len(tasks)} fits, {(time.time() - start) / 60:.1f} min",
                flush=True,
            )

    scores = pd.DataFrame(rows).sort_values(["model", "method", "k", "repeat", "fold"])
    scores.to_csv(OUT / "scores.csv", index=False)
    table = scores.pivot_table(
        index=["method", "k"], columns="model", values="r2", aggfunc="mean"
    )
    print(f"\nMean R² over {len(folds)} folds (wrote {OUT / 'scores.csv'}):")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
