"""Cross-validate models on the locked features, for the ensemble.

For each model: try every config of its small grid on the same 5 folds x 3
repeats as the GP (training folds without the outlier rows, R² on the inlier
validation rows), keep the best config, save its out-of-fold predictions, then
refit it on all inlier rows and predict the test set.

Models:
  gp          the GP (fixed config, see aml.models), for its OOF predictions
  svr         SVR with an RBF kernel: robust to noisy targets (epsilon loss)
  catboost    gradient boosting built for small noisy tabular data
  extratrees  random-split forest: lower variance than a random forest

Run on the HPC after scripts/preprocess.py:
  uv run python scripts/model_cv.py [model ...]     (default: all)
Writes outputs/models/<model>/grid.csv (R² per config), oof.csv, test.csv.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from joblib import Parallel
from joblib import delayed
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVR

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_features
from aml.models import make_gp


N_REPEATS = 3
OUT = OUTPUT_DIR / "models"

# name -> (grid, build(params, n_features)); every model runs single-threaded,
# the parallelism is over configs and folds
MODELS = {
    "gp": ({}, lambda p, d: make_gp(d)),
    "svr": (
        # features are ~N(0, 1), so gamma = 1/d is sklearn's "scale"
        {"C": [10, 30, 100, 300], "epsilon": [1, 3, 5], "gamma_x_d": [0.5, 1, 2]},
        lambda p, d: SVR(
            kernel="rbf", C=p["C"], epsilon=p["epsilon"], gamma=p["gamma_x_d"] / d
        ),
    ),
    "catboost": (
        {"depth": [4, 6, 8], "l2_leaf_reg": [3, 10], "learning_rate": [0.02, 0.05]},
        lambda p, d: CatBoostRegressor(
            iterations=2000,
            rsm=0.5,
            random_seed=SEED,
            thread_count=1,
            verbose=0,
            allow_writing_files=False,
            **p,
        ),
    ),
    "extratrees": (
        {"max_features": [0.1, 0.3, 1.0], "min_samples_leaf": [1, 3, 5]},
        lambda p, d: ExtraTreesRegressor(
            n_estimators=1000, random_state=SEED, n_jobs=1, **p
        ),
    ),
}


def fit_predict(name, params, X, y, train, X_pred):
    with warnings.catch_warnings():
        # GP noise level at its lower bound: harmless (see aml.models.gp_kernel)
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = MODELS[name][1](params, X.shape[1])
        model.fit(X.iloc[train], y.iloc[train])
    return model.predict(X_pred)


def run_fold(name, c, params, repeat, fold, X, y, train, val):
    pred = fit_predict(name, params, X, y, train, X.iloc[val])
    r2 = r2_score(y.iloc[val], pred)
    oof = pd.DataFrame(
        {
            "id": X.index[val],
            "repeat": repeat,
            "fold": fold,
            "y": y.iloc[val].to_numpy(),
            "pred": pred,
        }
    )
    return c, {"config": c, "repeat": repeat, "fold": fold, "r2": r2}, oof


def run_model(name, X, y, X_test, inlier, folds):
    configs = list(ParameterGrid(MODELS[name][0]))
    print(f"=== {name}: {len(configs)} configs x {len(folds)} folds ===", flush=True)
    start = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(run_fold)(
            name,
            c,
            params,
            i // CV_FOLDS + 1,
            i % CV_FOLDS + 1,
            X,
            y,
            train[inlier[train]],
            val[inlier[val]],
        )
        for c, params in enumerate(configs)
        for i, (train, val) in enumerate(folds)
    )
    scores = pd.DataFrame([s for _, s, _ in results])
    grid = scores.groupby("config")["r2"].agg(["mean", "std"])
    grid["params"] = [str(p) for p in configs]
    grid = grid.sort_values("mean", ascending=False)
    best = grid.index[0]
    print(grid.head(5).round(4).to_string(), flush=True)

    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "grid.csv")
    pd.concat([o for c, _, o in results if c == best]).sort_values(
        ["repeat", "fold", "id"]
    ).to_csv(out / "oof.csv", index=False)
    pred = fit_predict(name, configs[best], X, y, np.where(inlier)[0], X_test)
    pd.DataFrame({"id": X_test.index, "pred": pred}).to_csv(
        out / "test.csv", index=False
    )
    print(
        f"=== {name}: best {configs[best]} R² {grid.loc[best, 'mean']:.4f} "
        f"± {grid.loc[best, 'std']:.4f}, {(time.time() - start) / 60:.1f} min ===\n",
        flush=True,
    )


def main() -> None:
    names = sys.argv[1:] or list(MODELS)
    X, y, X_test, inlier = load_features()
    folds = list(
        RepeatedKFold(n_splits=CV_FOLDS, n_repeats=N_REPEATS, random_state=SEED).split(
            X
        )
    )
    print(f"{X.shape[1]} features, {inlier.sum()} inlier training rows", flush=True)
    for name in names:
        run_model(name, X, y, X_test, inlier, folds)


if __name__ == "__main__":
    main()
