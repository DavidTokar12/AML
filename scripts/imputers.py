"""Which imputer? Same GP and features, only the imputation step changes.

Features: the correlated block (185). Pipeline: quantile transform -> imputer
-> GP (aml.models.make_gp). CV: 5 folds x 3 repeats (the same folds as before),
training folds without the outlier rows, R² on the inlier validation rows.

Imputers:
  knn<k>, knn<k>_dist        KNNImputer, k = 3, 5, 10, 20, 50 neighbours,
                             uniform or distance-weighted (knn5 = current default)
  iterative_linear           IterativeImputer: each feature regressed on the
                             others (BayesianRidge), 10 rounds
  iterative_trees            IterativeImputer with ExtraTrees (MissForest-like),
                             5 rounds; nonlinear, much slower

Run on the HPC: uv run python scripts/imputers.py
Writes outputs/imputers/scores.csv (one row per imputer and fold) and
outputs/imputers/oof.csv (out-of-fold predictions, for ensembling later).
"""

import time
import warnings

import pandas as pd

from joblib import Parallel
from joblib import delayed
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_clean_data
from aml.features import correlated_block
from aml.models import make_gp


N_REPEATS = 3
BASELINE = "knn5"
OUT = OUTPUT_DIR / "imputers"

IMPUTERS = {
    **{
        f"knn{k}{suffix}": (lambda k=k, w=w: KNNImputer(n_neighbors=k, weights=w))
        for k in (3, 5, 10, 20, 50)
        for suffix, w in (("", "uniform"), ("_dist", "distance"))
    },
    "iterative_linear": lambda: IterativeImputer(max_iter=10, random_state=SEED),
    "iterative_trees": lambda: IterativeImputer(
        ExtraTreesRegressor(
            n_estimators=100,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=SEED,
            n_jobs=1,
        ),
        max_iter=5,
        random_state=SEED,
    ),
}


def fit_one(name, repeat, fold, X, y, train, val):
    start = time.time()
    with warnings.catch_warnings():
        # GP noise level at its lower bound (harmless, see gp_kernel) and
        # IterativeImputer stopping at max_iter
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = make_gp(X.shape[1], IMPUTERS[name]())
        model.fit(X.iloc[train], y.iloc[train])
        pred = model.predict(X.iloc[val])
    minutes = (time.time() - start) / 60
    r2 = r2_score(y.iloc[val], pred)
    print(f"  [{name} r{repeat} f{fold}] R² {r2:.4f}, {minutes:.1f} min", flush=True)
    score = {
        "imputer": name,
        "repeat": repeat,
        "fold": fold,
        "r2": r2,
        "minutes": minutes,
    }
    oof = pd.DataFrame(
        {
            "imputer": name,
            "id": X.index[val],
            "repeat": repeat,
            "fold": fold,
            "y": y.iloc[val].to_numpy(),
            "pred": pred,
        }
    )
    return score, oof


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, X_test, inlier = load_clean_data()
    block = set(correlated_block(pd.concat([X, X_test])))
    X = X[[c for c in X.columns if c in block]]
    print(f"{X.shape[1]} features, {len(IMPUTERS)} imputers", flush=True)

    folds = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    tasks = [
        delayed(fit_one)(
            name,
            i // CV_FOLDS + 1,
            i % CV_FOLDS + 1,
            X,
            y,
            train[inlier[train]],
            val[inlier[val]],
        )
        for name in IMPUTERS
        for i, (train, val) in enumerate(folds.split(X))
    ]
    results = Parallel(n_jobs=-1)(tasks)

    scores = pd.DataFrame([s for s, _ in results])
    scores.to_csv(OUT / "scores.csv", index=False)
    pd.concat([o for _, o in results]).to_csv(OUT / "oof.csv", index=False)

    r2 = scores.pivot_table(index=["repeat", "fold"], columns="imputer", values="r2")
    diff = r2.sub(r2[BASELINE], axis=0)
    summary = pd.DataFrame(
        {
            "r2_mean": r2.mean(),
            "r2_std": r2.std(),
            f"vs_{BASELINE}": diff.mean(),
            "folds_won": (diff > 0).sum().astype(str) + f"/{len(r2)}",
            "minutes": scores.groupby("imputer")["minutes"].mean(),
        }
    ).sort_values("r2_mean", ascending=False)
    summary.to_csv(OUT / "summary.csv")
    print(f"\nR² on inlier validation rows, 5 folds x 3 repeats (wrote {OUT}):")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
