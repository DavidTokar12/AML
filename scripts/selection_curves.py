"""How many features? Top-k curves for three rankings of the union set.

Union set: features RF or MI select (above their best shadow), plus the
correlated block. Everything else is dropped.
Rankings of the union set:
  rf    RF permutation importance (outputs/importance_rf.csv)
  mi    mutual information (outputs/importance_mi.csv)
  mean  average of the RF rank and the MI rank
For k = 50, 60, ... up to the whole union set, the top k features of each
ranking are scored with the GP (aml.models.make_gp): 5 folds x 3 repeats,
training folds without the outlier rows, R² on the inlier validation rows.

Run on the HPC after scripts/feature_importance.py:
  uv run python scripts/selection_curves.py
Writes outputs/selection_curves.csv (one row per ranking, k, repeat, fold).
"""

import time
import warnings

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
from aml.features import beats_shadows
from aml.features import correlated_block
from aml.models import make_gp


N_REPEATS = 3
K_START, K_STEP = 50, 10


def score(ranking, k, repeat, fold, X, y, train, val):
    with warnings.catch_warnings():
        # The noise level ends at its lower bound; harmless (see gp_kernel)
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = make_gp(X.shape[1]).fit(X.iloc[train], y.iloc[train])
    r2 = r2_score(y.iloc[val], model.predict(X.iloc[val]))
    return {"ranking": ranking, "k": k, "repeat": repeat, "fold": fold, "r2": r2}


def main() -> None:
    X, y, X_test, inlier = load_clean_data()

    rf = pd.read_csv(OUTPUT_DIR / "importance_rf.csv", index_col=0)
    mi = pd.read_csv(OUTPUT_DIR / "importance_mi.csv", index_col=0)
    block = set(correlated_block(pd.concat([X, X_test])))
    union = beats_shadows(rf) | beats_shadows(mi) | block
    pool = [c for c in X.columns if c in union]
    print(
        f"Union set: {len(pool)} features (RF {len(beats_shadows(rf))}, "
        f"MI {len(beats_shadows(mi))}, block {len(block)})",
        flush=True,
    )

    rank_rf = rf["mean"].reindex(pool).rank(ascending=False)
    rank_mi = mi["mean"].reindex(pool).rank(ascending=False)
    orders = {
        "rf": rank_rf.sort_values().index.tolist(),
        "mi": rank_mi.sort_values().index.tolist(),
        "mean": ((rank_rf + rank_mi) / 2).sort_values().index.tolist(),
    }
    ks = [*range(K_START, len(pool), K_STEP), len(pool)]

    folds = list(
        RepeatedKFold(n_splits=CV_FOLDS, n_repeats=N_REPEATS, random_state=SEED).split(
            X
        )
    )
    tasks = [
        delayed(score)(
            name,
            k,
            i // CV_FOLDS + 1,
            i % CV_FOLDS + 1,
            X[order[:k]],
            y,
            train[inlier[train]],
            val[inlier[val]],
        )
        for name, order in orders.items()
        for k in ks
        for i, (train, val) in enumerate(folds)
    ]
    print(f"{len(ks)} values of k x 3 rankings x {len(folds)} folds", flush=True)

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

    curves = pd.DataFrame(rows).sort_values(["ranking", "k", "repeat", "fold"])
    path = OUTPUT_DIR / "selection_curves.csv"
    curves.to_csv(path, index=False)

    mean = curves.pivot_table(index="k", columns="ranking", values="r2")
    print(f"\nMean R² per k (wrote {path}):")
    print(mean.round(4).to_string())
    for name in orders:
        best = mean[name].idxmax()
        print(f"{name}: best k={best}, R² {mean.loc[best, name]:.4f}")


if __name__ == "__main__":
    main()
