"""Throwaway: which preprocessing helps each model?

For each model, decide greedily, keeping each winner for the next step:
  outlier removal yes/no -> quantile transform yes/no -> KNN imputation yes/no
(the random forest only gets the outlier step: it ignores scaling and takes NaNs).
Each setting gets its own best number of top features k in [K_MIN, K_MAX].

All settings are scored on the non-outlier rows only (the test set looks
outlier-free), so they are compared on the same rows and the same folds.

Run on the HPC: uv run python scripts/preprocessing_experiments.py
Writes outputs/preprocessing_experiments.csv.
"""

import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_data
from aml.evaluate import cv_scores
from aml.evaluate import search_k
from aml.features import add_shadows
from aml.features import constant_features
from aml.features import outlier_scores
from aml.features import permutation_importance_cv
from aml.models import MODELS
from aml.models import make_model


OUTLIER_FRAC = 0.045  # last year's value: the top ~55 outlier scores
K_MIN, K_MAX = 50, 300
STEPS = ("outliers", "quantile", "knn")


def ranking(X: pd.DataFrame, y: pd.Series, name: str) -> list[str]:
    """Features by RF permutation importance, cached in outputs/ (slow)."""
    path = OUTPUT_DIR / f"importance_{name}.csv"
    if path.exists():
        print(f"Using cached {path}", flush=True)
        imp = pd.read_csv(path, index_col=0)
    else:
        print(f"Computing importance ({name} rows)...", flush=True)
        rf = RandomForestRegressor(
            n_estimators=1000, max_features="sqrt", random_state=SEED, n_jobs=-1
        )
        imp = permutation_importance_cv(rf, add_shadows(X, seed=SEED), y, seed=SEED)
        imp.to_csv(path)
    return [c for c in imp.index if not c.startswith("shadow_")]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    X, y, _ = load_data()
    X = X.drop(columns=constant_features(X))

    n_out = round(OUTLIER_FRAC * len(X))
    inlier = ~X.index.isin(outlier_scores(X, y, seed=SEED).index[:n_out])
    print(f"Flagged {n_out} of {len(X)} rows as outliers", flush=True)

    rankings = {
        False: ranking(X, y, "all"),
        True: ranking(X[inlier], y[inlier], "clean"),
    }

    rows = []
    best = {}  # settings -> (best k, fold scores at that k)

    def run(model: str, outliers: bool, quantile: bool, knn: bool):
        key = (model, outliers, quantile, knn)
        if key in best:
            return best[key]
        label = f"{model} outliers={outliers} quantile={quantile} knn={knn}"
        print(f"=== {label} ===", flush=True)
        train_mask = inlier if outliers else np.ones(len(X), dtype=bool)
        top = rankings[outliers]

        def score(k: int) -> np.ndarray:
            start = time.time()
            s = cv_scores(
                make_model(model, k, quantile, knn),
                X[top[:k]],
                y,
                train_mask,
                inlier,
                n_jobs=1 if model == "rf" else CV_FOLDS,
            )
            rows.append(
                {
                    "model": model,
                    "outliers": outliers,
                    "quantile": quantile,
                    "knn": knn,
                    "k": k,
                    "r2_mean": s.mean(),
                    "r2_std": s.std(),
                }
            )
            print(
                f"[{label}] k={k:3d}  R² {s.mean():.4f} ± {s.std():.4f}  "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )
            return s

        results = search_k(score, K_MIN, K_MAX)
        k = max(results, key=lambda k: results[k].mean())
        best[key] = (k, results[k])
        print(f"[{label}] best k={k}  R² {results[k].mean():.4f}\n", flush=True)
        return best[key]

    decisions = []
    for model in MODELS:
        settings = dict.fromkeys(STEPS, False)
        for step in STEPS[:1] if model == "rf" else STEPS:
            k0, s0 = run(model, **settings)
            k1, s1 = run(model, **{**settings, step: True})
            diff = s1 - s0  # same folds, so compare fold by fold
            settings[step] = bool(diff.mean() > 0)
            print(
                f">>> [{model}] {step}: {'YES' if settings[step] else 'NO'}  "
                f"R² {s0.mean():.4f} (k={k0}) -> {s1.mean():.4f} (k={k1}), "
                f"better in {(diff > 0).sum()}/{len(diff)} folds\n",
                flush=True,
            )
        k, s = run(model, **settings)
        decisions.append({"model": model, **settings, "k": k, "r2": s.mean()})

    out = OUTPUT_DIR / "preprocessing_experiments.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}\n")
    print("Final choice per model:")
    print(pd.DataFrame(decisions).to_string(index=False))


if __name__ == "__main__":
    main()
