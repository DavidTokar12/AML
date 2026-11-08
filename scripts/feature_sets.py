"""Which feature set? TabPFN and the GP on six sets, the same 5 folds.

Sets (from TabPFN's permutation importance, outputs/tabpfn/importance*.csv):
  significant   importance > 0 with t-test p < 0.01 over the 10 splits (~76)
  consistent    beats the best shadow in >= 8 of the 10 splits (~95)
  block         the correlated block (185, the current set)
  block_plus    block + the outside-block features that are consistent (~187)
  above_shadow  mean importance > the best shadow's (~272)
  all           all 726 cleaned features
Models: TabPFN on the raw features (missing values, no transform); the GP on
quantile-transformed, then KNN-imputed features (3 distance-weighted
neighbours), fitted on train + test per set. The GP skips "all". Note: the
iterative imputer of the locked features is a bit better (GP R² 0.719 vs
0.716) but too slow on many features; it can be swapped in for the final set.
CV: the folds of scripts/cv.py, training folds without the outlier rows, R² on
the inlier validation rows.

Run on the HPC (GPU for TabPFN):
  uv run python scripts/feature_sets.py [--models tabpfn,gp] [--sets block,all,...]
Writes outputs/feature_sets/scores.csv and oof_<model>_<set>.csv.
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from joblib import Parallel
from joblib import delayed
from scipy.stats import ttest_1samp
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_clean_data
from aml.features import correlated_block
from aml.models import make_model
from aml.models import make_preprocessor


OUT = OUTPUT_DIR / "feature_sets"
SET_NAMES = ("significant", "consistent", "block", "block_plus", "above_shadow", "all")


def feature_sets(columns: list[str], block: set[str]) -> dict[str, list[str]]:
    imp = pd.read_csv(OUTPUT_DIR / "tabpfn" / "importance.csv", index_col=0)
    splits = pd.read_csv(OUTPUT_DIR / "tabpfn" / "importance_splits.csv")
    per_split = splits.pivot_table(
        index="feature", columns=["repeat", "fold"], values="mean"
    )
    is_shadow = per_split.index.str.startswith("shadow_")
    wins = per_split.gt(per_split[is_shadow].max(), axis=1).sum(axis=1)
    pvalue = splits.groupby("feature")["mean"].apply(
        lambda s: ttest_1samp(s, 0, alternative="greater").pvalue
    )
    best_shadow = imp.loc[imp.index.str.startswith("shadow_"), "mean"].max()
    consistent = set(wins.index[wins >= 8])
    chosen = {
        "significant": set(pvalue.index[pvalue < 0.01]),
        "consistent": consistent,
        "block": block,
        "block_plus": block | consistent,
        "above_shadow": set(imp.index[imp["mean"] > best_shadow]),
        "all": set(columns),
    }
    return {name: [c for c in columns if c in s] for name, s in chosen.items()}


def gp_features(features, X_raw, X_test_raw):
    """Training features of a set, quantile-transformed then KNN-imputed
    (fitted on train + test, no y)."""
    both = pd.concat([X_raw[features], X_test_raw[features]], ignore_index=True)
    Z = make_preprocessor(knn=True).fit_transform(both).iloc[: len(X_raw)]
    return Z.set_axis(X_raw.index)


def run_fold(model, X, y, train, val):
    with warnings.catch_warnings():
        # GP noise level at its lower bound: harmless (see aml.models.gp_kernel)
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = make_model(model, X.shape[1]).fit(X.iloc[train], y.iloc[train])
    return fitted.predict(X.iloc[val])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="tabpfn,gp")
    parser.add_argument("--sets", default=",".join(SET_NAMES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    X_raw, y, X_test_raw, inlier = load_clean_data()
    block = set(correlated_block(pd.concat([X_raw, X_test_raw], ignore_index=True)))
    sets = feature_sets(list(X_raw.columns), block)
    print({name: len(f) for name, f in sets.items()}, flush=True)
    folds = [
        (train[inlier[train]], val[inlier[val]])
        for train, val in RepeatedKFold(
            n_splits=CV_FOLDS, n_repeats=1, random_state=SEED
        ).split(X_raw)
    ]

    scores_path = OUT / "scores.csv"
    for model in args.models.split(","):
        for name in args.sets.split(","):
            features = sets[name]
            if model == "gp" and name == "all":
                print("=== gp on all: skipped (726 features) ===", flush=True)
                continue
            print(f"=== {model} on {name} ({len(features)} features) ===", flush=True)
            start = time.time()
            if model == "gp":
                X = gp_features(features, X_raw, X_test_raw)
                preds = Parallel(n_jobs=CV_FOLDS)(
                    delayed(run_fold)(model, X, y, tr, va) for tr, va in folds
                )
            else:  # TabPFN: raw features, one fold at a time (GPU)
                X = X_raw[features]
                preds = [run_fold(model, X, y, tr, va) for tr, va in folds]

            oof = pd.concat(
                pd.DataFrame(
                    {
                        "id": X.index[va].astype(int),
                        "fold": f,
                        "y": y.iloc[va].to_numpy(),
                        "pred": p,
                    }
                )
                for f, ((_, va), p) in enumerate(zip(folds, preds, strict=True), 1)
            )
            oof.to_csv(OUT / f"oof_{model}_{name}.csv", index=False)
            r2 = [r2_score(g["y"], g["pred"]) for _, g in oof.groupby("fold")]
            pd.DataFrame(
                {
                    "model": model,
                    "set": name,
                    "n_features": len(features),
                    "fold": range(1, len(r2) + 1),
                    "r2": r2,
                }
            ).to_csv(
                scores_path, mode="a", header=not scores_path.exists(), index=False
            )
            print(
                f"=== {model} on {name}: R² {np.mean(r2):.4f} ± {np.std(r2):.4f}, "
                f"{(time.time() - start) / 60:.1f} min ===",
                flush=True,
            )

    scores = pd.read_csv(scores_path).drop_duplicates(
        ["model", "set", "fold"], keep="last"
    )
    table = scores.pivot_table(
        index=["set", "n_features"], columns="model", values="r2"
    )
    print("\nMean R² over 5 folds:")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
