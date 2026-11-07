"""Permutation importance of TabPFN on all 726 cleaned features.

Importance of a feature = drop in validation R² when that feature is shuffled.
Robust version:
  - out of fold, over 5 folds x 2 repeats (10 different train/validation splits)
  - 10 independent shuffles per feature and split, averaged
  - 20 shadow features (shuffled copies of real features, also seen in training):
    the best shadow is the noise level a real feature has to beat
Speed: TabPFN caches the training set (fit_mode="fit_with_cache") and the 10
shuffled copies of the validation rows go through one predict call.
Raw features (missing values, no transform), training folds without the
outlier rows, R² on the inlier validation rows.

Run on the HPC (GPU): uv run python scripts/tabpfn_importance.py
Each finished split is appended to outputs/tabpfn/importance_splits.csv, so a
cut-off run continues where it stopped. At the end writes
outputs/tabpfn/importance.csv: mean and std over splits, most important first.
"""

import time

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from tabpfn import TabPFNRegressor

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_clean_data
from aml.features import add_shadows


N_SPLIT_REPEATS = 2  # x 5 folds = 10 splits
N_SHUFFLES = 10
OUT = OUTPUT_DIR / "tabpfn"
SPLITS_FILE = OUT / "importance_splits.csv"


def split_importance(X, y, train, val, seed):
    """Mean and std of the R² drop per feature, over N_SHUFFLES shuffles."""
    model = TabPFNRegressor(
        device="auto", random_state=SEED, fit_mode="fit_with_cache"
    ).fit(X.iloc[train], y.iloc[train])
    X_val, y_val = X.iloc[val], y.iloc[val].to_numpy()
    base = r2_score(y_val, model.predict(X_val))
    rng = np.random.default_rng(seed)
    n = len(val)
    stacked = pd.concat([X_val] * N_SHUFFLES, ignore_index=True)
    rows = []
    for j, col in enumerate(X.columns):
        shuffled = stacked.copy()
        shuffled[col] = np.concatenate(
            [rng.permutation(X_val[col].to_numpy()) for _ in range(N_SHUFFLES)]
        )
        pred = model.predict(shuffled).reshape(N_SHUFFLES, n)
        drops = base - np.array([r2_score(y_val, p) for p in pred])
        rows.append({"feature": col, "mean": drops.mean(), "std": drops.std()})
        if (j + 1) % 100 == 0:
            print(f"    {j + 1}/{X.shape[1]} features", flush=True)
    return base, pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, _, inlier = load_clean_data()
    X = add_shadows(X, seed=SEED)
    print(f"{X.shape[1]} features (incl. 20 shadows), {inlier.sum()} rows", flush=True)

    done = set()
    if SPLITS_FILE.exists():
        prev = pd.read_csv(SPLITS_FILE)
        done = set(zip(prev["repeat"], prev["fold"], strict=True))
        print(f"Resuming: {len(done)} splits already done", flush=True)

    folds = RepeatedKFold(
        n_splits=CV_FOLDS, n_repeats=N_SPLIT_REPEATS, random_state=SEED
    )
    for i, (train, val) in enumerate(folds.split(X)):
        repeat, fold = i // CV_FOLDS + 1, i % CV_FOLDS + 1
        if (repeat, fold) in done:
            continue
        start = time.time()
        print(f"=== split r{repeat} f{fold} ===", flush=True)
        base, imp = split_importance(
            X, y, train[inlier[train]], val[inlier[val]], seed=SEED + i
        )
        imp.assign(repeat=repeat, fold=fold, base_r2=base).to_csv(
            SPLITS_FILE, mode="a", header=not SPLITS_FILE.exists(), index=False
        )
        print(
            f"=== r{repeat} f{fold}: base R² {base:.4f}, "
            f"{(time.time() - start) / 60:.1f} min ===",
            flush=True,
        )

    splits = pd.read_csv(SPLITS_FILE)
    imp = (
        splits.groupby("feature")["mean"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    imp.to_csv(OUT / "importance.csv")
    is_shadow = imp.index.str.startswith("shadow_")
    best_shadow = imp.loc[is_shadow, "mean"].max()
    n_beat = (imp.loc[~is_shadow, "mean"] > best_shadow).sum()
    print(f"\n{n_beat} of {(~is_shadow).sum()} features beat the best shadow")
    print(imp.head(30).round(5).to_string())


if __name__ == "__main__":
    main()
