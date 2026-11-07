"""Feature cleaning (steps 1-6 of the notebook) and feature scoring methods."""

import time

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer


def spread_stats(X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature spread on non-null values, sorted by std (highest first)."""
    return pd.DataFrame(
        {
            "std": X.std(),
            "nunique": X.nunique(),
            "top_share": X.apply(lambda c: c.value_counts(normalize=True).iloc[0]),
            "missing": X.isna().mean(),
        }
    ).sort_values("std", ascending=False)


def constant_features(X: pd.DataFrame) -> list[str]:
    """Features with a single non-null value."""
    return X.columns[X.nunique() <= 1].tolist()


def duplicate_features(X: pd.DataFrame, threshold: float = 0.9999) -> list[str]:
    """Features perfectly correlated with an earlier column (one of each pair).

    Correlation is computed on the rows where both features are present.
    """
    corr = np.triu(X.corr().abs().to_numpy(), k=1)
    return X.columns[(corr >= threshold).any(axis=0)].tolist()


def robust_scale(X: pd.DataFrame) -> pd.DataFrame:
    """(x - median) / IQR per feature, so outliers don't set the scale."""
    iqr = X.quantile(0.75) - X.quantile(0.25)
    return (X - X.median()) / iqr.replace(0, np.nan)


def spearman(X: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation: rank each feature once, then Pearson on the ranks.

    pandas' exact version re-ranks every pair on the rows both have (~1 min
    here, far longer on the HPC); this differs by < 0.003 and gives the same
    copy groups and the same correlated block.
    """
    return X.rank().corr()


def copy_groups(corr: pd.DataFrame, threshold: float = 0.94) -> list[list[str]]:
    """Groups of features linked by |corr| >= threshold (noisy copies).

    corr: absolute correlation matrix, e.g. spearman(X).abs().
    Linking is transitive: if a~b and b~c, then a, b, c form one group.
    """
    n, labels = connected_components(
        csr_matrix(corr.to_numpy() >= threshold), directed=False
    )
    groups = [corr.columns[labels == g].tolist() for g in range(n)]
    return [g for g in groups if len(g) > 1]


def aligned_copies(X: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Group members on a common scale (robust_scale), sign-aligned to the first."""
    sign = np.sign(X[group].corr(method="spearman")[group[0]])
    return robust_scale(X[group]) * sign


def copy_noise(X: pd.DataFrame, group: list[str]) -> pd.Series:
    """Each member's robust spread around the group's row-wise median, lowest first."""
    W = aligned_copies(X, group)
    deviation = W.sub(W.median(axis=1), axis=0)
    return (deviation.abs().median() * 1.4826).sort_values()


def merge_copies(X: pd.DataFrame, groups: list[list[str]]) -> pd.DataFrame:
    """Replace each group by its least noisy member, with its missing values
    filled from the next least noisy member that has one, and so on.

    The merged feature is on the robust scale and keeps the least noisy
    member's name.
    """
    merged = {}
    for g in groups:
        order = copy_noise(X, g).index
        merged[order[0]] = aligned_copies(X, g)[order].bfill(axis=1).iloc[:, 0]
    out = X.drop(columns=[c for g in groups for c in g])
    return pd.concat([out, pd.DataFrame(merged)], axis=1)


def outlier_scores(X: pd.DataFrame, y: pd.Series, seed: int = 25) -> pd.Series:
    """Row outlier score, highest (most likely outlier) first.

    Quantile-transform to normal (so extreme values can't dominate) and
    median-impute (only for this step), project to 2D with PLS on y, then score
    each row with an IsolationForest.
    """
    X_scaled = make_pipeline(
        QuantileTransformer(output_distribution="normal", random_state=seed),
        SimpleImputer(strategy="median"),
    ).fit_transform(X)
    z = PLSRegression(n_components=2).fit(X_scaled, y).transform(X_scaled)
    iso = IsolationForest(random_state=seed, n_jobs=-1).fit(z)
    scores = pd.Series(-iso.score_samples(z), index=X.index, name="outlier_score")
    return scores.sort_values(ascending=False)


def add_shadows(X: pd.DataFrame, n_shadow: int = 20, seed: int = 25) -> pd.DataFrame:
    """Add `n_shadow` shuffled copies of random features ("shadow_*").

    A shuffled feature has no relation to y, so the best shadow's score is the
    noise level a real feature has to beat.
    """
    rng = np.random.default_rng(seed)
    shadows = pd.DataFrame(
        {
            f"shadow_{c}": rng.permutation(X[c].to_numpy())
            for c in rng.choice(X.columns, n_shadow, replace=False)
        },
        index=X.index,
    )
    return pd.concat([X, shadows], axis=1)


def beats_shadows(imp: pd.DataFrame) -> set[str]:
    """Features whose importance is above the best shadow feature's."""
    is_shadow = imp.index.str.startswith("shadow_")
    best_shadow = imp.loc[is_shadow, "mean"].max()
    return set(imp.index[(imp["mean"] > best_shadow) & ~is_shadow])


def _summary(scores: np.ndarray, columns: pd.Index) -> pd.DataFrame:
    """Mean and std over the rows of `scores` (folds or seeds), best first."""
    return pd.DataFrame(
        {"mean": scores.mean(axis=0), "std": scores.std(axis=0)}, index=columns
    ).sort_values("mean", ascending=False)


def permutation_importance_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 25,
) -> pd.DataFrame:
    """Out-of-fold permutation importance (drop in validation R²).

    Fit on each training fold, shuffle one feature at a time in the validation
    fold. Returns mean and std over folds, most important first.
    """
    folds = []
    start = time.time()
    splits = KFold(n_splits, shuffle=True, random_state=seed).split(X)
    for i, (train, val) in enumerate(splits, 1):
        print(f"  [fold {i}/{n_splits}] fitting...", flush=True)
        fitted = clone(model).fit(X.iloc[train], y.iloc[train])
        if "n_jobs" in fitted.get_params(deep=False):
            fitted.set_params(n_jobs=1)  # permutation_importance parallelizes
        result = permutation_importance(
            fitted,
            X.iloc[val],
            y.iloc[val],
            scoring="r2",
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=-1,
        )
        folds.append(result.importances_mean)
        elapsed = time.time() - start
        print(
            f"  [fold {i}/{n_splits}] done, {elapsed / 60:.1f} min elapsed, "
            f"~{elapsed / i * (n_splits - i) / 60:.1f} min left",
            flush=True,
        )
    return _summary(np.array(folds), X.columns)


def mutual_information(
    X: pd.DataFrame, y: pd.Series, n_seeds: int = 5, seed: int = 25
) -> pd.DataFrame:
    """Mutual information of each feature with y, on the rows where it is present.

    Quantile-transformed first: MI itself ignores monotone transforms, but the
    nearest-neighbour estimator is thrown off by extreme values. Averaged over
    `n_seeds` runs of the (noisy) estimator.
    """
    Z = (
        QuantileTransformer(output_distribution="normal", random_state=seed)
        .set_output(transform="pandas")
        .fit_transform(X)
    )
    scores = np.zeros((n_seeds, X.shape[1]))
    for j, col in enumerate(Z.columns):
        present = Z[col].notna().to_numpy()
        for s in range(n_seeds):
            scores[s, j] = mutual_info_regression(
                Z.loc[present, [col]], y[present], random_state=seed + s
            )[0]
    return _summary(scores, X.columns)


def correlated_block(X: pd.DataFrame, threshold: float = 0.3) -> list[str]:
    """Features with |Spearman| >= threshold to at least one other feature.

    In this data the other features correlate with nothing and carry no signal.
    """
    corr = spearman(X).abs().to_numpy().copy()
    np.fill_diagonal(corr, 0)
    return X.columns[corr.max(axis=0) >= threshold].tolist()


def uniform_noise_features(X: pd.DataFrame) -> list[str]:
    """Features whose values span exactly 0.1: a constant plus uniform noise."""
    value_range = X.max() - X.min()
    return value_range.index[value_range.between(0.09, 0.11)].tolist()


def clean_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, copy_threshold: float = 0.94
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The notebook's steps 1-6: drop constant features, exact copies and
    uniform noise, then merge groups of noisy copies (fit on train + test).
    """
    drop = constant_features(X_train)
    drop += duplicate_features(X_train.drop(columns=drop))
    drop += uniform_noise_features(X_train.drop(columns=drop))
    X_all = pd.concat(
        [X_train.drop(columns=drop), X_test.drop(columns=drop)], ignore_index=True
    )
    groups = copy_groups(spearman(X_all).abs(), copy_threshold)
    merged = merge_copies(X_all, groups)
    n = len(X_train)
    return (
        merged.iloc[:n].set_index(X_train.index),
        merged.iloc[n:].set_index(X_test.index),
    )
