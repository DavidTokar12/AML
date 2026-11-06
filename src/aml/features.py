"""Feature selection steps that work on raw data (NaNs allowed, no scaling)."""

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


def outlier_scores(X: pd.DataFrame, y: pd.Series, seed: int = 25) -> pd.Series:
    """Row outlier score, highest (most likely outlier) first.

    Median-impute and scale (only for this step), project to 2D with PLS on y,
    then score each row with an IsolationForest.
    """
    X_scaled = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    ).fit_transform(X)
    z = PLSRegression(n_components=2).fit(X_scaled, y).transform(X_scaled)
    iso = IsolationForest(random_state=seed, n_jobs=-1).fit(z)
    scores = pd.Series(-iso.score_samples(z), index=X.index, name="outlier_score")
    return scores.sort_values(ascending=False)


def rf_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    n_shadow: int = 20,
    n_estimators: int = 1000,
    seed: int = 25,
) -> pd.DataFrame:
    """Out-of-fold permutation importance (drop in R²) of a random forest on raw X.

    NaNs are handled by the forest itself. Adds `n_shadow` shuffled copies of
    random features ("shadow_*") as a noise baseline for picking a threshold.
    Returns mean and std over folds, most important first.
    """

    rng = np.random.default_rng(seed)
    
    shadows = pd.DataFrame(
        {
            f"shadow_{c}": rng.permutation(X[c].to_numpy())
            for c in rng.choice(X.columns, n_shadow, replace=False)
        },
        index=X.index,
    )
    
    X = pd.concat([X, shadows], axis=1)

    folds = []
    for train, val in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ).fit(X.iloc[train], y.iloc[train])
        rf.set_params(n_jobs=1)  # permutation_importance parallelizes instead
        result = permutation_importance(
            rf,
            X.iloc[val],
            y.iloc[val],
            scoring="r2",
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=-1,
        )
        folds.append(result.importances_mean)

    imp = pd.DataFrame(np.array(folds).T, index=X.columns)
    return pd.DataFrame({"mean": imp.mean(axis=1), "std": imp.std(axis=1)}).sort_values(
        "mean", ascending=False
    )
