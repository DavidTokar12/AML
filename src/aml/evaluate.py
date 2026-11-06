"""Cross-validation and the search for the best number of features."""

import math

from collections.abc import Callable

import numpy as np
import pandas as pd

from joblib import Parallel
from joblib import delayed
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from aml.config import CV_FOLDS
from aml.config import SEED


def cv_scores(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    score_mask: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """R² per fold: fit on training-fold rows in train_mask, score on
    validation-fold rows in score_mask. Folds are the same for every call.
    """

    def fold(train, val):
        train, val = train[train_mask[train]], val[score_mask[val]]
        fitted = clone(model).fit(X.iloc[train], y.iloc[train])
        return r2_score(y.iloc[val], fitted.predict(X.iloc[val]))

    cv = KFold(CV_FOLDS, shuffle=True, random_state=SEED)
    return np.array(
        Parallel(n_jobs=n_jobs)(delayed(fold)(tr, va) for tr, va in cv.split(X))
    )


def search_k(
    score: Callable[[int], np.ndarray],
    lo: int,
    hi: int,
    n_coarse: int = 6,
    tol: int = 5,
) -> dict[int, np.ndarray]:
    """Find the k in [lo, hi] with the best mean score.

    Coarse grid first, then golden-section search around the best grid point.
    Returns every evaluated k with its fold scores.
    """
    cache = {}

    def f(k: int) -> float:
        if k not in cache:
            cache[k] = score(k)
        return cache[k].mean()

    grid = np.linspace(lo, hi, n_coarse).round().astype(int).tolist()
    best = int(np.argmax([f(k) for k in grid]))
    a, b = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]

    inv_phi = (math.sqrt(5) - 1) / 2
    while b - a > tol:
        c = round(b - inv_phi * (b - a))
        d = round(a + inv_phi * (b - a))
        if f(c) >= f(d):
            b = d
        else:
            a = c
    return cache
