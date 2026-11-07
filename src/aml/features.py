"""Feature cleaning (steps 1-6 of the notebook) and feature scoring methods."""

import time

import numpy as np
import pandas as pd

from joblib import Parallel
from joblib import delayed
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
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


def copy_groups(corr: pd.DataFrame, threshold: float = 0.94) -> list[list[str]]:
    """Groups of features linked by |corr| >= threshold (noisy copies).

    corr: absolute correlation matrix, e.g. X.corr(method="spearman").abs().
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


def quantile_knn(X: pd.DataFrame, seed: int = 25) -> pd.DataFrame:
    """Quantile-transform to normal, then KNN-impute (no y involved)."""
    Z = make_pipeline(
        QuantileTransformer(output_distribution="normal", random_state=seed),
        KNNImputer(),
    ).fit_transform(X)
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


def lasso_stability(
    X: pd.DataFrame,
    y: pd.Series,
    n_subsamples: int = 100,
    seed: int = 25,
) -> pd.DataFrame:
    """Stability selection: how often Lasso keeps each feature.

    Alpha is picked once by LassoCV on all rows, then Lasso is refit on
    `n_subsamples` random halves of the rows. Score = share of fits in which
    the coefficient is non-zero. Quantile transform + KNN imputation first.
    """
    Z = quantile_knn(X, seed)
    alpha = LassoCV(cv=5, random_state=seed, n_jobs=-1).fit(Z, y).alpha_
    print(f"  Lasso alpha {alpha:.4g}", flush=True)
    rng = np.random.default_rng(seed)
    selected = []
    for _ in range(n_subsamples):
        rows = rng.choice(len(Z), len(Z) // 2, replace=False)
        coef = Lasso(alpha=alpha, max_iter=10_000).fit(Z.iloc[rows], y.iloc[rows]).coef_
        selected.append(coef != 0)
    return _summary(np.array(selected, dtype=float), X.columns)


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
    corr = X.corr(method="spearman").abs().to_numpy().copy()
    np.fill_diagonal(corr, 0)
    return X.columns[corr.max(axis=0) >= threshold].tolist()


def _verbose_lbfgs(label: str, max_iter: int):
    """GP optimizer: sklearn's L-BFGS-B, printing progress every 25 iterations."""

    def optimizer(obj_func, initial_theta, bounds):
        state = {"iter": 0, "value": np.nan, "start": time.time()}

        def func(theta):
            state["value"], grad = obj_func(theta)
            return state["value"], grad

        def callback(_):
            state["iter"] += 1
            if state["iter"] % 25 == 0:
                print(
                    f"  [{label}] iter {state['iter']}, "
                    f"-log marginal likelihood {state['value']:.2f}, "
                    f"{(time.time() - state['start']) / 60:.1f} min",
                    flush=True,
                )

        result = minimize(
            func,
            initial_theta,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=callback,
            options={"maxiter": max_iter},
        )
        return result.x, result.fun

    return optimizer


def gp_ard_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    max_iter: int = 2000,
    seed: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GP with one RBF length scale per feature (ARD), two importances per feature.

    - permutation: drop in validation R² when the feature is shuffled. Measures
      how much the predictions depend on the feature.
    - length scale: 1 / length scale. Measures how fast the prediction changes
      along the feature, so a big smooth effect can score lower than a small
      sharp one.

    Quantile transform + KNN imputation first, so all features share a scale.
    One fit per training fold (folds in parallel), averaged over folds, so one
    optimizer run's local optimum doesn't decide the order. Also prints each
    fold's validation R².
    """
    Z = quantile_knn(X, seed)
    d = Z.shape[1]

    def fit_fold(i, train, val):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            np.full(d, np.sqrt(d)), (1e-2, 1e5)
        ) + WhiteKernel(0.3, (1e-3, 1.0))
        gp = GaussianProcessRegressor(
            kernel,
            normalize_y=True,
            optimizer=_verbose_lbfgs(f"fold {i}", max_iter),
            random_state=seed,
        ).fit(Z.iloc[train], y.iloc[train])
        r2 = gp.score(Z.iloc[val], y.iloc[val])
        print(f"  [fold {i}] fitted, validation R² {r2:.4f}, permuting...", flush=True)
        perm = permutation_importance(
            gp,
            Z.iloc[val],
            y.iloc[val],
            scoring="r2",
            n_repeats=n_repeats,
            random_state=seed,
        ).importances_mean
        print(f"  [fold {i}] done", flush=True)
        return perm, 1 / gp.kernel_.k1.k2.length_scale, r2

    splits = KFold(n_splits, shuffle=True, random_state=seed).split(Z)
    results = Parallel(n_jobs=n_splits)(
        delayed(fit_fold)(i, train, val) for i, (train, val) in enumerate(splits, 1)
    )
    perm, inv_length_scale, r2 = (np.array(r) for r in zip(*results, strict=True))
    print(f"  ARD GP out-of-fold R² {r2.mean():.4f} ± {r2.std():.4f}", flush=True)
    return _summary(perm, X.columns), _summary(inv_length_scale, X.columns)


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
    groups = copy_groups(X_all.corr(method="spearman").abs(), copy_threshold)
    merged = merge_copies(X_all, groups)
    n = len(X_train)
    return (
        merged.iloc[:n].set_index(X_train.index),
        merged.iloc[n:].set_index(X_test.index),
    )
