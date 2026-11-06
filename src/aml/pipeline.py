"""Outlier removal, preprocessing + feature selection pipeline, and CV evaluation."""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import r_regression
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import validate_data

from aml.config import Config


def make_imputer(cfg: Config):
    match cfg.imputer:
        case "median" | "mean":
            return SimpleImputer(strategy=cfg.imputer)
        case "knn":
            return KNNImputer()
    raise ValueError(f"Unknown imputer: {cfg.imputer}")


def make_scaler(cfg: Config):
    match cfg.scaler:
        case "standard":
            return StandardScaler()
        case "robust":
            return RobustScaler()
    raise ValueError(f"Unknown scaler: {cfg.scaler}")


def make_model(cfg: Config):
    match cfg.model:
        case "svr":
            return SVR(kernel="rbf", C=cfg.svr_c, gamma="scale")
        case "extra_trees":
            return ExtraTreesRegressor(
                n_estimators=1000, random_state=cfg.seed, n_jobs=-1
            )
        case "gbr":
            return GradientBoostingRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                random_state=cfg.seed,
            )
    raise ValueError(f"Unknown model: {cfg.model}")


class DropCorrelated(SelectorMixin, BaseEstimator):
    """Drop features with |corr| >= threshold to an already kept feature.

    Features are visited in order of |corr| with the target, so the most
    predictive one of each correlated group is kept.
    """

    def __init__(self, threshold: float = 0.99):
        self.threshold = threshold

    def fit(self, X, y):
        X = validate_data(self, X)
        order = np.argsort(-np.abs(r_regression(X, y)))
        corr = np.abs(np.corrcoef(X, rowvar=False))
        kept = []
        for j in order:
            if not kept or corr[j, kept].max() < self.threshold:
                kept.append(j)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[kept] = True
        return self

    def _get_support_mask(self):
        return self.support_


def build_pipeline(cfg: Config) -> Pipeline:
    rf = RandomForestRegressor(
        n_estimators=cfg.rf_n_estimators,
        max_features="sqrt",
        random_state=cfg.seed,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("impute", make_imputer(cfg)),
            ("scale", make_scaler(cfg)),
            ("low_var", VarianceThreshold(cfg.low_var_threshold)),
            ("top_k_corr", SelectKBest(f_regression, k=cfg.top_k_corr)),
            ("decorrelate", DropCorrelated(cfg.corr_feature_max)),
            (
                "rf_select",
                SelectFromModel(
                    rf,
                    threshold=-np.inf,
                    max_features=lambda X: min(cfg.rf_keep, X.shape[1]),
                ),
            ),
            ("model", make_model(cfg)),
        ]
    )


def remove_outliers(
    cfg: Config, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop samples flagged by an IsolationForest on a 2D PLS embedding of X."""
    if not cfg.contamination:
        return X, y
    X_scaled = make_pipeline(make_imputer(cfg), make_scaler(cfg)).fit_transform(X)
    z = PLSRegression(n_components=2).fit(X_scaled, y).transform(X_scaled)
    iso = IsolationForest(
        contamination=cfg.contamination, random_state=cfg.seed, n_jobs=-1
    )
    inlier = iso.fit_predict(z) == 1
    print(f"Outliers: removed {(~inlier).sum()} of {len(X)} samples")
    return X.loc[inlier], y.loc[inlier]


def cross_validate(cfg: Config, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """R² per fold. Outliers are removed first; everything else is fit per fold."""
    X, y = remove_outliers(cfg, X, y)
    cv = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    return cross_val_score(build_pipeline(cfg), X, y, cv=cv, scoring="r2")
