"""Settings used by `train` and as the baseline for `tune`.

Defaults are the best values from last year's grid search.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 25

    # Preprocessing
    imputer: str = "median"  # "median" | "mean" | "knn"
    scaler: str = "standard"  # "standard" | "robust"

    # Outlier removal on the training set (None disables it)
    contamination: float | None = 0.045

    # Feature selection
    low_var_threshold: float = 0.1
    top_k_corr: int = 200  # keep the k features most correlated with y
    corr_feature_max: float = 0.99  # drop near-duplicate features above this |r|
    rf_keep: int = 167  # keep the k most important random forest features
    rf_n_estimators: int = 1000

    # Model
    model: str = "svr"  # "svr" | "extra_trees" | "gbr"
    svr_c: float = 52

    # Evaluation
    cv_folds: int = 10

    # Paths
    data_dir: str | None = None  # None downloads the data via kagglehub
    output_dir: str = "outputs"


CONFIG = Config()
