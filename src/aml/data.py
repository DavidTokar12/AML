from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from aml.config import FEATURES_DIR
from aml.config import OUTLIER_FRAC
from aml.config import SEED
from aml.features import clean_features
from aml.features import outlier_scores


COMPETITION = "eth-aml-2026-project-1"


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X_train, y_train, X_test (indexed by id), downloaded via kagglehub."""
    load_dotenv()  # KAGGLE_API_TOKEN
    path = Path(kagglehub.competition_download(COMPETITION))
    X_train = pd.read_csv(path / "X_train.csv", index_col="id")
    y_train = pd.read_csv(path / "y_train.csv", index_col="id")["y"]
    X_test = pd.read_csv(path / "X_test.csv", index_col="id")
    return X_train, y_train, X_test


def load_clean_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    """Cleaned features (notebook steps 1-6) and the inlier mask of the
    training rows: X_train, y_train, X_test, inlier.
    """
    print("Loading data...", flush=True)
    X_train, y, X_test = load_data()
    print("Cleaning...", flush=True)
    X, X_test = clean_features(X_train, X_test)
    print("Flagging outlier rows...", flush=True)
    n_out = round(OUTLIER_FRAC * len(X))
    inlier = ~X.index.isin(outlier_scores(X, y, seed=SEED).index[:n_out])
    return X, y, X_test, inlier


def load_features() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    """The locked, preprocessed features written by scripts/preprocess.py:
    X_train, y_train, X_test, inlier.
    """
    train = pd.read_csv(FEATURES_DIR / "train.csv", index_col="id")
    X_test = pd.read_csv(FEATURES_DIR / "test.csv", index_col="id")
    y, inlier = train.pop("y"), train.pop("inlier").to_numpy()
    return train, y, X_test, inlier
