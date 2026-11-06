from pathlib import Path

import kagglehub
import pandas as pd

from dotenv import load_dotenv


COMPETITION = "eth-aml-2026-project-1"


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X_train, y_train, X_test (indexed by id), downloaded via kagglehub."""
    load_dotenv()  # KAGGLE_API_TOKEN
    path = Path(kagglehub.competition_download(COMPETITION))
    X_train = pd.read_csv(path / "X_train.csv", index_col="id")
    y_train = pd.read_csv(path / "y_train.csv", index_col="id")["y"]
    X_test = pd.read_csv(path / "X_test.csv", index_col="id")
    return X_train, y_train, X_test
