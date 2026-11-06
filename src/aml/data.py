from pathlib import Path

import kagglehub
import pandas as pd

from dotenv import load_dotenv


COMPETITION = "eth-aml-2026-project-1"


def load_data(
    data_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X_train, y_train, X_test (indexed by id).

    Downloads the competition data with kagglehub (cached) unless `data_dir` is given.
    """
    if data_dir is None:
        load_dotenv()  # KAGGLE_API_TOKEN
        data_dir = kagglehub.competition_download(COMPETITION)
    path = Path(data_dir)
    X_train = pd.read_csv(path / "X_train.csv", index_col="id")
    y_train = pd.read_csv(path / "y_train.csv", index_col="id")["y"]
    X_test = pd.read_csv(path / "X_test.csv", index_col="id")
    return X_train, y_train, X_test


def save_submission(ids: pd.Index, y_pred, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "y": y_pred}).to_csv(path, index=False)
    print(f"Wrote {path}")
