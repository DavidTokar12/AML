"""Fit the pipeline from config.py on all training data and write a submission."""

from pathlib import Path

from aml.config import CONFIG
from aml.data import load_data
from aml.data import save_submission
from aml.pipeline import build_pipeline
from aml.pipeline import remove_outliers


def main() -> None:
    X_train, y_train, X_test = load_data(CONFIG.data_dir)
    X_train, y_train = remove_outliers(CONFIG, X_train, y_train)
    model = build_pipeline(CONFIG).fit(X_train, y_train)
    out = Path(CONFIG.output_dir) / "submission.csv"
    save_submission(X_test.index, model.predict(X_test), out)


if __name__ == "__main__":
    main()
