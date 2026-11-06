"""Compute RF permutation importances on raw data (slow: meant for the HPC)."""

from pathlib import Path

from aml.config import CONFIG
from aml.data import load_data
from aml.features import constant_features
from aml.features import rf_permutation_importance


def main() -> None:
    X, y, _ = load_data(CONFIG.data_dir)
    X = X.drop(columns=constant_features(X))
    imp = rf_permutation_importance(X, y, seed=CONFIG.seed)
    out = Path(CONFIG.output_dir) / "importance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
