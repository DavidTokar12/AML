"""5-fold CV of the final models (same configs and inputs as scripts/submit.py),
saving every out-of-fold prediction for error analysis and ensembling.

Folds: the first repeat of the folds used in all earlier runs. Training folds
without the outlier rows, R² on the inlier validation rows.

Run on the HPC after scripts/preprocess.py:
  uv run python scripts/cv.py [model ...]     (default: all)
Writes outputs/cv/<model>.csv (id, fold, y, pred; the GP also its predicted
std) and prints R² per fold.
"""

import sys
import time
import warnings

import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_features
from aml.models import MODEL_NAMES
from aml.models import RAW_INPUT
from aml.models import make_model


OUT = OUTPUT_DIR / "cv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or MODEL_NAMES:
        X, y, _, inlier = load_features(raw=name in RAW_INPUT)
        folds = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=1, random_state=SEED)
        print(f"=== {name} ===", flush=True)
        rows, scores = [], []
        for fold, (train, val) in enumerate(folds.split(X), 1):
            train, val = train[inlier[train]], val[inlier[val]]
            start = time.time()
            with warnings.catch_warnings():
                # GP noise level at its lower bound: harmless (see aml.models)
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = make_model(name, X.shape[1])
                model.fit(X.iloc[train], y.iloc[train])
            oof = pd.DataFrame({"id": X.index[val], "fold": fold, "y": y.iloc[val]})
            if name == "gp":  # its own uncertainty, for uncertainty-aware ensembling
                oof["pred"], oof["std"] = model.predict(X.iloc[val], return_std=True)
            else:
                oof["pred"] = model.predict(X.iloc[val])
            scores.append(r2_score(oof["y"], oof["pred"]))
            rows.append(oof)
            print(
                f"  fold {fold}: R² {scores[-1]:.4f}, "
                f"{(time.time() - start) / 60:.1f} min",
                flush=True,
            )
        pd.concat(rows).to_csv(OUT / f"{name}.csv", index=False)
        s = pd.Series(scores)
        print(f"=== {name}: R² {s.mean():.4f} ± {s.std():.4f} ===\n", flush=True)


if __name__ == "__main__":
    main()
