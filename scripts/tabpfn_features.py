"""Does TabPFN need our feature selection? Same model, two feature sets.

  block  the 185 features of the correlated block (what TabPFN uses now)
  all    all 726 cleaned features: only constant features, the exact copy and
         the uniform-noise features dropped, copy groups merged
Both raw (missing values, no transform), the same 5 folds as scripts/cv.py,
training folds without the outlier rows, R² on the inlier validation rows.

Run on the HPC (GPU): uv run python scripts/tabpfn_features.py [block|all ...]
Writes outputs/tabpfn/<feature set>.csv (id, fold, y, pred).
"""

import sys
import time

import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_clean_data
from aml.features import correlated_block
from aml.models import make_model


OUT = OUTPUT_DIR / "tabpfn"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X_all, y, X_test, inlier = load_clean_data()
    block = set(correlated_block(pd.concat([X_all, X_test], ignore_index=True)))
    feature_sets = {
        "block": [c for c in X_all.columns if c in block],
        "all": list(X_all.columns),
    }
    folds = list(
        RepeatedKFold(n_splits=CV_FOLDS, n_repeats=1, random_state=SEED).split(X_all)
    )

    for name in sys.argv[1:] or feature_sets:
        X = X_all[feature_sets[name]]
        print(f"=== {name}: {X.shape[1]} features ===", flush=True)
        rows, scores = [], []
        for fold, (train, val) in enumerate(folds, 1):
            train, val = train[inlier[train]], val[inlier[val]]
            start = time.time()
            model = make_model("tabpfn", X.shape[1]).fit(X.iloc[train], y.iloc[train])
            pred = model.predict(X.iloc[val])
            scores.append(r2_score(y.iloc[val], pred))
            rows.append(
                pd.DataFrame(
                    {
                        "id": X.index[val].astype(int),
                        "fold": fold,
                        "y": y.iloc[val].to_numpy(),
                        "pred": pred,
                    }
                )
            )
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
