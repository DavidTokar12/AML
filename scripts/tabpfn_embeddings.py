"""Other models on TabPFN's embeddings: does anything beat TabPFN itself?

A training row's embedding is computed with that row's own target in TabPFN's
context, so downstream models must not see such embeddings (they would learn
the answer from the embedding). Nested CV instead:
  outer 5 folds (the usual ones): for each,
    - training rows get out-of-fold embeddings: 5 inner folds, each row embedded
      by a TabPFN fitted on the other inner folds
    - validation rows are embedded by a TabPFN fitted on the whole outer
      training set, which also gives TabPFN's own prediction (the reference)
    - downstream models are trained on the training embeddings, predict the
      validation embeddings
Embeddings: averaged over TabPFN's internal estimators. Features: the block
(185, raw), all training rows (no outlier removal, the notebook's best).
R² on the inlier validation rows.

Run on the HPC (GPU): uv run python scripts/tabpfn_embeddings.py
Writes outputs/embeddings/oof.csv (one prediction column per model).
"""

import time
import warnings

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tabpfn import TabPFNRegressor

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_clean_data
from aml.features import correlated_block
from aml.models import gp_kernel


OUT = OUTPUT_DIR / "embeddings"


def downstream(d: int) -> dict:
    """Models trained on d-dimensional embeddings (standardized first)."""
    scaled = lambda model: make_pipeline(StandardScaler(), model)
    return {
        "ridge": scaled(RidgeCV(alphas=np.logspace(-2, 4, 20))),
        "svr": scaled(SVR(kernel="rbf", C=30, epsilon=1, gamma="scale")),
        "gp": scaled(
            GaussianProcessRegressor(
                gp_kernel(d),
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=SEED,
            )
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            random_seed=SEED,
            verbose=0,
            allow_writing_files=False,
        ),
        "knn": scaled(KNeighborsRegressor(n_neighbors=20, weights="distance")),
    }


def fit_tabpfn(X, y):
    return TabPFNRegressor(device="auto", random_state=SEED).fit(X, y)


def embed(model, X):
    """Test-side embeddings, averaged over the internal estimators."""
    e = model.get_embeddings(X, data_source="test")
    return e.mean(axis=0) if e.ndim == 3 else e


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, X_test, inlier = load_clean_data()
    block = set(correlated_block(pd.concat([X, X_test], ignore_index=True)))
    X = X[[c for c in X.columns if c in block]]
    outer = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=1, random_state=SEED)

    rows = []
    for fold, (tr, va) in enumerate(outer.split(X), 1):
        start = time.time()
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        # out-of-fold embeddings of the training rows
        E_tr = None
        for itr, iva in KFold(CV_FOLDS, shuffle=True, random_state=SEED).split(X_tr):
            e = embed(fit_tabpfn(X_tr.iloc[itr], y_tr.iloc[itr]), X_tr.iloc[iva])
            E_tr = np.zeros((len(tr), e.shape[1])) if E_tr is None else E_tr
            E_tr[iva] = e
        # validation rows: TabPFN fitted on the whole outer training set
        full = fit_tabpfn(X_tr, y_tr)
        E_va = embed(full, X.iloc[va])
        pred = {"tabpfn": full.predict(X.iloc[va])}
        print(
            f"fold {fold}: embeddings {E_tr.shape[1]}-dim, "
            f"{(time.time() - start) / 60:.1f} min",
            flush=True,
        )

        for name, model in downstream(E_tr.shape[1]).items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                pred[name] = model.fit(E_tr, y_tr).predict(E_va)
        rows.append(
            pd.DataFrame(
                {
                    "id": X.index[va].astype(int),
                    "fold": fold,
                    "y": y.iloc[va].to_numpy(),
                    "inlier": inlier[va],
                    **pred,
                }
            )
        )
        keep = inlier[va]
        print(
            "  "
            + ", ".join(
                f"{m} {r2_score(y.iloc[va][keep], p[keep]):.4f}"
                for m, p in pred.items()
            ),
            flush=True,
        )

    oof = pd.concat(rows)
    oof.to_csv(OUT / "oof.csv", index=False)
    scored = oof[oof["inlier"]]
    models = [c for c in oof.columns if c not in ("id", "fold", "y", "inlier")]

    def fold_r2(pred: pd.Series) -> float:
        return float(
            np.mean(
                [r2_score(g["y"], pred[g.index]) for _, g in scored.groupby("fold")]
            )
        )

    scored = scored.reset_index(drop=True)
    summary = {}
    for m in models:
        summary[m] = fold_r2(scored[m])
        if m != "tabpfn":
            summary[f"tabpfn + {m} (50/50)"] = fold_r2(
                (scored["tabpfn"] + scored[m]) / 2
            )
    print("\nR² on inlier validation rows (mean over 5 folds):")
    print(pd.Series(summary).round(4).to_string())


if __name__ == "__main__":
    main()
