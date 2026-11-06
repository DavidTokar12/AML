"""For each model, find how many top-ranked features (by RF importance) to keep.

Coarse grid over [K_MIN, K_MAX] to find roughly where the best k is, then
golden-section search around it. All k use the same CV folds. Slow: meant for the HPC.
"""

import math
import time

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from aml.config import CONFIG
from aml.data import load_data


K_MIN, K_MAX = 50, 300
N_COARSE = 6  # coarse grid points (50, 100, ..., 300)
K_TOL = 5  # stop when the search interval is this narrow
CV_FOLDS = 5


def make_rf(k: int):
    # Handles NaNs natively: no imputation or scaling.
    return RandomForestRegressor(
        n_estimators=500, max_features=1 / 3, random_state=CONFIG.seed, n_jobs=-1
    )


def make_gp(k: int):
    kernel = ConstantKernel() * RBF(length_scale=math.sqrt(k)) + WhiteKernel()
    gp = GaussianProcessRegressor(
        kernel, normalize_y=True, n_restarts_optimizer=2, random_state=CONFIG.seed
    )
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), gp)


def make_svr(k: int):
    svr = SVR(kernel="rbf", C=50, gamma="scale")
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), svr)


# name -> (model factory, n_jobs for the CV folds)
MODELS = {
    "rf": (make_rf, 1),  # the forest itself already uses all cores
    "gp": (make_gp, CV_FOLDS),
    "svr": (make_svr, CV_FOLDS),
}


def search_k(score, lo: int, hi: int) -> dict[int, float]:
    """Coarse grid, then golden-section search around the best grid point."""
    cache = {}

    def f(k: int) -> float:
        if k not in cache:
            cache[k] = score(k)
        return cache[k]

    grid = np.linspace(lo, hi, N_COARSE).round().astype(int).tolist()
    best = int(np.argmax([f(k) for k in grid]))
    a, b = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]

    inv_phi = (math.sqrt(5) - 1) / 2
    while b - a > K_TOL:
        c = round(b - inv_phi * (b - a))
        d = round(a + inv_phi * (b - a))
        if f(c) >= f(d):
            b = d
        else:
            a = c
    return cache


def main() -> None:
    X, y, _ = load_data(CONFIG.data_dir)
    imp = pd.read_csv(Path(CONFIG.output_dir) / "importance.csv", index_col=0)
    ranking = [c for c in imp.index if not c.startswith("shadow_")]
    cv = KFold(CV_FOLDS, shuffle=True, random_state=CONFIG.seed)

    rows = []
    for name, (make_model, n_jobs) in MODELS.items():
        print(f"=== {name} ===", flush=True)

        def score(k, make_model=make_model, n_jobs=n_jobs, name=name):
            start = time.time()
            s = cross_val_score(
                make_model(k), X[ranking[:k]], y, cv=cv, scoring="r2", n_jobs=n_jobs
            )
            rows.append({"model": name, "k": k, "r2_mean": s.mean(), "r2_std": s.std()})
            print(
                f"[{name}] k={k:3d}  R² {s.mean():.4f} ± {s.std():.4f}  "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )
            return s.mean()

        results = search_k(score, K_MIN, K_MAX)
        best = max(results, key=results.get)
        print(f"[{name}] best k={best}  R² {results[best]:.4f}\n", flush=True)

    out = Path(CONFIG.output_dir) / "sweep.csv"
    df = pd.DataFrame(rows).sort_values(["model", "k"])
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.loc[df.groupby("model")["r2_mean"].idxmax()].to_string(index=False))


if __name__ == "__main__":
    main()
