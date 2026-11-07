"""Compare GP kernels on a feature set, saving every fit's predictions.

Feature sets (first argument, default "block"):
  block    the correlated block of the cleaned features (~185, no y involved)
  gp_perm  cleaned features with GP permutation importance > 0 (219)
Preprocessing (fixed): quantile transform to normal, then KNNImputer (default).
CV: 5 folds x 3 repeats. Training folds exclude the outlier rows, R² is scored
on the inlier validation rows (predictions are saved for all rows).

Kernels (see aml.models.gp_kernel):
  stage 1, one shared length scale: rbf, matern05/15/25, rq, each plain,
           + linear term, + long RBF                           (15 kernels)
           and each plain one again with a noise floor of 0.01 ("_nf", 5 kernels):
           without the floor the noise level collapses to ~1e-5
  stage 2, one length scale per feature (ARD): rbf, matern15, matern25, each
           plain and + linear term, optimizer capped at MAX_ITER  (6 kernels)
Every kernel is also fitted on all inlier training rows to predict the test set.

Outputs in outputs/gp_kernels/<feature set>/:
  <kernel>_oof.csv   out-of-fold predictions: id, repeat, fold, y, pred, std, inlier
  <kernel>_test.csv  test predictions: id, pred, std
  <kernel>_fits.csv  per fit: repeat, fold (0 = full fit), r2, log marginal
                     likelihood, fitted kernel, minutes
  summary.csv        per kernel: R² mean/std, difference to the baseline, folds won
Kernels whose _fits.csv exists are skipped, so a crashed run can be restarted.

Run on the HPC: uv run python scripts/gp_kernels.py [block|gp_perm]
"""

import sys
import time
import warnings

from collections import defaultdict

import numpy as np
import pandas as pd

from joblib import Parallel
from joblib import delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from aml.config import CV_FOLDS
from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_data
from aml.features import clean_features
from aml.features import correlated_block
from aml.features import outlier_scores
from aml.features import verbose_lbfgs
from aml.models import GP_BASES
from aml.models import gp_kernel


OUTLIER_FRAC = 0.045
N_REPEATS = 3
ISO_RESTARTS = 2  # extra optimizer starts for the cheap shared-length-scale kernels
MAX_ITER = 300  # optimizer steps for ARD kernels
NOISE_FLOOR = 1e-2  # the floor of the old setup (R² 0.694)

STAGES = {
    "shared length scale": [
        (base, False, extra, 1e-5)
        for base in GP_BASES
        for extra in (None, "linear", "long")
    ]
    + [(base, False, None, NOISE_FLOOR) for base in GP_BASES],
    "ARD": [
        (base, True, extra, 1e-5)
        for base in ("rbf", "matern15", "matern25")
        for extra in (None, "linear")
    ],
}
# Parallel fits per stage: an ARD fit needs ~3-5 GB, so 64 of them don't fit
# in 128 GB
N_JOBS = {"shared length scale": 64, "ARD": 16}
BASELINE = "rbf"  # the kernel used so far


def kernel_name(base: str, ard: bool, extra: str | None, noise_floor: float) -> str:
    return (
        base
        + ("_ard" if ard else "")
        + (f"+{extra}" if extra else "")
        + ("_nf" if noise_floor > 1e-5 else "")
    )


def make_gp(base, ard, extra, noise_floor, d: int, label: str):
    gp = GaussianProcessRegressor(
        gp_kernel(base, d, ard, extra, noise_floor),
        normalize_y=True,
        optimizer=verbose_lbfgs(label, MAX_ITER) if ard else "fmin_l_bfgs_b",
        n_restarts_optimizer=0 if ard else ISO_RESTARTS,
        random_state=SEED,
    )
    return make_pipeline(
        QuantileTransformer(
            n_quantiles=500, output_distribution="normal", random_state=SEED
        ),
        KNNImputer(),
        gp,
    )


def fit_one(spec, repeat, fold, train, val, X, y, X_test, inlier):
    """One fit. fold 0 = fit on all inlier rows and predict the test set."""
    name = kernel_name(*spec)
    label = f"{name} r{repeat} f{fold}"
    start = time.time()
    with warnings.catch_warnings():
        # Unused parts end at their bounds (linear weight, RQ alpha, the length
        # scale of an ignored feature or of the long RBF): expected, so hidden.
        # The noise level warning stays.
        warnings.filterwarnings(
            "ignore",
            message=r".*(constant_value|sigma_0|alpha|length_scale is close to the specified upper).*",
            category=ConvergenceWarning,
        )
        model = make_gp(*spec, X.shape[1], label).fit(X.iloc[train], y.iloc[train])

    if fold == 0:
        pred, std = model.predict(X_test, return_std=True)
        rows = pd.DataFrame({"id": X_test.index, "pred": pred, "std": std})
        r2 = np.nan
    else:
        pred, std = model.predict(X.iloc[val], return_std=True)
        rows = pd.DataFrame(
            {
                "id": X.index[val],
                "repeat": repeat,
                "fold": fold,
                "y": y.iloc[val].to_numpy(),
                "pred": pred,
                "std": std,
                "inlier": inlier[val],
            }
        )
        scored = rows[rows["inlier"]]
        r2 = r2_score(scored["y"], scored["pred"])

    gp = model[-1]
    fit = {
        "repeat": repeat,
        "fold": fold,
        "r2": r2,
        "log_marginal_likelihood": gp.log_marginal_likelihood_value_,
        "kernel": str(gp.kernel_),
        "minutes": (time.time() - start) / 60,
    }
    print(f"  [{label}] R² {r2:.4f}, {fit['minutes']:.1f} min", flush=True)
    return name, fit, rows


def save(out, name: str, results: list) -> None:
    fits = pd.DataFrame([fit for fit, _ in results]).sort_values(["repeat", "fold"])
    rows = {fit["fold"] == 0: [] for fit, _ in results}
    for fit, r in results:
        rows[fit["fold"] == 0].append(r)
    pd.concat(rows[False]).sort_values(["repeat", "fold", "id"]).to_csv(
        out / f"{name}_oof.csv", index=False
    )
    pd.concat(rows[True]).to_csv(out / f"{name}_test.csv", index=False)
    fits.to_csv(out / f"{name}_fits.csv", index=False)
    cv = fits[fits["fold"] > 0]["r2"]
    print(f"=== {name}: R² {cv.mean():.4f} ± {cv.std():.4f}, saved ===", flush=True)


def summarize(out) -> pd.DataFrame:
    scores = {}
    for path in sorted(out.glob("*_fits.csv")):
        fits = pd.read_csv(path)
        fits = fits[fits["fold"] > 0].sort_values(["repeat", "fold"])
        scores[path.name.removesuffix("_fits.csv")] = fits["r2"].to_numpy()
    base = scores[BASELINE]
    summary = pd.DataFrame(
        {
            name: {
                "r2_mean": s.mean(),
                "r2_std": s.std(),
                "vs_baseline": (s - base).mean(),
                "folds_won": f"{(s > base).sum()}/{len(s)}",
            }
            for name, s in scores.items()
        }
    ).T.sort_values("r2_mean", ascending=False)
    summary.to_csv(out / "summary.csv")
    return summary


def main() -> None:
    feature_set = sys.argv[1] if len(sys.argv) > 1 else "block"
    out = OUTPUT_DIR / "gp_kernels" / feature_set
    out.mkdir(parents=True, exist_ok=True)
    X_train, y, X_test = load_data()
    X, X_test = clean_features(X_train, X_test)

    # Outlier rows flagged on all cleaned features, as in the earlier scripts
    n_out = round(OUTLIER_FRAC * len(X))
    inlier = ~X.index.isin(outlier_scores(X, y, seed=SEED).index[:n_out])

    if feature_set == "block":
        keep = set(correlated_block(pd.concat([X, X_test])))
    elif feature_set == "gp_perm":
        imp = pd.read_csv(OUTPUT_DIR / "importance_gp_perm.csv", index_col=0)
        imp = imp[~imp.index.str.startswith("shadow_")]
        keep = set(imp.index[imp["mean"] > 0])
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    features = [c for c in X.columns if c in keep]
    X, X_test = X[features], X_test[features]
    print(
        f"Feature set {feature_set}: {len(features)} features, {len(X)} rows ({inlier.sum()} inliers), "
        f"{len(X_test)} test rows",
        flush=True,
    )

    folds = list(
        RepeatedKFold(n_splits=CV_FOLDS, n_repeats=N_REPEATS, random_state=SEED).split(
            X
        )
    )
    all_inliers = np.where(inlier)[0]

    for stage, specs in STAGES.items():
        specs = [s for s in specs if not (out / f"{kernel_name(*s)}_fits.csv").exists()]
        if not specs:
            print(f"##### {stage}: all kernels done, skipping #####", flush=True)
            continue
        print(
            f"##### {stage}: {len(specs)} kernels: "
            f"{[kernel_name(*s) for s in specs]} #####",
            flush=True,
        )
        tasks = []
        for spec in specs:
            for i, (train, val) in enumerate(folds):
                repeat, fold = i // CV_FOLDS + 1, i % CV_FOLDS + 1
                tasks.append(
                    delayed(fit_one)(
                        spec,
                        repeat,
                        fold,
                        train[inlier[train]],
                        val,
                        X,
                        y,
                        X_test,
                        inlier,
                    )
                )
            tasks.append(
                delayed(fit_one)(spec, 0, 0, all_inliers, None, X, y, X_test, inlier)
            )

        per_kernel = len(folds) + 1
        collected = defaultdict(list)
        start = time.time()
        results = Parallel(n_jobs=N_JOBS[stage], return_as="generator_unordered")(tasks)
        for done, (name, fit, rows) in enumerate(results, 1):
            collected[name].append((fit, rows))
            if done % 10 == 0:
                print(
                    f"--- {stage}: {done}/{len(tasks)} fits, "
                    f"{(time.time() - start) / 60:.1f} min ---",
                    flush=True,
                )
            if len(collected[name]) == per_kernel:
                save(out, name, collected.pop(name))

    print("\nSummary (R² on inlier validation rows, 5 folds x 3 repeats):")
    print(summarize(out).to_string())


if __name__ == "__main__":
    main()
