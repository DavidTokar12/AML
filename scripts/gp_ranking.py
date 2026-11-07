"""Final feature ranking with an ARD GP (one length scale per feature).

Candidates: every feature that beats its best shadow in at least one of the
first-stage methods (outputs/importance_{rf,hgb,svr,lasso,mi}.csv), plus every
feature of the correlated block, so no feature gets left behind. Same rows as
the first stage (cleaned features, outlier rows removed) plus 20 new shadows.

Run on the HPC after scripts/feature_importance.py:
  uv run python scripts/gp_ranking.py
Writes outputs/importance_gp_perm.csv (permutation importance, the one to\ntrust) and outputs/importance_gp_ls.csv (1 / length scale, for comparison).
"""

import pandas as pd

from aml.config import OUTPUT_DIR
from aml.config import SEED
from aml.data import load_data
from aml.features import add_shadows
from aml.features import clean_features
from aml.features import correlated_block
from aml.features import gp_ard_importance
from aml.features import outlier_scores


OUTLIER_FRAC = 0.045
SELECTORS = ("rf", "hgb", "svr", "lasso", "mi")


def beats_shadows(imp: pd.DataFrame) -> set[str]:
    """Features scoring above the best shadow feature."""
    is_shadow = imp.index.str.startswith("shadow_")
    best_shadow = imp.loc[is_shadow, "mean"].max()
    return set(imp.index[(imp["mean"] > best_shadow) & ~is_shadow])


def main() -> None:
    X_train, y, X_test = load_data()
    X, X_test = clean_features(X_train, X_test)

    selected = set()
    for name in SELECTORS:
        picked = beats_shadows(
            pd.read_csv(OUTPUT_DIR / f"importance_{name}.csv", index_col=0)
        )
        print(f"{name}: {len(picked)} features", flush=True)
        selected |= picked
    block = set(correlated_block(pd.concat([X, X_test])))
    candidates = [c for c in X.columns if c in selected | block]
    print(
        f"Union {len(selected)}, correlated block {len(block)}, "
        f"candidates {len(candidates)}",
        flush=True,
    )

    n_out = round(OUTLIER_FRAC * len(X))
    inlier = ~X.index.isin(outlier_scores(X, y, seed=SEED).index[:n_out])
    X = add_shadows(X.loc[inlier, candidates], seed=SEED)
    y = y[inlier]
    print(f"{X.shape[1]} features (incl. 20 shadows), {len(X)} rows", flush=True)

    perm, length_scale = gp_ard_importance(X, y, seed=SEED)
    for name, imp in (("perm", perm), ("ls", length_scale)):
        path = OUTPUT_DIR / f"importance_gp_{name}.csv"
        imp.to_csv(path)
        is_shadow = imp.index.str.startswith("shadow_")
        best_shadow = imp.loc[is_shadow, "mean"].max()
        n_beat = (imp.loc[~is_shadow, "mean"] > best_shadow).sum()
        print(f"\n{name}: {n_beat} features beat the best shadow, wrote {path}")
        print(imp.head(20).to_string())


if __name__ == "__main__":
    main()
