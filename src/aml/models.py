"""Locked preprocessing (quantile transform + iterative imputation) and the GP."""

import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from aml.config import SEED


def make_preprocessor():
    """Quantile transform to normal, then iterative imputation (each feature
    regressed on the others with BayesianRidge, 10 rounds). Uses no y.

    The imputer won the comparison (GP R² 0.719 vs 0.711 for KNN with 5
    neighbours, better in 14/15 folds); KNN with 3-50 neighbours, uniform or
    distance-weighted, and iterative imputation with ExtraTrees were worse.
    Scaling comes first so the imputer sees comparable features.
    """
    return make_pipeline(
        QuantileTransformer(
            n_quantiles=500, output_distribution="normal", random_state=SEED
        ),
        IterativeImputer(max_iter=10, random_state=SEED),
    ).set_output(transform="pandas")


def gp_kernel(d: int):
    """GP kernel for d quantile-transformed features: amplitude * Matern(2.5) +
    noise, one length scale shared by all features.

    Won the kernel comparison on the correlated block (R² 0.711, 5x3 CV): RBF,
    Matern 0.5/1.5, RationalQuadratic, extra linear or long-RBF terms, a noise
    floor and one length scale per feature (overfits: 0.64) were all equal or
    worse. The noise level ends at its lower bound; a floor of 0.01 gave the
    same R², so that is harmless.
    """
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        np.sqrt(d), (1e-1, 1e4), nu=2.5
    ) + WhiteKernel(0.3, (1e-5, 10.0))


def make_gp(d: int) -> GaussianProcessRegressor:
    """GP for d preprocessed features."""
    return GaussianProcessRegressor(
        gp_kernel(d), normalize_y=True, n_restarts_optimizer=2, random_state=SEED
    )
