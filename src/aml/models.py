"""The model: quantile transform + KNN imputation + GP with a Matern 2.5 kernel."""

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from aml.config import SEED


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


def make_gp(d: int):
    """Pipeline for d features. Scaling before imputing, so KNN distances
    aren't dominated by features with huge values."""
    return make_pipeline(
        QuantileTransformer(
            n_quantiles=500, output_distribution="normal", random_state=SEED
        ),
        KNNImputer(),
        GaussianProcessRegressor(
            gp_kernel(d), normalize_y=True, n_restarts_optimizer=2, random_state=SEED
        ),
    )
