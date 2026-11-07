"""Models, with last year's hyperparameters (to be re-tuned separately)."""

import math

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from aml.config import SEED


MODELS = ("rf", "gp", "svr")
GP_BASES = ("rbf", "matern05", "matern15", "matern25", "rq")
MATERN_NU = {"matern05": 0.5, "matern15": 1.5, "matern25": 2.5}


def gp_kernel(
    base: str,
    d: int,
    ard: bool = False,
    extra: str | None = None,
    noise_floor: float = 1e-5,
):
    """GP kernel for d quantile-transformed features: amplitude * base + noise.

    base: "rbf", "matern05", "matern15", "matern25" (rougher to smoother) or
      "rq" (rational quadratic: a mix of length scales).
    ard: one length scale per feature instead of one shared length scale.
    extra: "linear" adds a linear term (big smooth trends), "long" adds a
      second RBF with a long length scale (broad trend plus local detail).
    noise_floor: lower bound of the noise level (on normalized y). Without a
      floor the noise can collapse to ~0 and the GP fits the training ages as
      if they were exact.
    """
    length_scale = np.full(d, np.sqrt(d)) if ard else np.sqrt(d)
    bounds = (1e-2, 1e5) if ard else (1e-1, 1e4)
    if base == "rbf":
        k = RBF(length_scale, bounds)
    elif base in MATERN_NU:
        k = Matern(length_scale, bounds, nu=MATERN_NU[base])
    elif base == "rq" and not ard:
        k = RationalQuadratic(length_scale, alpha=1.0, length_scale_bounds=bounds)
    else:
        raise ValueError(f"Unknown kernel: {base} (ard={ard})")

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * k
    if extra == "linear":
        kernel += ConstantKernel(0.1, (1e-5, 1e2)) * DotProduct(1.0, (1e-5, 1e2))
    elif extra == "long":
        kernel += ConstantKernel(0.1, (1e-5, 1e2)) * RBF(10 * np.sqrt(d), (1.0, 1e5))
    elif extra is not None:
        raise ValueError(f"Unknown extra: {extra}")
    return kernel + WhiteKernel(0.3, (noise_floor, 10.0))


def make_model(name: str, k: int, quantile: bool = False, knn: bool = False):
    """Model for k input features.

    quantile: QuantileTransformer (to normal) instead of StandardScaler.
    knn: KNN imputation instead of median imputation.
    The random forest takes NaNs natively and ignores scaling, so it skips both.
    """
    if name == "rf":
        return RandomForestRegressor(n_estimators=1000, random_state=SEED, n_jobs=-1)

    if name == "svr":
        model = SVR(kernel="rbf", C=52, gamma="scale")
    elif name == "gp":
        kernel = ConstantKernel() * RBF(length_scale=math.sqrt(k)) + WhiteKernel(
            noise_level=0.3, noise_level_bounds=(1e-2, 1.0)
        )
        model = GaussianProcessRegressor(
            kernel, normalize_y=True, n_restarts_optimizer=2, random_state=SEED
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    scaler = (
        QuantileTransformer(
            n_quantiles=500, output_distribution="normal", random_state=SEED
        )
        if quantile
        else StandardScaler()
    )
    imputer = KNNImputer() if knn else SimpleImputer(strategy="median")
    # Scale before imputing, so KNN distances aren't dominated by huge features.
    return make_pipeline(scaler, imputer, model)
