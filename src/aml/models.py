"""Models, with last year's hyperparameters (to be re-tuned separately)."""

import math

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from aml.config import SEED


MODELS = ("rf", "gp", "svr")


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
