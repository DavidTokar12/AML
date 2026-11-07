from pathlib import Path


SEED = 25
CV_FOLDS = 5
OUTPUT_DIR = Path("outputs")
OUTLIER_FRAC = 0.045  # share of training rows flagged as outliers (~55 rows)
FEATURES_DIR = OUTPUT_DIR / "features"  # the locked, preprocessed features
