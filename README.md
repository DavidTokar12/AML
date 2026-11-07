# AML 2026 – Project 1

Predict age (`y`) from 832 noisy features with missing values.

## Setup

Open the repo in the devcontainer, or run `uv sync` locally.

Create a `.env` file in the repo root with your Kaggle token
(kaggle.com → Settings → API → Create New Token):

```
KAGGLE_API_TOKEN=KGAT_...
```

The data is downloaded and cached by `kagglehub` the first time you run anything.

## Run (on the HPC, in this order)

```bash
uv run python scripts/preprocess.py   # locked features -> outputs/features/{train,test}.csv
uv run python scripts/submit.py       # every model fit on all inlier rows -> outputs/submission_<model>.csv
```

`scripts/feature_importance.py` (RF + MI importance) is only needed for the notebook's record of the selection.

## Pipeline

1. **Cleaning:** drop constant features, exact copies and the 55 uniform-noise features (value range exactly 0.1), and merge groups of noisy copies (|Spearman| ≥ 0.94) into their least noisy member.
2. **Outlier rows:** flag the ~4.5% most anomalous training rows (IsolationForest on a 2-D PLS projection) and leave them out of training and scoring.
3. **Features:** keep the correlated block, the 185 features correlated with at least one other feature; the rest correlate with nothing and carry no signal.
4. **Scaling:** quantile-transform every feature to a normal distribution, which removes the huge differences in scale and tails.
5. **Imputation:** iterative imputation, predicting each feature's missing values from the other features with linear regression, 10 rounds.
6. **Model:** GP with a Matern 2.5 kernel and one length scale shared by all features.
7. **Validation:** 5 folds × 3 repeats; train without outlier rows, R² on inlier rows.

Steps 1–5 use no `y` and are fitted on train + test together.

## Layout

- `src/aml/config.py`: seed, CV folds, outlier share, output folders.
- `src/aml/data.py`: downloads and loads the data; `load_clean_data` adds cleaning and the outlier mask; `load_features` reads the locked features.
- `src/aml/features.py`: cleaning steps, outlier scores, RF and MI importance.
- `src/aml/models.py`: the locked preprocessing and the GP.
- `scripts/`: HPC runs.
- `notebooks/feature_selection.ipynb`: looking at the data and every selection step.
