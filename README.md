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
uv run python scripts/feature_importance.py   # RF + MI importance -> outputs/importance_{rf,mi}.csv
uv run python scripts/selection_curves.py     # GP R² vs number of features -> outputs/selection_curves.csv
uv run python scripts/gp_cv.py                # GP on the correlated block, OOF + test predictions -> outputs/gp/
```

## Pipeline

1. Cleaning (`clean_features`): drop constant features, exact copies and uniform-noise features, merge groups of noisy copies.
2. Drop the ~4.5% outlier rows from training (scored on inliers only).
3. Features: the correlated block (features correlated with at least one other), ranked by RF permutation importance and mutual information.
4. Model: quantile transform → KNN imputation → GP with a Matern 2.5 kernel.

## Layout

- `src/aml/config.py`: seed, CV folds, outlier share, output folder.
- `src/aml/data.py`: downloads and loads the data; `load_clean_data` adds cleaning and the outlier mask.
- `src/aml/features.py`: cleaning steps, outlier scores, RF and MI importance.
- `src/aml/models.py`: the GP pipeline.
- `scripts/`: HPC runs.
- `notebooks/feature_selection.ipynb`: looking at the data and every selection step.
