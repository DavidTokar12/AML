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

## Run

```bash
uv run python scripts/preprocessing_experiments.py   # slow, run on the HPC
```

Flags outliers, ranks features by RF permutation importance (cached in `outputs/importance_*.csv`), then for each model (random forest, Gaussian process, SVR) decides: outlier removal → quantile transform → KNN imputation, each with its own best number of top features. Writes `outputs/preprocessing_experiments.csv`.

## Layout

- `src/aml/config.py`: seed, CV folds, output folder.
- `src/aml/data.py`: downloads and loads the data.
- `src/aml/features.py`: constant features, row outlier scores, RF permutation importance.
- `src/aml/models.py`: the models and their preprocessing options (last year's hyperparameters for now).
- `src/aml/evaluate.py`: cross-validation and the search for the best number of features.
- `scripts/`: one-off experiments.
- `notebooks/feature_selection.ipynb`: looking at the data and the feature selection steps.
