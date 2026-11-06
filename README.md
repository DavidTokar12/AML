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
uv run train   # fit with src/aml/config.py, writes outputs/submission.csv
uv run tune    # CV grid search from src/aml/tune.py, writes outputs/tuning.csv
uv run importance  # RF permutation importances (slow, run on the HPC), writes outputs/importance.csv
```

## Layout

- `src/aml/config.py`: all settings (imputer, scaler, outlier removal, feature selection, model). `train` uses these.
- `src/aml/tune.py`: the hyperparameter grid. Any `Config` field can go in it. Copy the best values back into `config.py`.
- `src/aml/pipeline.py`: outlier removal, the sklearn pipeline, and cross-validation.
- `src/aml/data.py`: loading data and writing the submission.
- `src/aml/features.py`: feature selection on raw data (work in progress).
- `notebooks/feature_selection.ipynb`: exploring the feature selection steps.

## Pipeline

1. Outlier removal: impute, scale, 2D PLS embedding, then IsolationForest drops a `contamination` fraction of training samples.
2. Impute, then scale.
3. Feature selection: drop low-variance features, keep the `top_k_corr` features most correlated with `y`, drop near-duplicates (|r| ≥ `corr_feature_max`), keep the `rf_keep` most important features of a random forest.
4. RBF SVR.

Steps 2–4 are one sklearn `Pipeline`, so in CV they are fit on the training folds only.
