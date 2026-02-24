# Bagging Training Process

This document describes how the bagging training pipeline works in this project.

## Purpose

The bagging trainer builds and compares three bagging ensembles for purchase-intent prediction:

- Bagging + Logistic Regression
- Bagging + Decision Tree
- Bagging + KNN

The current implementation uses processed features because the next feature-engineering iteration is still in progress.

## Data Inputs

The script expects:

- `data/processed/features_first_n.csv`
- `data/processed/baseline_labels.csv`

The pipeline:

1. Loads both files.
2. Merges on `session_id`.
3. Drops rows where `label` is missing.
4. Uses numeric feature columns only.
5. Casts labels to integer class values.
6. Splits data into train/test with stratification.

## Modeling Flow

The trainer runs one full tuning-and-evaluation cycle for each base estimator family:

1. Build a base estimator configuration.
2. Wrap it inside `BaggingClassifier`.
3. Tune hyperparameters with `GridSearchCV`.
4. Refit the best configuration on training data.
5. Evaluate on train and test sets.
6. Save model artifacts and metrics.

## Fine-Tuning Objective

Hyperparameter tuning optimizes **ROC-AUC only**.

`GridSearchCV(scoring="roc_auc")` is used for all three bagging families.

## Evaluation Metrics

Each trained model is evaluated and reported on:

- ROC-AUC
- Precision
- Recall
- F1
- Macro-F1
- Cohen's Kappa
- Accuracy

## Hyperparameters Tuned

### Bagging-level

- `n_estimators`
- `max_samples`
- `max_features`
- `bootstrap`

### Logistic Regression base estimator

- `C`

### Decision Tree base estimator

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

### KNN base estimator

- `n_neighbors`
- `weights`
- `p`

## Outputs

The pipeline writes artifacts to `models/`:

- `bagging_logistic_regression.pkl`
- `bagging_decision_tree.pkl`
- `bagging_knn.pkl`
- `bagging_logistic_regression_metrics.txt`
- `bagging_decision_tree_metrics.txt`
- `bagging_knn_metrics.txt`
- `bagging_model_comparison.csv`

The comparison file ranks models by test ROC-AUC.

## Run Command

```bash
uv run python -m online_retail_prediction.modeling.bagging_train
```

Optional arguments include custom features path, labels path, output directory, test-size, CV folds, and random seed.
