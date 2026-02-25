# Bagging Training Process

This document describes how the bagging training pipeline works in this project.

## Purpose

The bagging trainer builds and compares three bagging ensembles for purchase-intent prediction:

- Bagging + Logistic Regression
- Bagging + Decision Tree
- Bagging + KNN

The current implementation uses processed features and cluster-derived intent labels.

## Data Inputs

The script expects:

- `data/processed/features_first_n.csv`
- `data/cluster_outputs/cluster_assignments.csv`
- `data/cluster_outputs/cluster_label.csv`

The pipeline:

1. Loads processed features, cluster assignments, and cluster label mapping.
2. Merges `cluster_assignments.csv` with `cluster_label.csv` on `cluster_id`.
3. Maps cluster intent labels to binary target labels (`low-intent=0`, `high-intent=1`).
4. Drops `cluster_id` and keeps session-level (`session_id`, `label`) targets.
5. Merges derived labels into features on `session_id`.
6. Uses numeric feature columns only.
7. Splits data into train/test with stratification.

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

Optional arguments include custom features path, cluster-assignments path, cluster-labels path,
output directory, test-size, CV folds, and random seed.
