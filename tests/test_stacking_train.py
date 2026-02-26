"""Tests for stacking ensemble training pipeline."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from online_retail_prediction.modeling import stacking_train


def _write_sample_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    features = pd.DataFrame(
        {
            "session_id": list(range(1, 13)),
            "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            "feature_2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1],
            "feature_3": [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6],
        }
    )
    cluster_assignments = pd.DataFrame(
        {
            "session_id": list(range(1, 13)),
            "cluster_id": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    cluster_labels = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "intent_label": ["low-intent", "high-intent"],
        }
    )

    features_path = tmp_path / "features.csv"
    assignments_path = tmp_path / "cluster_assignments.csv"
    labels_path = tmp_path / "cluster_label.csv"
    features.to_csv(features_path, index=False)
    cluster_assignments.to_csv(assignments_path, index=False)
    cluster_labels.to_csv(labels_path, index=False)
    return features_path, assignments_path, labels_path


def test_build_base_estimator_configs_returns_all_models():
    configs = stacking_train.build_base_estimator_configs(random_state=42)

    assert set(configs.keys()) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    }
    for name, (estimator, param_grid) in configs.items():
        assert estimator is not None, f"{name} estimator is None"
        assert isinstance(param_grid, dict), f"{name} param_grid is not a dict"
        assert len(param_grid) > 0, f"{name} param_grid is empty"


def test_tune_base_estimator_returns_fitted_model(tmp_path: Path):
    features_path, assignments_path, labels_path = _write_sample_data(tmp_path)
    X, y = stacking_train.load_processed_features_and_labels(
        features_path, cluster_assignments_path=assignments_path, cluster_labels_path=labels_path
    )
    X_train, _, y_train, _ = stacking_train.split_data(X, y, test_size=0.25, random_state=42)

    configs = stacking_train.build_base_estimator_configs(random_state=42)
    estimator, _ = configs["logistic_regression"]
    tiny_grid = {"classifier__C": [1.0], "classifier__solver": ["lbfgs"]}

    best_model, best_params, best_score, cv_results = stacking_train.tune_base_estimator(
        name="logistic_regression",
        estimator=estimator,
        param_grid=tiny_grid,
        X_train=X_train,
        y_train=y_train,
        cv=2,
    )

    assert best_model is not None
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert 0.0 <= best_score <= 1.0
    assert isinstance(cv_results, pd.DataFrame)
    assert "mean_test_score" in cv_results.columns
    assert "mean_train_score" in cv_results.columns


def test_build_stacking_classifier_creates_valid_stack():
    configs = stacking_train.build_base_estimator_configs(random_state=42)
    estimators = [(name, est) for name, (est, _) in configs.items()]

    stack = stacking_train.build_stacking_classifier(
        tuned_estimators=estimators, cv=2, random_state=42
    )

    assert isinstance(stack, StackingClassifier)
    assert len(stack.estimators) == 4
    assert isinstance(stack.final_estimator, LogisticRegression)


def test_evaluate_model_returns_required_metrics(tmp_path: Path):
    features_path, assignments_path, labels_path = _write_sample_data(tmp_path)
    X, y = stacking_train.load_processed_features_and_labels(
        features_path, cluster_assignments_path=assignments_path, cluster_labels_path=labels_path
    )
    X_train, X_test, y_train, y_test = stacking_train.split_data(
        X, y, test_size=0.25, random_state=42
    )

    configs = stacking_train.build_base_estimator_configs(random_state=42)
    estimators = [(name, est) for name, (est, _) in configs.items()]
    stack = stacking_train.build_stacking_classifier(
        tuned_estimators=estimators, cv=2, random_state=42
    )
    stack.fit(X_train, y_train)

    metrics = stacking_train.evaluate_model(stack, X_test, y_test)
    assert set(metrics.keys()) == {
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "cohens_kappa",
        "accuracy",
    }


def test_main_creates_model_artifacts(tmp_path: Path, monkeypatch):
    features_path, assignments_path, labels_path = _write_sample_data(tmp_path)
    output_dir = tmp_path / "models"

    original_build = stacking_train.build_base_estimator_configs

    def _tiny_configs(random_state: int = 42):
        original = original_build(random_state)
        return {
            "logistic_regression": (
                original["logistic_regression"][0],
                {"classifier__C": [1.0], "classifier__solver": ["lbfgs"]},
            ),
            "random_forest": (
                original["random_forest"][0],
                {"n_estimators": [10], "max_depth": [3], "min_samples_split": [2]},
            ),
            "xgboost": (
                original["xgboost"][0],
                {
                    "n_estimators": [10],
                    "max_depth": [3],
                    "learning_rate": [0.1],
                    "subsample": [1.0],
                    "colsample_bytree": [1.0],
                },
            ),
            "lightgbm": (
                original["lightgbm"][0],
                {
                    "n_estimators": [10],
                    "max_depth": [3],
                    "learning_rate": [0.1],
                    "subsample": [1.0],
                    "colsample_bytree": [1.0],
                },
            ),
        }

    monkeypatch.setattr(stacking_train, "build_base_estimator_configs", _tiny_configs)

    stacking_train.main(
        features_path=features_path,
        cluster_assignments_path=assignments_path,
        cluster_labels_path=labels_path,
        output_dir=output_dir,
        test_size=0.25,
        cv=2,
        random_state=42,
    )

    assert (output_dir / "stacking_ensemble_model.pkl").exists()
    assert (output_dir / "stacking_ensemble_metrics.txt").exists()
    assert (output_dir / "stacking_ensemble_base_comparison.csv").exists()

    # Individual best base estimator models
    assert (output_dir / "stacking_base_logistic_regression.pkl").exists()
    assert (output_dir / "stacking_base_random_forest.pkl").exists()
    assert (output_dir / "stacking_base_xgboost.pkl").exists()
    assert (output_dir / "stacking_base_lightgbm.pkl").exists()

    # CV results for validation/learning curves
    cv_results_dir = output_dir / "stacking_cv_results"
    assert (cv_results_dir / "stacking_logistic_regression_cv_results.csv").exists()
    assert (cv_results_dir / "stacking_random_forest_cv_results.csv").exists()
    assert (cv_results_dir / "stacking_xgboost_cv_results.csv").exists()
    assert (cv_results_dir / "stacking_lightgbm_cv_results.csv").exists()
    assert (cv_results_dir / "stacking_meta_learner_cv_results.csv").exists()
