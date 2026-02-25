"""Tests for bagging model training with cluster-derived labels."""

from pathlib import Path

import pandas as pd
import pytest

from online_retail_prediction.modeling import bagging_train


def _write_sample_features_cluster_labels(tmp_path: Path) -> tuple[Path, Path, Path]:
    features = pd.DataFrame(
        {
            "session_id": list(range(1, 13)),
            "feature_1": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
            ],
            "feature_2": [
                1.0,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
                0.0,
                -0.1,
            ],
            "feature_3": [
                5,
                4,
                3,
                2,
                1,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
            ],
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
    cluster_assignments_path = tmp_path / "cluster_assignments.csv"
    cluster_labels_path = tmp_path / "cluster_label.csv"
    features.to_csv(features_path, index=False)
    cluster_assignments.to_csv(cluster_assignments_path, index=False)
    cluster_labels.to_csv(cluster_labels_path, index=False)

    return features_path, cluster_assignments_path, cluster_labels_path


def test_load_processed_features_and_labels_success(tmp_path: Path):
    features_path, cluster_assignments_path, cluster_labels_path = (
        _write_sample_features_cluster_labels(tmp_path)
    )

    X, y = bagging_train.load_processed_features_and_labels(
        features_path=features_path,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
    )

    assert not X.empty
    assert len(X) == len(y) == 12
    assert list(X.columns) == ["feature_1", "feature_2", "feature_3"]
    assert y.nunique() == 2


def test_split_data_preserves_total_rows(tmp_path: Path):
    features_path, cluster_assignments_path, cluster_labels_path = (
        _write_sample_features_cluster_labels(tmp_path)
    )
    X, y = bagging_train.load_processed_features_and_labels(
        features_path=features_path,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
    )

    X_train, X_test, y_train, y_test = bagging_train.split_data(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_load_processed_features_and_labels_raises_for_invalid_intent_label(tmp_path: Path):
    features_path, cluster_assignments_path, cluster_labels_path = (
        _write_sample_features_cluster_labels(tmp_path)
    )
    cluster_labels = pd.read_csv(cluster_labels_path)
    cluster_labels.loc[cluster_labels["cluster_id"] == 1, "intent_label"] = "medium-intent"
    cluster_labels.to_csv(cluster_labels_path, index=False)

    with pytest.raises(ValueError, match="Unsupported intent labels"):
        bagging_train.load_processed_features_and_labels(
            features_path=features_path,
            cluster_assignments_path=cluster_assignments_path,
            cluster_labels_path=cluster_labels_path,
        )


def test_evaluate_model_returns_required_metrics(tmp_path: Path):
    features_path, cluster_assignments_path, cluster_labels_path = (
        _write_sample_features_cluster_labels(tmp_path)
    )
    X, y = bagging_train.load_processed_features_and_labels(
        features_path=features_path,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
    )
    X_train, X_test, y_train, y_test = bagging_train.split_data(
        X, y, test_size=0.25, random_state=42
    )

    configs = bagging_train.build_model_configs(random_state=42)
    base_estimator, param_grid = configs["decision_tree"]
    small_grid = {
        "n_estimators": [10],
        "max_samples": [0.8],
        "max_features": [1.0],
        "bootstrap": [True],
        "estimator__max_depth": [3],
        "estimator__min_samples_split": [2],
        "estimator__min_samples_leaf": [1],
    }

    model, _, _ = bagging_train.tune_and_train_bagging_model(
        model_name="decision_tree",
        base_estimator=base_estimator,
        param_grid=small_grid,
        X_train=X_train,
        y_train=y_train,
        cv=2,
        random_state=42,
    )

    metrics = bagging_train.evaluate_model(model, X_test, y_test)
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
    features_path, cluster_assignments_path, cluster_labels_path = (
        _write_sample_features_cluster_labels(tmp_path)
    )
    output_dir = tmp_path / "models"
    original_build_model_configs = bagging_train.build_model_configs

    def _small_model_configs(random_state: int = 42):
        original_configs = original_build_model_configs(random_state)
        return {
            "logistic_regression": (
                original_configs["logistic_regression"][0],
                {
                    "n_estimators": [10],
                    "max_samples": [1.0],
                    "max_features": [1.0],
                    "bootstrap": [False],
                    "estimator__classifier__C": [1.0],
                },
            ),
            "decision_tree": (
                original_configs["decision_tree"][0],
                {
                    "n_estimators": [10],
                    "max_samples": [1.0],
                    "max_features": [1.0],
                    "bootstrap": [False],
                    "estimator__max_depth": [3],
                    "estimator__min_samples_split": [2],
                    "estimator__min_samples_leaf": [1],
                },
            ),
            "knn": (
                original_configs["knn"][0],
                {
                    "n_estimators": [10],
                    "max_samples": [1.0],
                    "max_features": [1.0],
                    "bootstrap": [False],
                    "estimator__classifier__n_neighbors": [3],
                    "estimator__classifier__weights": ["uniform"],
                    "estimator__classifier__p": [2],
                },
            ),
        }

    monkeypatch.setattr(bagging_train, "build_model_configs", _small_model_configs)

    bagging_train.main(
        features_path=features_path,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
        output_dir=output_dir,
        test_size=0.25,
        cv=2,
        random_state=42,
    )

    assert (output_dir / "bagging_logistic_regression.pkl").exists()
    assert (output_dir / "bagging_decision_tree.pkl").exists()
    assert (output_dir / "bagging_knn.pkl").exists()
    assert (output_dir / "bagging_logistic_regression_metrics.txt").exists()
    assert (output_dir / "bagging_decision_tree_metrics.txt").exists()
    assert (output_dir / "bagging_knn_metrics.txt").exists()
    assert (output_dir / "bagging_model_comparison.csv").exists()
