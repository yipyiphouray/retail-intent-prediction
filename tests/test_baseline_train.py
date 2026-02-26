"""Tests for baseline training with cluster-derived session labels."""

from pathlib import Path

import pandas as pd
import pytest

from online_retail_prediction.modeling import baseline_train


def _write_sample_cluster_labeled_data(tmp_path: Path) -> tuple[Path, Path, Path]:
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


def test_load_and_prepare_data_builds_session_labels(tmp_path: Path):
    features_path, cluster_assignments_path, cluster_labels_path = _write_sample_cluster_labeled_data(
        tmp_path
    )

    features, labels = baseline_train.load_and_prepare_data(
        features_path=features_path,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
    )

    assert not features.empty
    assert list(labels.columns) == ["session_id", "label"]
    assert labels["label"].isin([0, 1]).all()
    assert labels["session_id"].nunique() == len(labels)


def test_load_and_prepare_data_raises_for_invalid_cluster_label(tmp_path: Path):
    features_path, cluster_assignments_path, cluster_labels_path = _write_sample_cluster_labeled_data(
        tmp_path
    )
    cluster_labels = pd.read_csv(cluster_labels_path)
    cluster_labels.loc[cluster_labels["cluster_id"] == 1, "intent_label"] = "medium-intent"
    cluster_labels.to_csv(cluster_labels_path, index=False)

    with pytest.raises(ValueError, match="Unsupported intent labels"):
        baseline_train.load_and_prepare_data(
            features_path=features_path,
            cluster_assignments_path=cluster_assignments_path,
            cluster_labels_path=cluster_labels_path,
        )


def test_main_trains_with_cluster_label_files(tmp_path: Path):
    features_path, cluster_assignments_path, cluster_labels_path = _write_sample_cluster_labeled_data(
        tmp_path
    )
    model_path = tmp_path / "baseline_model.pkl"

    baseline_train.main(
        features_path=features_path,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
        model_path=model_path,
        model_type="logistic_regression",
        test_size=0.25,
        random_state=42,
        tune=False,
    )

    assert model_path.exists()
    assert (tmp_path / "baseline_model_metrics.txt").exists()
