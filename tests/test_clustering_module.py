"""Tests for session clustering, cluster labeling propagation, and output exports."""

from pathlib import Path

import pandas as pd
import pytest

from online_retail_prediction.modeling import clustering
from online_retail_prediction.modeling.clustering import SessionClusterer


@pytest.fixture
def sample_session_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": list(range(1, 13)),
            "n_clicks_observed": [3, 4, 5, 3, 4, 5, 9, 10, 11, 9, 10, 11],
            "price_mean": [10, 11, 9, 10.5, 11.2, 9.3, 100, 105, 98, 101, 108, 96],
            "n_unique_pages": [1, 1, 2, 2, 2, 1, 5, 6, 6, 5, 6, 5],
            "category_entropy": [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 1.3, 1.4, 1.2, 1.3, 1.5, 1.2],
            "last_page": [1, 1, 2, 2, 2, 1, 4, 4, 5, 4, 5, 4],
            "non_numeric_tag": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"],
        }
    )


def test_fit_clustering_returns_session_cluster_mapping(
    sample_session_features: pd.DataFrame,
) -> None:
    clusterer = SessionClusterer(random_state=7)
    assignments = clusterer.fit_clustering(sample_session_features, k=3)

    assert set(assignments.columns) == {"session_id", "cluster_id"}
    assert len(assignments) == len(sample_session_features)
    assert assignments["session_id"].nunique() == len(sample_session_features)


def test_cluster_summary_contains_required_fields(sample_session_features: pd.DataFrame) -> None:
    clusterer = SessionClusterer(random_state=7)
    clusterer.fit_clustering(sample_session_features, k=3)

    summary = clusterer.get_cluster_summary()

    assert "cluster_size" in summary.columns
    assert "cluster_percentage" in summary.columns
    assert all(
        column.startswith("mean_")
        for column in summary.columns
        if column not in ["cluster_id", "cluster_size", "cluster_percentage"]
    )
    assert "mean_purchase_flag" not in summary.columns
    assert "mean_revenue" not in summary.columns


def test_representatives_rank_and_columns(sample_session_features: pd.DataFrame) -> None:
    clusterer = SessionClusterer(random_state=7)
    clusterer.fit_clustering(sample_session_features, k=3)

    top_one = clusterer.get_representatives(top_n=1)
    top_five = clusterer.get_representatives(top_n=5)

    assert set(top_one.columns) == {
        "session_id",
        "cluster_id",
        "distance_to_centroid",
        "rank_within_cluster",
    }
    assert top_one["rank_within_cluster"].max() == 1
    assert len(top_five) >= len(top_one)


def test_manual_labels_propagate_to_sessions(sample_session_features: pd.DataFrame) -> None:
    clusterer = SessionClusterer(random_state=7)
    assignments = clusterer.fit_clustering(sample_session_features, k=3)

    labels = pd.DataFrame(
        {
            "cluster_id": sorted(assignments["cluster_id"].unique().tolist()),
            "intent_label": ["low_intent", "medium_intent", "high_intent"],
        }
    )

    labeled_sessions = clusterer.apply_manual_labels(labels)

    assert set(labeled_sessions.columns) == {"session_id", "cluster_id", "intent_label"}
    assert labeled_sessions["session_id"].nunique() == len(sample_session_features)
    merged = labeled_sessions.merge(assignments, on=["session_id", "cluster_id"], how="inner")
    assert len(merged) == len(sample_session_features)


def test_export_outputs_writes_data_and_models_metrics(
    tmp_path: Path, sample_session_features: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "data" / "outputs"

    monkeypatch.setattr(clustering, "MODELS_DIR", model_dir)

    clusterer = SessionClusterer(random_state=7)
    assignments = clusterer.fit_clustering(sample_session_features, k=3)

    manual_labels = pd.DataFrame(
        {
            "cluster_id": sorted(assignments["cluster_id"].unique().tolist()),
            "intent_label": ["segment_a", "segment_b", "segment_c"],
        }
    )
    clusterer.apply_manual_labels(manual_labels)

    exports = clusterer.export_outputs(output_dir)

    assert exports["cluster_assignments"].exists()
    assert exports["cluster_summary"].exists()
    assert exports["cluster_representatives"].exists()
    assert exports["labeled_sessions"].exists()

    assert exports["clustering_metrics"].exists()
    assert exports["clustering_metrics"].name == "clustering_metrics.json"
    assert exports["clustering_metrics"].parent.name == "models"


def test_apply_auto_labels_raises_when_purchase_rate_missing(
    sample_session_features: pd.DataFrame,
) -> None:
    clusterer = SessionClusterer(random_state=7)
    clusterer.fit_clustering(sample_session_features, k=3)

    with pytest.raises(ValueError, match="purchase_rate"):
        clusterer.apply_auto_labels(high_threshold=0.8, low_threshold=0.2)


def test_fit_clustering_ignores_non_whitelisted_numeric_features(
    sample_session_features: pd.DataFrame,
) -> None:
    clusterer = SessionClusterer(random_state=7)
    features_with_noise = sample_session_features.copy()
    features_with_noise["new_numeric_noise"] = list(range(len(features_with_noise)))

    clusterer.fit_clustering(features_with_noise, k=3)

    assert "new_numeric_noise" not in clusterer.feature_columns
    ignored = clusterer.metrics.get("ignored_extra_numeric_features", [])
    assert "new_numeric_noise" in ignored
