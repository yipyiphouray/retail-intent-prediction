"""Tests for the Streamlit labeling service helpers."""

from pathlib import Path

import pandas as pd
import pytest

from online_retail_prediction.modeling import clustering
from online_retail_prediction.modeling.cluster_labeling_service import (
    run_clustering_and_load_summary,
    save_cluster_labels,
)


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
        }
    )


def test_run_clustering_and_load_summary_creates_summary(
    tmp_path: Path,
    sample_session_features: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features_path = tmp_path / "features.csv"
    output_dir = tmp_path / "cluster_outputs"
    monkeypatch.setattr(clustering, "MODELS_DIR", tmp_path / "models")
    sample_session_features.to_csv(features_path, index=False)

    summary_df = run_clustering_and_load_summary(features_path=features_path, output_dir=output_dir)

    summary_path = output_dir / "cluster_summary.csv"
    assert summary_path.exists()
    assert {"cluster_id", "cluster_size", "cluster_percentage"}.issubset(summary_df.columns)
    assert any(column.startswith("mean_") for column in summary_df.columns)


def test_save_cluster_labels_rejects_missing_columns(tmp_path: Path) -> None:
    labels_df = pd.DataFrame({"cluster_id": [0, 1, 2]})

    with pytest.raises(ValueError, match="required columns"):
        save_cluster_labels(labels_df=labels_df, output_path=tmp_path / "cluster_label.csv")


def test_save_cluster_labels_rejects_invalid_label_values(tmp_path: Path) -> None:
    labels_df = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "intent_label": ["low-intent", "medium-intent"],
        }
    )

    with pytest.raises(ValueError, match="Invalid intent_label values"):
        save_cluster_labels(labels_df=labels_df, output_path=tmp_path / "cluster_label.csv")


def test_save_cluster_labels_writes_sorted_schema(tmp_path: Path) -> None:
    labels_df = pd.DataFrame(
        {
            "cluster_id": [3, 1, 2],
            "intent_label": ["low-intent", "high-intent", "high-intent"],
        }
    )
    output_path = tmp_path / "cluster_label.csv"

    saved_path = save_cluster_labels(labels_df=labels_df, output_path=output_path)

    assert saved_path == output_path
    saved_df = pd.read_csv(saved_path)
    assert saved_df.columns.tolist() == ["cluster_id", "intent_label"]
    assert saved_df["cluster_id"].tolist() == [1, 2, 3]
    assert saved_df["intent_label"].tolist() == ["high-intent", "high-intent", "low-intent"]
