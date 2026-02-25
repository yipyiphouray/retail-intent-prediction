"""Tests for Streamlit cluster-labeling UI data preparation helpers."""

import json
from pathlib import Path

import pandas as pd

from apps.cluster_labeling_app import (
    _build_cluster_descriptions,
    _build_selected_feature_summary,
    _load_cluster_interpretations,
    _load_selected_feature_names,
)


def test_load_selected_feature_names_reads_feature_list(tmp_path: Path) -> None:
    selected_features_path = tmp_path / "selected_features.json"
    selected_features_path.write_text(
        json.dumps(
            {
                "features": [
                    {"feature": "n_clicks_observed"},
                    {"feature": "price_mean"},
                ]
            }
        ),
        encoding="utf-8",
    )

    feature_names, error_message = _load_selected_feature_names(selected_features_path)

    assert error_message is None
    assert feature_names == ["n_clicks_observed", "price_mean"]


def test_build_selected_feature_summary_only_includes_selected_columns() -> None:
    summary_df = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "cluster_size": [5, 6],
            "cluster_percentage": [45.0, 55.0],
            "mean_n_clicks_observed": [2.1, 4.2],
            "mean_price_mean": [30.0, 42.0],
            "mean_category_entropy": [0.2, 0.4],
        }
    )

    selected_summary_df, selected_columns = _build_selected_feature_summary(
        summary_df=summary_df,
        selected_feature_names=["n_clicks_observed", "price_mean"],
    )

    assert selected_columns == ["mean_n_clicks_observed", "mean_price_mean"]
    assert selected_summary_df.columns.tolist() == [
        "cluster_id",
        "cluster_size",
        "cluster_percentage",
        "mean_n_clicks_observed",
        "mean_price_mean",
    ]


def test_load_cluster_interpretations_validates_schema(tmp_path: Path) -> None:
    interpretations_path = tmp_path / "cluster_interpretations.csv"
    pd.DataFrame({"cluster_id": [0], "cluster_label": ["Segment A"]}).to_csv(
        interpretations_path, index=False
    )

    interpretations_df, error_message = _load_cluster_interpretations(interpretations_path)

    assert interpretations_df.empty
    assert error_message is not None
    assert "missing columns" in error_message


def test_build_cluster_descriptions_fills_missing_interpretations() -> None:
    summary_df = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "cluster_percentage": [10.0, 20.0],
        }
    )
    interpretations_df = pd.DataFrame(
        {
            "cluster_id": [0],
            "cluster_label": ["Known Segment"],
            "cluster_size_pct": ["10.0%"],
            "business_intent": ["Known description."],
        }
    )

    description_df = _build_cluster_descriptions(summary_df, interpretations_df)

    assert description_df["cluster_label"].tolist() == ["Known Segment", "Cluster 1"]
    assert description_df["cluster_size_pct"].tolist() == ["10.0%", "20.0%"]
    assert description_df["business_intent"].tolist() == [
        "Known description.",
        "Description unavailable.",
    ]
