"""Tests for session feature generation modes and replaceable intent-label strategies."""

from pathlib import Path

import pandas as pd
import pytest

from online_retail_prediction.modeling.feature_engineering import build_session_features
from online_retail_prediction.modeling.labeling import (
    ExternalPartialLabelStrategy,
    OverrideLabelStrategy,
    ProxyHybridIntentLabelStrategy,
    generate_session_labels,
)


def _sample_clickstream() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session ID": [1, 1, 1, 2, 2, 3],
            "order": [1, 2, 3, 1, 2, 1],
            "price": [20, 30, 40, 100, 120, 10],
            "price 2": [2, 1, 1, 1, 1, 2],
            "page 2 (clothing model)": ["A1", "A2", "A3", "B1", "B2", "C1"],
            "main_category": ["trousers", "skirts", "skirts", "sale", "sale", "blouses"],
            "colour": ["black", "black", "red", "white", "white", "blue"],
            "page": [1, 1, 2, 3, 4, 1],
            "location": [1, 2, 3, 4, 5, 6],
        }
    )


def test_build_session_features_uses_first_n_and_keeps_short_sessions() -> None:
    clickstream = _sample_clickstream()

    features = build_session_features(clickstream=clickstream, n_clicks=2)

    assert set(features["session_id"]) == {1, 2, 3}
    assert features.loc[features["session_id"] == 1, "n_clicks_observed"].item() == 2
    assert features.loc[features["session_id"] == 3, "n_clicks_observed"].item() == 1


def test_build_session_features_full_session_uses_all_clicks() -> None:
    clickstream = _sample_clickstream()

    features = build_session_features(
        clickstream=clickstream,
        aggregation_mode="full_session",
    )

    assert set(features["session_id"]) == {1, 2, 3}
    assert features.loc[features["session_id"] == 1, "n_clicks_observed"].item() == 3
    assert features.loc[features["session_id"] == 2, "n_clicks_observed"].item() == 2
    assert features.loc[features["session_id"] == 3, "n_clicks_observed"].item() == 1


def test_build_session_features_rejects_invalid_aggregation_mode() -> None:
    clickstream = _sample_clickstream()

    with pytest.raises(ValueError, match="Invalid aggregation_mode"):
        build_session_features(clickstream=clickstream, aggregation_mode="invalid_mode")


def test_proxy_hybrid_labels_follow_thresholds() -> None:
    clickstream = _sample_clickstream()
    strategy = ProxyHybridIntentLabelStrategy(min_session_clicks=2, min_high_price_share=0.5)

    labels = strategy.generate(clickstream)
    labels = labels.set_index("session_id")

    assert labels.loc[1, "label"] == 1
    assert labels.loc[2, "label"] == 1
    assert labels.loc[3, "label"] == 0


def test_external_partial_labels_keep_unlabeled_sessions(tmp_path: Path) -> None:
    clickstream = _sample_clickstream()
    label_file = tmp_path / "manual_labels.csv"
    pd.DataFrame({"session_id": [1], "label": [1]}).to_csv(label_file, index=False)

    strategy = ExternalPartialLabelStrategy(labels_path=label_file, label_source="manual")
    labels = strategy.generate(clickstream).set_index("session_id")

    assert labels.loc[1, "label"] == 1
    assert pd.isna(labels.loc[2, "label"])
    assert labels.loc[2, "label_source"] == "unlabeled"


def test_override_strategy_replaces_proxy_labels_when_manual_exists(tmp_path: Path) -> None:
    clickstream = _sample_clickstream()
    manual_file = tmp_path / "manual_labels.csv"
    pd.DataFrame({"session_id": [3], "label": [1]}).to_csv(manual_file, index=False)

    proxy = ProxyHybridIntentLabelStrategy(min_session_clicks=2, min_high_price_share=0.5)
    manual = ExternalPartialLabelStrategy(labels_path=manual_file, label_source="manual")
    strategy = OverrideLabelStrategy(base_strategy=proxy, override_strategy=manual)

    features = build_session_features(clickstream=clickstream, n_clicks=2)
    labels = generate_session_labels(
        clickstream=clickstream,
        label_strategy=strategy,
        session_ids=features["session_id"],
    ).set_index("session_id")

    assert labels.loc[1, "label"] == 1
    assert labels.loc[2, "label"] == 1
    assert labels.loc[3, "label"] == 1
    assert labels.loc[3, "label_source"] == "manual"
