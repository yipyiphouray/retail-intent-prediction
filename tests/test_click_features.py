"""Tests for click-level sequential feature enrichment and session-level aggregation."""

import numpy as np
import pandas as pd

from online_retail_prediction.modeling.feature_engineering import (
    _enrich_clicks_sequential,
    _prepare_clickstream,
    build_session_features,
)


def _sample_clickstream() -> pd.DataFrame:
    """Three sessions: 3 clicks, 2 clicks, 1 click."""
    return pd.DataFrame(
        {
            "session ID": [1, 1, 1, 2, 2, 3],
            "order": [1, 2, 3, 1, 2, 1],
            "price": [20, 30, 20, 100, 120, 10],
            "price 2": [2, 1, 2, 1, 1, 2],
            "page 2 (clothing model)": ["A1", "A2", "A1", "B1", "B2", "C1"],
            "main_category": ["trousers", "skirts", "trousers", "sale", "sale", "blouses"],
            "colour": ["black", "black", "red", "white", "white", "blue"],
            "page": [1, 1, 2, 3, 4, 1],
            "location": [1, 2, 3, 4, 5, 6],
        }
    )


# --- _enrich_clicks_sequential tests ---


def test_change_flags_first_click_is_zero() -> None:
    prepared = _prepare_clickstream(_sample_clickstream())
    enriched = _enrich_clicks_sequential(prepared)

    first_clicks = enriched.groupby("session_id").first()
    assert (first_clicks["category_changed"] == 0).all()
    assert (first_clicks["colour_changed"] == 0).all()
    assert (first_clicks["model_changed"] == 0).all()


def test_change_flags_detect_transitions() -> None:
    prepared = _prepare_clickstream(_sample_clickstream())
    enriched = _enrich_clicks_sequential(prepared)

    # Session 1: trousers -> skirts -> trousers (two category changes)
    s1 = enriched[enriched["session_id"] == 1].sort_values("order")
    assert list(s1["category_changed"]) == [0, 1, 1]

    # Session 2: sale -> sale (no category change)
    s2 = enriched[enriched["session_id"] == 2].sort_values("order")
    assert list(s2["category_changed"]) == [0, 0]


def test_price_diff_and_direction() -> None:
    prepared = _prepare_clickstream(_sample_clickstream())
    enriched = _enrich_clicks_sequential(prepared)

    # Session 1: prices 20, 30, 20 -> diffs 0, 10, -10
    s1 = enriched[enriched["session_id"] == 1].sort_values("order")
    assert list(s1["price_diff"]) == [0.0, 10.0, -10.0]
    assert list(s1["price_direction"]) == [0, 1, -1]


def test_cumulative_unique_monotonically_nondecreasing() -> None:
    prepared = _prepare_clickstream(_sample_clickstream())
    enriched = _enrich_clicks_sequential(prepared)

    for _, group in enriched.groupby("session_id"):
        group = group.sort_values("order")
        for col in ["cumul_unique_categories", "cumul_unique_models", "cumul_unique_colours"]:
            values = group[col].tolist()
            assert values == sorted(values), f"{col} not monotonic for session"


def test_cumulative_unique_values() -> None:
    prepared = _prepare_clickstream(_sample_clickstream())
    enriched = _enrich_clicks_sequential(prepared)

    # Session 1: categories trousers, skirts, trousers -> cumul unique: 1, 2, 2
    s1 = enriched[enriched["session_id"] == 1].sort_values("order")
    assert list(s1["cumul_unique_categories"]) == [1, 2, 2]

    # Session 1: models A1, A2, A1 -> cumul unique: 1, 2, 2
    assert list(s1["cumul_unique_models"]) == [1, 2, 2]


def test_revisit_flags() -> None:
    prepared = _prepare_clickstream(_sample_clickstream())
    enriched = _enrich_clicks_sequential(prepared)

    # Session 1: categories trousers, skirts, trousers
    # Third click revisits trousers
    s1 = enriched[enriched["session_id"] == 1].sort_values("order")
    assert list(s1["is_category_revisit"]) == [0, 0, 1]

    # Session 1: models A1, A2, A1 -> third click revisits A1
    assert list(s1["is_model_revisit"]) == [0, 0, 1]


def test_position_encoding_range() -> None:
    prepared = _prepare_clickstream(_sample_clickstream())
    enriched = _enrich_clicks_sequential(prepared)

    assert (enriched["click_position_norm"] >= 0.0).all()
    assert (enriched["click_position_norm"] <= 1.0).all()


def test_position_encoding_single_click_session() -> None:
    prepared = _prepare_clickstream(_sample_clickstream())
    enriched = _enrich_clicks_sequential(prepared)

    # Session 3 has 1 click -> position should be 0.0
    s3 = enriched[enriched["session_id"] == 3]
    assert s3["click_position_norm"].item() == 0.0


def test_position_encoding_multi_click_session() -> None:
    prepared = _prepare_clickstream(_sample_clickstream())
    enriched = _enrich_clicks_sequential(prepared)

    # Session 1 has 3 clicks -> positions should be 0.0, 0.5, 1.0
    s1 = enriched[enriched["session_id"] == 1].sort_values("order")
    np.testing.assert_allclose(s1["click_position_norm"].tolist(), [0.0, 0.5, 1.0])


# --- build_session_features integration tests ---


EXPECTED_SEQUENTIAL_COLUMNS = [
    "category_switch_rate",
    "colour_switch_rate",
    "model_switch_rate",
    "mean_price_diff",
    "std_price_diff",
    "max_abs_price_diff",
    "n_price_increases",
    "n_price_decreases",
    "mean_price_deviation",
    "category_revisit_rate",
    "model_revisit_rate",
    "colour_revisit_rate",
    "exploration_velocity_categories",
    "exploration_velocity_models",
]


def test_session_features_contain_sequential_columns() -> None:
    clickstream = _sample_clickstream()
    features = build_session_features(clickstream=clickstream, n_clicks=5)

    for col in EXPECTED_SEQUENTIAL_COLUMNS:
        assert col in features.columns, f"Missing column: {col}"


def test_single_click_session_sequential_features() -> None:
    clickstream = _sample_clickstream()
    features = build_session_features(clickstream=clickstream, n_clicks=5)

    s3 = features[features["session_id"] == 3].iloc[0]
    # Single-click: all switch rates and revisit rates should be 0
    assert s3["category_switch_rate"] == 0.0
    assert s3["colour_switch_rate"] == 0.0
    assert s3["model_switch_rate"] == 0.0
    assert s3["category_revisit_rate"] == 0.0
    assert s3["model_revisit_rate"] == 0.0
    assert s3["colour_revisit_rate"] == 0.0
    assert s3["mean_price_diff"] == 0.0
    assert s3["max_abs_price_diff"] == 0.0


def test_session_features_sequential_values() -> None:
    clickstream = _sample_clickstream()
    features = build_session_features(clickstream=clickstream, n_clicks=5)

    # Session 1: 3 clicks, prices 20,30,20 -> 1 increase, 1 decrease
    s1 = features[features["session_id"] == 1].iloc[0]
    assert s1["n_price_increases"] == 1
    assert s1["n_price_decreases"] == 1
    assert s1["max_abs_price_diff"] == 10.0

    # Session 1: categories trousers, skirts, trousers -> 2/3 switch, 1/3 revisit
    np.testing.assert_allclose(s1["category_switch_rate"], 2 / 3, rtol=1e-6)
    np.testing.assert_allclose(s1["category_revisit_rate"], 1 / 3, rtol=1e-6)


def test_full_session_mode_uses_all_clicks_when_more_than_n() -> None:
    clickstream = _sample_clickstream()

    features_first_n = build_session_features(
        clickstream=clickstream,
        n_clicks=2,
        aggregation_mode="first_n",
    )
    features_full = build_session_features(
        clickstream=clickstream,
        aggregation_mode="full_session",
    )

    s1_first_n = features_first_n.loc[features_first_n["session_id"] == 1, "n_clicks_observed"].item()
    s1_full = features_full.loc[features_full["session_id"] == 1, "n_clicks_observed"].item()

    assert s1_first_n == 2
    assert s1_full == 3

def test_full_session_mode_excludes_sequential_aggregation_columns() -> None:
    """Verify that sequential aggregation features are only in first_n mode."""
    clickstream = _sample_clickstream()

    features_first_n = build_session_features(
        clickstream=clickstream,
        aggregation_mode="first_n",
    )
    features_full = build_session_features(
        clickstream=clickstream,
        aggregation_mode="full_session",
    )

    # First_n should have all sequential columns
    for col in EXPECTED_SEQUENTIAL_COLUMNS:
        assert col in features_first_n.columns, f"Missing in first_n: {col}"

    # Full_session should NOT have any sequential columns
    for col in EXPECTED_SEQUENTIAL_COLUMNS:
        assert col not in features_full.columns, f"Should not be in full_session: {col}"