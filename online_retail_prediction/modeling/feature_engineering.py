from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_REQUIRED_COLUMNS = [
    "session_id",
    "order",
    "price",
    "higher_than_average",
    "page_2_model",
    "main_category",
    "colour",
    "page",
]


def _to_snake_case(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace("(", "").replace(")", "")
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace("__", "_")
    return normalized


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {_col: _to_snake_case(_col) for _col in df.columns}
    output = df.rename(columns=renamed).copy()

    if "session_id" not in output.columns and "sessionid" in output.columns:
        output = output.rename(columns={"sessionid": "session_id"})

    if "page_2_clothing_model" in output.columns and "page_2_model" not in output.columns:
        output = output.rename(columns={"page_2_clothing_model": "page_2_model"})

    if "higher_than_average" not in output.columns and "price_2" in output.columns:
        output["higher_than_average"] = output["price_2"].map({1: 1, 2: 0})

    if "main_category" not in output.columns and "page_1_main_category" in output.columns:
        category_mapping = {1: "trousers", 2: "skirts", 3: "blouses", 4: "sale"}
        output["main_category"] = output["page_1_main_category"].map(category_mapping)

    return output


def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _prepare_clickstream(df: pd.DataFrame) -> pd.DataFrame:
    prepared = _normalize_columns(df)
    _validate_columns(prepared, DEFAULT_REQUIRED_COLUMNS)

    prepared["session_id"] = pd.to_numeric(prepared["session_id"], errors="coerce")
    prepared["order"] = pd.to_numeric(prepared["order"], errors="coerce")
    prepared["price"] = pd.to_numeric(prepared["price"], errors="coerce")
    prepared["higher_than_average"] = pd.to_numeric(
        prepared["higher_than_average"], errors="coerce"
    )

    cleaned = prepared.dropna(subset=["session_id", "order", "price", "higher_than_average"])
    cleaned["session_id"] = cleaned["session_id"].astype(int)
    cleaned["order"] = cleaned["order"].astype(int)
    cleaned["higher_than_average"] = cleaned["higher_than_average"].astype(int)

    return cleaned.sort_values(["session_id", "order"]).reset_index(drop=True)


def _category_entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True)
    if probabilities.empty:
        return 0.0
    entropy = -(probabilities * np.log2(probabilities)).sum()
    return float(entropy)


def build_session_features(clickstream: pd.DataFrame, n_clicks: int = 5) -> pd.DataFrame:
    """
    Build one feature row per session from the first n clicks only.

    Args:
        clickstream: Click-level dataframe.
        n_clicks: Number of initial clicks to use for features.

    Returns:
        Session-level feature dataframe indexed by session_id.
    """

    if n_clicks <= 0:
        raise ValueError("n_clicks must be greater than 0")

    prepared = _prepare_clickstream(clickstream)
    first_n = prepared.groupby("session_id", group_keys=False).head(n_clicks).copy()

    grouped = first_n.groupby("session_id")
    features = grouped.agg(
        n_clicks_observed=("order", "count"),
        n_unique_pages=("page", "nunique"),
        n_unique_models=("page_2_model", "nunique"),
        n_unique_categories=("main_category", "nunique"),
        n_unique_colours=("colour", "nunique"),
        price_mean=("price", "mean"),
        price_min=("price", "min"),
        price_max=("price", "max"),
        price_std=("price", "std"),
        high_price_share_first_n=("higher_than_average", "mean"),
        high_price_count_first_n=("higher_than_average", "sum"),
    )

    features["price_std"] = features["price_std"].fillna(0.0)

    category_entropy = grouped["main_category"].apply(_category_entropy).rename("category_entropy")
    features = features.join(category_entropy)

    category_shares = (
        grouped["main_category"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .add_prefix("category_share_")
    )
    features = features.join(category_shares)
    features["top_category_share"] = category_shares.max(axis=1).fillna(0.0)

    model_freq = first_n["page_2_model"].value_counts(normalize=True)
    first_n["model_frequency"] = first_n["page_2_model"].map(model_freq).fillna(0.0)
    model_features = grouped["page_2_model"].agg(
        first_model=(lambda x: x.iloc[0]),
        last_model=(lambda x: x.iloc[-1]),
    )
    model_frequency_stats = first_n.groupby("session_id").agg(
        mean_model_frequency=("model_frequency", "mean"),
        max_model_frequency=("model_frequency", "max"),
        min_model_frequency=("model_frequency", "min"),
    )
    features = features.join(model_frequency_stats)

    last_click = grouped.tail(1).set_index("session_id")
    features["last_page"] = last_click["page"]
    features["last_location"] = last_click["location"]

    category_freq = first_n["main_category"].value_counts(normalize=True)
    colour_freq = first_n["colour"].value_counts(normalize=True)
    model_freq_full = first_n["page_2_model"].value_counts(normalize=True)

    features["first_model_frequency"] = (
        model_features["first_model"].map(model_freq_full).fillna(0.0)
    )
    features["last_model_frequency"] = (
        model_features["last_model"].map(model_freq_full).fillna(0.0)
    )
    features["last_category_frequency"] = (
        last_click["main_category"].map(category_freq).fillna(0.0)
    )
    features["last_colour_frequency"] = last_click["colour"].map(colour_freq).fillna(0.0)

    transitions = first_n.groupby("session_id")["main_category"].apply(
        lambda values: int(values.ne(values.shift()).sum() - 1)
    )
    features["category_transition_count"] = transitions.clip(lower=0)

    return features.reset_index()
