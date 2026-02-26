from __future__ import annotations

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


def _running_nunique(values: pd.Series) -> pd.Series:
    seen: set[str] = set()
    counts: list[int] = []
    for value in values:
        seen.add(value)
        counts.append(len(seen))
    return pd.Series(counts, index=values.index)


def build_click_level_features(clickstream: pd.DataFrame, n_clicks: int = 5) -> pd.DataFrame:
    """
    Build one feature row per click using only each session's first n clicks.

    Args:
        clickstream: Click-level dataframe.
        n_clicks: Number of initial clicks per session to keep.

    Returns:
        Click-level feature dataframe.
    """

    if n_clicks <= 0:
        raise ValueError("n_clicks must be greater than 0")

    prepared = _prepare_clickstream(clickstream)
    click_features = prepared.groupby("session_id", group_keys=False).head(n_clicks).copy()

    grouped = click_features.groupby("session_id")
    click_features["click_position"] = grouped.cumcount() + 1
    click_features["n_clicks_observed"] = grouped["order"].transform("count")
    click_features["is_first_click"] = (click_features["click_position"] == 1).astype(int)
    click_features["is_last_observed_click"] = (
        click_features["click_position"] == click_features["n_clicks_observed"]
    ).astype(int)

    click_features["running_high_price_count"] = grouped["higher_than_average"].cumsum()
    click_features["running_high_price_share"] = (
        click_features["running_high_price_count"] / click_features["click_position"]
    )

    click_features["running_unique_pages"] = grouped["page"].transform(_running_nunique)
    click_features["running_unique_models"] = grouped["page_2_model"].transform(_running_nunique)
    click_features["running_unique_categories"] = grouped["main_category"].transform(
        _running_nunique
    )
    click_features["running_unique_colours"] = grouped["colour"].transform(_running_nunique)

    click_features["price_delta_from_prev"] = grouped["price"].diff().fillna(0.0)
    click_features["category_changed_from_prev"] = (
        grouped["main_category"].transform(lambda values: values.ne(values.shift())).astype(int)
    )
    click_features.loc[click_features["click_position"] == 1, "category_changed_from_prev"] = 0

    category_freq = click_features["main_category"].value_counts(normalize=True)
    colour_freq = click_features["colour"].value_counts(normalize=True)
    model_freq = click_features["page_2_model"].value_counts(normalize=True)

    click_features["category_frequency"] = (
        click_features["main_category"].map(category_freq).fillna(0.0)
    )
    click_features["colour_frequency"] = click_features["colour"].map(colour_freq).fillna(0.0)
    click_features["model_frequency"] = click_features["page_2_model"].map(model_freq).fillna(0.0)

    return click_features.reset_index(drop=True)

