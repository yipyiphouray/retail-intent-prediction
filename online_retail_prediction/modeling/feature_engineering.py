from __future__ import annotations

import numpy as np
import pandas as pd

VALID_AGGREGATION_MODES = {"first_n", "full_session"}

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


def _cumul_nunique(series: pd.Series) -> pd.Series:
    """Return the cumulative count of unique values seen so far (inclusive)."""
    seen: set = set()
    counts = []
    for val in series:
        seen.add(val)
        counts.append(len(seen))
    return pd.Series(counts, index=series.index, dtype=int)


def _is_revisit(series: pd.Series) -> pd.Series:
    """Return 1 if the value has appeared earlier in the series, else 0."""
    seen: set = set()
    flags = []
    for val in series:
        flags.append(1 if val in seen else 0)
        seen.add(val)
    return pd.Series(flags, index=series.index, dtype=int)


def _enrich_clicks_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """Add click-level sequential features within each session.

    Expects a DataFrame already sorted by [session_id, order].
    Adds change/transition flags, cumulative stats, revisit indicators,
    and normalised click position.
    """
    enriched = df.copy()
    g = enriched.groupby("session_id", group_keys=False)

    # --- Change / transition flags ---
    enriched["category_changed"] = g["main_category"].transform(
        lambda s: s.ne(s.shift()).astype(int)
    )
    enriched["colour_changed"] = g["colour"].transform(lambda s: s.ne(s.shift()).astype(int))
    enriched["model_changed"] = g["page_2_model"].transform(lambda s: s.ne(s.shift()).astype(int))
    # First click in each session should be 0, not 1
    first_mask = g.cumcount() == 0
    enriched.loc[first_mask, "category_changed"] = 0
    enriched.loc[first_mask, "colour_changed"] = 0
    enriched.loc[first_mask, "model_changed"] = 0

    enriched["price_diff"] = g["price"].transform(lambda s: s.diff().fillna(0.0))
    enriched["price_direction"] = np.sign(enriched["price_diff"]).astype(int)

    # --- Cumulative / running stats ---
    enriched["cumul_unique_categories"] = g["main_category"].transform(_cumul_nunique)
    enriched["cumul_unique_models"] = g["page_2_model"].transform(_cumul_nunique)
    enriched["cumul_unique_colours"] = g["colour"].transform(_cumul_nunique)

    enriched["cumul_mean_price"] = g["price"].transform(lambda s: s.expanding().mean())
    enriched["price_vs_cumul_mean"] = enriched["price"] - enriched["cumul_mean_price"]

    # --- Revisit indicators ---
    enriched["is_category_revisit"] = g["main_category"].transform(_is_revisit)
    enriched["is_model_revisit"] = g["page_2_model"].transform(_is_revisit)
    enriched["is_colour_revisit"] = g["colour"].transform(_is_revisit)

    # --- Position encoding ---
    session_len = g["order"].transform("count")
    click_pos = g.cumcount()  # 0-indexed position within session
    enriched["click_position_norm"] = np.where(
        session_len > 1,
        click_pos / (session_len - 1),
        0.0,
    )

    return enriched


def _category_entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True)
    if probabilities.empty:
        return 0.0
    entropy = -(probabilities * np.log2(probabilities)).sum()
    return float(entropy)


def build_session_features(
    clickstream: pd.DataFrame,
    n_clicks: int = 5,
    aggregation_mode: str = "first_n",
) -> pd.DataFrame:
    """
    Build one feature row per session from either first-N clicks or full session clicks.

    Args:
        clickstream: Click-level dataframe.
        n_clicks: Number of initial clicks to use for features.
        aggregation_mode: Aggregation strategy, one of {"first_n", "full_session"}.

    Returns:
        Session-level feature dataframe indexed by session_id.
    """

    mode = aggregation_mode.strip().lower()
    if mode not in VALID_AGGREGATION_MODES:
        raise ValueError(
            f"Invalid aggregation_mode='{aggregation_mode}'. "
            f"Use one of: {sorted(VALID_AGGREGATION_MODES)}."
        )

    if mode == "first_n" and n_clicks <= 0:
        raise ValueError("n_clicks must be greater than 0 when aggregation_mode='first_n'.")

    prepared = _prepare_clickstream(clickstream)
    selected_clicks = (
        prepared.groupby("session_id", group_keys=False).head(n_clicks).copy()
        if mode == "first_n"
        else prepared.copy()
    )
    selected_clicks = _enrich_clicks_sequential(selected_clicks)

    grouped = selected_clicks.groupby("session_id")
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

    model_freq = selected_clicks["page_2_model"].value_counts(normalize=True)
    selected_clicks["model_frequency"] = (
        selected_clicks["page_2_model"].map(model_freq).fillna(0.0)
    )
    model_features = grouped["page_2_model"].agg(
        first_model=(lambda x: x.iloc[0]),
        last_model=(lambda x: x.iloc[-1]),
    )
    model_frequency_stats = selected_clicks.groupby("session_id").agg(
        mean_model_frequency=("model_frequency", "mean"),
        max_model_frequency=("model_frequency", "max"),
        min_model_frequency=("model_frequency", "min"),
    )
    features = features.join(model_frequency_stats)

    last_click = grouped.tail(1).set_index("session_id")
    features["last_page"] = last_click["page"]
    features["last_location"] = (
        last_click["location"] if "location" in last_click.columns else np.nan
    )

    category_freq = selected_clicks["main_category"].value_counts(normalize=True)
    colour_freq = selected_clicks["colour"].value_counts(normalize=True)
    model_freq_full = selected_clicks["page_2_model"].value_counts(normalize=True)

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

    transitions = selected_clicks.groupby("session_id")["main_category"].apply(
        lambda values: int(values.ne(values.shift()).sum() - 1)
    )
    features["category_transition_count"] = transitions.clip(lower=0)

    # --- Aggregations of sequential click-level features ---
    # Only include for first_n aggregation mode
    if mode == "first_n":
        seq_agg = selected_clicks.groupby("session_id").agg(
            category_switch_rate=("category_changed", "mean"),
            colour_switch_rate=("colour_changed", "mean"),
            model_switch_rate=("model_changed", "mean"),
            mean_price_diff=("price_diff", "mean"),
            std_price_diff=("price_diff", "std"),
            mean_price_deviation=("price_vs_cumul_mean", "mean"),
            category_revisit_rate=("is_category_revisit", "mean"),
            model_revisit_rate=("is_model_revisit", "mean"),
            colour_revisit_rate=("is_colour_revisit", "mean"),
        )
        seq_agg["std_price_diff"] = seq_agg["std_price_diff"].fillna(0.0)

        # max absolute price change across clicks
        seq_agg["max_abs_price_diff"] = (
            selected_clicks.assign(abs_price_diff=lambda d: d["price_diff"].abs())
            .groupby("session_id")["abs_price_diff"]
            .max()
        )

        # counts of price direction
        seq_agg["n_price_increases"] = selected_clicks.groupby("session_id")[
            "price_direction"
        ].apply(lambda s: (s > 0).sum())
        seq_agg["n_price_decreases"] = selected_clicks.groupby("session_id")[
            "price_direction"
        ].apply(lambda s: (s < 0).sum())

        # Exploration velocity: how early diversity was discovered.
        # mean(cumul_unique) / max(cumul_unique) — closer to 1 means early discovery.
        for col, name in [
            ("cumul_unique_categories", "exploration_velocity_categories"),
            ("cumul_unique_models", "exploration_velocity_models"),
        ]:
            col_max = selected_clicks.groupby("session_id")[col].max()
            col_mean = selected_clicks.groupby("session_id")[col].mean()
            seq_agg[name] = np.where(col_max > 0, col_mean / col_max, 0.0)

        features = features.join(seq_agg)

    return features.reset_index()
