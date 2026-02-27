from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from online_retail_prediction.config import DATA_DIR, PROCESSED_DATA_DIR
from online_retail_prediction.modeling.cluster_labeling_service import (
    ALLOWED_INTENT_LABELS,
    run_clustering_and_load_summary,
    save_cluster_labels,
)

FEATURES_PATH = PROCESSED_DATA_DIR / "features_full_session.csv"
OUTPUT_DIR = DATA_DIR / "cluster_outputs"
LABEL_OUTPUT_PATH = OUTPUT_DIR / "cluster_label.csv"
SELECTED_FEATURES_PATH = OUTPUT_DIR / "selected_features.json"
POSITIONING_MAP_PATH = OUTPUT_DIR / "cluster_positioning_map.png"
ENGAGEMENT_COMPARISON_PATH = OUTPUT_DIR / "engagement_comparison.png"
INTERPRETATIONS_PATH_CANDIDATES = (
    OUTPUT_DIR / "cluster_interpretations.csv",
    OUTPUT_DIR / "clusters_interpretations.csv",
)
SUMMARY_STATE_KEY = "cluster_summary_df"
ERROR_STATE_KEY = "clustering_error"


def _run_clustering_once() -> None:
    summary_df = run_clustering_and_load_summary(
        features_path=FEATURES_PATH,
        output_dir=OUTPUT_DIR,
        random_state=42,
    )
    st.session_state[SUMMARY_STATE_KEY] = summary_df.sort_values("cluster_id").reset_index(drop=True)
    st.session_state[ERROR_STATE_KEY] = None


def _load_selected_feature_names(path: Path) -> tuple[list[str], str | None]:
    if not path.exists():
        return [], f"Selected features file not found: {path}"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [], f"Could not read selected features from {path}: {exc}"

    feature_entries = payload.get("features")
    if not isinstance(feature_entries, list):
        return [], "selected_features.json is missing the `features` list."

    feature_names: list[str] = []
    for entry in feature_entries:
        if isinstance(entry, dict):
            feature_name = entry.get("feature")
            if isinstance(feature_name, str) and feature_name:
                feature_names.append(feature_name)

    if not feature_names:
        return [], "No selected features were found in selected_features.json."

    return feature_names, None


def _resolve_interpretations_path() -> Path | None:
    for candidate in INTERPRETATIONS_PATH_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _load_cluster_interpretations(path: Path | None) -> tuple[pd.DataFrame, str | None]:
    columns = ["cluster_id", "cluster_label", "cluster_size_pct", "business_intent"]
    if path is None:
        return pd.DataFrame(columns=columns), (
            "Cluster interpretations CSV not found. "
            "Expected one of: cluster_interpretations.csv, clusters_interpretations.csv."
        )

    try:
        interpretations_df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(columns=columns), f"Could not read cluster interpretations from {path}: {exc}"

    missing_columns = set(columns).difference(interpretations_df.columns)
    if missing_columns:
        return pd.DataFrame(columns=columns), (
            f"Cluster interpretations file is missing columns: {sorted(missing_columns)}"
        )

    interpretations_df = interpretations_df[columns].copy()
    interpretations_df["cluster_id"] = pd.to_numeric(
        interpretations_df["cluster_id"], errors="coerce"
    )
    interpretations_df = interpretations_df.dropna(subset=["cluster_id"])
    interpretations_df["cluster_id"] = interpretations_df["cluster_id"].astype(int)
    interpretations_df = interpretations_df.sort_values("cluster_id").reset_index(drop=True)
    return interpretations_df, None


def _build_selected_feature_summary(
    summary_df: pd.DataFrame, selected_feature_names: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    base_columns = [
        col
        for col in ("cluster_id", "cluster_size", "cluster_percentage")
        if col in summary_df.columns
    ]

    selected_columns: list[str] = []
    for feature_name in selected_feature_names:
        candidate_columns = (f"mean_{feature_name}", feature_name)
        selected_column = next(
            (
                column
                for column in candidate_columns
                if column in summary_df.columns and column not in selected_columns
            ),
            None,
        )
        if selected_column:
            selected_columns.append(selected_column)

    if selected_columns:
        return summary_df[base_columns + selected_columns].copy(), selected_columns

    return summary_df.copy(), []


def _build_cluster_descriptions(
    summary_df: pd.DataFrame, interpretations_df: pd.DataFrame
) -> pd.DataFrame:
    description_df = summary_df[["cluster_id", "cluster_percentage"]].copy()
    if not interpretations_df.empty:
        description_df = description_df.merge(
            interpretations_df,
            on="cluster_id",
            how="left",
        )
    else:
        description_df["cluster_label"] = pd.NA
        description_df["cluster_size_pct"] = pd.NA
        description_df["business_intent"] = pd.NA

    description_df["cluster_label"] = description_df["cluster_label"].fillna(
        description_df["cluster_id"].map(lambda cluster_id: f"Cluster {int(cluster_id)}")
    )
    description_df["cluster_size_pct"] = description_df["cluster_size_pct"].fillna(
        description_df["cluster_percentage"].map(lambda percentage: f"{percentage:.1f}%")
    )
    description_df["business_intent"] = description_df["business_intent"].fillna(
        "Description unavailable."
    )
    return description_df.sort_values("cluster_id").reset_index(drop=True)


def _estimate_description_card_height(description_df: pd.DataFrame) -> int:
    """Estimate a fixed card height that fits the longest description in a 2-column layout."""
    if description_df.empty:
        return 220

    max_intent_length = int(description_df["business_intent"].astype(str).str.len().max())
    # Conservative approximation for text wrapping in two columns.
    estimated_lines = (max_intent_length // 58) + 1
    return max(220, min(420, 120 + estimated_lines * 22))


def _plot_engagement_comparison(summary_df: pd.DataFrame) -> Figure:
    required_columns = {
        "cluster_id",
        "mean_n_clicks_observed",
        "mean_n_unique_pages",
        "mean_price_mean",
        "mean_category_entropy",
    }
    missing_columns = sorted(required_columns.difference(summary_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    clusters = summary_df.sort_values("cluster_id").reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].bar(clusters["cluster_id"], clusters["mean_n_clicks_observed"], color="coral", alpha=0.7)
    axes[0, 0].set_xlabel("Cluster ID")
    axes[0, 0].set_ylabel("Mean Clicks")
    axes[0, 0].set_title("Average Clicks per Session")
    axes[0, 0].grid(axis="y", alpha=0.3)

    axes[0, 1].bar(clusters["cluster_id"], clusters["mean_n_unique_pages"], color="skyblue", alpha=0.7)
    axes[0, 1].set_xlabel("Cluster ID")
    axes[0, 1].set_ylabel("Mean Unique Pages")
    axes[0, 1].set_title("Average Unique Pages Visited")
    axes[0, 1].grid(axis="y", alpha=0.3)

    axes[1, 0].bar(clusters["cluster_id"], clusters["mean_price_mean"], color="lightgreen", alpha=0.7)
    axes[1, 0].set_xlabel("Cluster ID")
    axes[1, 0].set_ylabel("Mean Price (€)")
    axes[1, 0].set_title("Average Price of Products Viewed")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(
        clusters["cluster_id"], clusters["mean_category_entropy"], color="plum", alpha=0.7
    )
    axes[1, 1].set_xlabel("Cluster ID")
    axes[1, 1].set_ylabel("Category Entropy")
    axes[1, 1].set_title("Category Diversity (Entropy)")
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_positioning_map(
    summary_df: pd.DataFrame, cluster_labels: dict[int, str]
) -> Figure:
    required_columns = {
        "cluster_id",
        "cluster_percentage",
        "mean_n_clicks_observed",
        "mean_price_mean",
    }
    missing_columns = sorted(required_columns.difference(summary_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    clusters = summary_df.sort_values("cluster_id").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        clusters["mean_n_clicks_observed"],
        clusters["mean_price_mean"],
        s=clusters["cluster_percentage"] * 50,
        c=clusters["cluster_id"],
        cmap="tab10",
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
    )

    for row in clusters.itertuples(index=False):
        cluster_id = int(row.cluster_id)
        label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        ax.annotate(
            f"{cluster_id}: {label}",
            (row.mean_n_clicks_observed, row.mean_price_mean),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )

    ax.set_xlabel("Average Clicks per Session (Engagement)", fontsize=12)
    ax.set_ylabel("Average Price Viewed (€)", fontsize=12)
    ax.set_title(
        "Customer Segment Positioning: Engagement vs Price Interest\n"
        "(Bubble size = % of sessions)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, label="Cluster ID")
    ax.axhline(
        y=clusters["mean_price_mean"].median(),
        color="gray",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
    )
    ax.axvline(
        x=clusters["mean_n_clicks_observed"].median(),
        color="gray",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
    )
    fig.tight_layout()
    return fig


def _render_plot_with_fallback(
    plot_func: Callable[[], Figure],
    fallback_png_path: Path,
) -> None:
    try:
        fig = plot_func()
    except Exception as exc:  # noqa: BLE001
        if fallback_png_path.exists():
            st.warning(
                f"Could not render plot in Streamlit ({exc}). Showing saved image: {fallback_png_path.name}"
            )
            st.image(str(fallback_png_path), use_container_width=True)
            return
        st.error(f"Could not render plot and fallback image was not found: {exc}")
        return

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _normalize_intent_labels(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    normalized = normalized.str.replace("_", "-", regex=False)
    normalized = normalized.str.replace(" ", "-", regex=False)
    normalized = normalized.str.replace("‑", "-", regex=False)
    normalized = normalized.replace({"": pd.NA, "nan": pd.NA, "<na>": pd.NA})
    return normalized


def _load_existing_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    labels_df = pd.read_csv(path)
    required = {"cluster_id", "intent_label"}
    if not required.issubset(labels_df.columns):
        return {}

    labels_df = labels_df[["cluster_id", "intent_label"]].copy()
    labels_df["cluster_id"] = pd.to_numeric(labels_df["cluster_id"], errors="coerce")
    labels_df = labels_df.dropna(subset=["cluster_id"])
    labels_df["cluster_id"] = labels_df["cluster_id"].astype(int)
    labels_df["intent_label"] = _normalize_intent_labels(labels_df["intent_label"])
    labels_df = labels_df[labels_df["intent_label"].isin(ALLOWED_INTENT_LABELS)]
    return dict(zip(labels_df["cluster_id"], labels_df["intent_label"]))


def _build_label_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    label_map = _load_existing_labels(LABEL_OUTPUT_PATH)
    label_df = summary_df[["cluster_id"]].copy()
    label_df["intent_label"] = label_df["cluster_id"].map(label_map)
    return label_df


def _render_label_inputs(summary_df: pd.DataFrame) -> pd.DataFrame:
    initial_labels = _build_label_frame(summary_df)
    records: list[dict[str, int | str | None]] = []

    for row in initial_labels.itertuples(index=False):
        cluster_id = int(row.cluster_id)
        default_value = row.intent_label if isinstance(row.intent_label, str) else None
        key = f"cluster_label_{cluster_id}"

        if key not in st.session_state:
            st.session_state[key] = default_value

        cluster_col, label_col = st.columns([1, 2], gap="small")
        with cluster_col:
            st.caption(f"Cluster {cluster_id}")
        with label_col:
            selected_label = st.selectbox(
                label=f"label_{cluster_id}",
                options=list(ALLOWED_INTENT_LABELS),
                index=None,
                placeholder="Select label",
                key=key,
                label_visibility="collapsed",
            )
        records.append({"cluster_id": cluster_id, "intent_label": selected_label})

    return pd.DataFrame(records)


def main() -> None:
    st.set_page_config(page_title="Cluster Labeling", layout="wide")
    st.title("Cluster Labeling")

    if SUMMARY_STATE_KEY not in st.session_state and ERROR_STATE_KEY not in st.session_state:
        with st.spinner("Running clustering..."):
            try:
                _run_clustering_once()
            except Exception as exc:  # noqa: BLE001
                st.session_state[ERROR_STATE_KEY] = str(exc)

    if st.button("Refresh clustering"):
        with st.spinner("Running clustering..."):
            try:
                _run_clustering_once()
            except Exception as exc:  # noqa: BLE001
                st.session_state[ERROR_STATE_KEY] = str(exc)

    error_message = st.session_state.get(ERROR_STATE_KEY)
    if error_message:
        st.error(error_message)
        st.stop()

    summary_df = st.session_state.get(SUMMARY_STATE_KEY)
    if summary_df is None:
        st.error(f"Clustering output was not generated from: {FEATURES_PATH}")
        st.stop()

    selected_feature_names, selected_features_error = _load_selected_feature_names(
        SELECTED_FEATURES_PATH
    )
    selected_summary_df, selected_summary_columns = _build_selected_feature_summary(
        summary_df=summary_df,
        selected_feature_names=selected_feature_names,
    )

    interpretations_path = _resolve_interpretations_path()
    interpretations_df, interpretations_error = _load_cluster_interpretations(
        path=interpretations_path
    )
    cluster_descriptions_df = _build_cluster_descriptions(
        summary_df=summary_df,
        interpretations_df=interpretations_df,
    )
    description_card_height = _estimate_description_card_height(cluster_descriptions_df)

    table_height = min(700, max(360, 45 + len(summary_df) * 35))
    label_panel_height = table_height
    visual_panel_height = 560

    left_col, right_col = st.columns([5, 2], gap="large")
    with left_col:
        st.subheader("Cluster Summary (Selected Features)")
        st.dataframe(
            selected_summary_df,
            hide_index=True,
            use_container_width=True,
            height=table_height,
        )

        if selected_features_error:
            st.warning(selected_features_error)
        elif selected_summary_columns:
            shown_features = [column.removeprefix("mean_") for column in selected_summary_columns]
            st.caption(f"Features shown from selected_features.json: {', '.join(shown_features)}")
        else:
            st.info("No selected feature columns matched the summary. Showing available columns.")

    with right_col:
        st.subheader("Labels")
        with st.container(height=label_panel_height, border=True):
            labels_df = _render_label_inputs(summary_df=summary_df)

    st.subheader("Cluster Descriptions")
    if interpretations_error:
        st.warning(interpretations_error)

    description_records = cluster_descriptions_df.to_dict(orient="records")
    for idx in range(0, len(description_records), 2):
        row_columns = st.columns(2, gap="large")
        row_cards = description_records[idx : idx + 2]

        for col_idx, card in enumerate(row_cards):
            with row_columns[col_idx]:
                with st.container(height=description_card_height, border=True):
                    cluster_id = int(card["cluster_id"])
                    cluster_label = str(card["cluster_label"])
                    cluster_size_pct = str(card["cluster_size_pct"])
                    business_intent = str(card["business_intent"])

                    st.markdown(f"**Cluster {cluster_id}: {cluster_label}**")
                    st.caption(f"Segment size: {cluster_size_pct}")
                    st.write(business_intent)

    st.subheader("Visual Diagnostics")
    positioning_col, engagement_col = st.columns(2, gap="large")

    label_map = {
        int(row.cluster_id): str(row.cluster_label)
        for row in cluster_descriptions_df[["cluster_id", "cluster_label"]].itertuples(index=False)
    }

    with positioning_col:
        with st.container(height=visual_panel_height, border=True):
            st.markdown("**Positioning Map**")
            _render_plot_with_fallback(
                plot_func=lambda: _plot_positioning_map(
                    summary_df=summary_df, cluster_labels=label_map
                ),
                fallback_png_path=POSITIONING_MAP_PATH,
            )

    with engagement_col:
        with st.container(height=visual_panel_height, border=True):
            st.markdown("**Engagement Level Comparison**")
            _render_plot_with_fallback(
                plot_func=lambda: _plot_engagement_comparison(summary_df=summary_df),
                fallback_png_path=ENGAGEMENT_COMPARISON_PATH,
            )

    labels_df = labels_df[["cluster_id", "intent_label"]].copy()
    labels_df["intent_label"] = _normalize_intent_labels(labels_df["intent_label"])
    valid_mask = labels_df["intent_label"].apply(
        lambda value: isinstance(value, str) and value in ALLOWED_INTENT_LABELS
    )
    invalid_mask = ~valid_mask
    all_labels_selected = not invalid_mask.any()

    if not all_labels_selected:
        missing_clusters = labels_df.loc[invalid_mask, "cluster_id"].tolist()
        st.info(f"Select a label for every cluster to enable Save. Missing: {missing_clusters}")

    if st.button("Save", disabled=not all_labels_selected):
        try:
            saved_path = save_cluster_labels(labels_df=labels_df, output_path=LABEL_OUTPUT_PATH)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
        else:
            st.success(f"Saved labels to {saved_path}")


if __name__ == "__main__":
    main()
