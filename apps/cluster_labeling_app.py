from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from online_retail_prediction.config import DATA_DIR, PROCESSED_DATA_DIR
from online_retail_prediction.modeling.cluster_labeling_service import (
    ALLOWED_INTENT_LABELS,
    run_clustering_and_load_summary,
    save_cluster_labels,
)

FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
OUTPUT_DIR = DATA_DIR / "cluster_outputs"
LABEL_OUTPUT_PATH = OUTPUT_DIR / "cluster_label.csv"
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

    table_height = min(700, max(360, 45 + len(summary_df) * 35))
    label_title_block_height = 56
    label_panel_height = max(220, table_height - label_title_block_height)

    left_col, right_col = st.columns([5, 2], gap="large")
    with left_col:
        st.dataframe(summary_df, hide_index=True, use_container_width=True, height=table_height)
    with right_col:
        st.markdown(
            (
                "<div style='margin:0; line-height:1.05; font-size:2.2rem; "
                "font-weight:700;'>Labels</div>"
            ),
            unsafe_allow_html=True,
        )
        with st.container(height=label_panel_height, border=True):
            labels_df = _render_label_inputs(summary_df=summary_df)

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
