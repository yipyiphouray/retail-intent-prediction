from __future__ import annotations

from pathlib import Path

import pandas as pd

from online_retail_prediction.modeling.clustering import SessionClusterer

ALLOWED_INTENT_LABELS = ("low-intent", "high-intent")
_REQUIRED_LABEL_COLUMNS = {"cluster_id", "intent_label"}
_REQUIRED_SUMMARY_COLUMNS = {"cluster_id", "cluster_size", "cluster_percentage"}


def run_clustering_and_load_summary(
    features_path: Path,
    output_dir: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run clustering from a session-features CSV, export artifacts, and load summary."""
    if not features_path.exists():
        raise FileNotFoundError(f"Session features file not found: {features_path}")

    session_features_df = pd.read_csv(features_path)
    clusterer = SessionClusterer(random_state=random_state)
    clusterer.fit_clustering(session_features_df=session_features_df)
    exports = clusterer.export_outputs(path=output_dir)

    summary_path = exports["cluster_summary"]
    if not summary_path.exists():
        raise FileNotFoundError(f"Cluster summary file not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    missing_columns = _REQUIRED_SUMMARY_COLUMNS.difference(summary_df.columns)
    if missing_columns:
        raise ValueError(f"Cluster summary is missing required columns: {sorted(missing_columns)}")
    return summary_df


def save_cluster_labels(labels_df: pd.DataFrame, output_path: Path) -> Path:
    """Validate and persist cluster-level labels to cluster_label.csv."""
    missing_columns = _REQUIRED_LABEL_COLUMNS.difference(labels_df.columns)
    if missing_columns:
        raise ValueError(f"labels_df is missing required columns: {sorted(missing_columns)}")

    clean_labels = labels_df[["cluster_id", "intent_label"]].copy()
    clean_labels["cluster_id"] = pd.to_numeric(clean_labels["cluster_id"], errors="coerce")
    if clean_labels["cluster_id"].isna().any():
        raise ValueError("cluster_id must be a valid integer for every row.")
    clean_labels["cluster_id"] = clean_labels["cluster_id"].astype(int)

    if clean_labels["cluster_id"].duplicated().any():
        raise ValueError("cluster_id must be unique (one row per cluster).")

    clean_labels["intent_label"] = clean_labels["intent_label"].astype("string").str.strip()
    if clean_labels["intent_label"].isna().any():
        raise ValueError("intent_label is required for every cluster.")

    invalid_labels = sorted(
        set(clean_labels["intent_label"].tolist()).difference(ALLOWED_INTENT_LABELS)
    )
    if invalid_labels:
        raise ValueError(
            "Invalid intent_label values found: "
            f"{invalid_labels}. Allowed values: {list(ALLOWED_INTENT_LABELS)}"
        )

    clean_labels = clean_labels.sort_values("cluster_id").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_labels.to_csv(output_path, index=False)
    return output_path
