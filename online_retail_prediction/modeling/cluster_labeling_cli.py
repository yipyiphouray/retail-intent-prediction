from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from online_retail_prediction.config import DATA_DIR, PROCESSED_DATA_DIR
from online_retail_prediction.modeling.cluster_config import (
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_DROP_CORRELATED,
    DEFAULT_DROP_LOW_VARIANCE,
    DEFAULT_MIN_VARIANCE,
)
from online_retail_prediction.modeling.clustering import SessionClusterer

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features_full_session.csv",
    random_state: int = 42,
    output_dir: Path = DATA_DIR / "cluster_outputs",
    representatives_top_n: int = 5,
    manual_labels_path: Path | None = None,
    label_column: str = "intent_label",
    use_auto_labels: bool = False,
    high_threshold: float = 0.8,
    low_threshold: float = 0.2,
    drop_correlated: bool = DEFAULT_DROP_CORRELATED,
    drop_low_variance: bool = DEFAULT_DROP_LOW_VARIANCE,
    min_variance: float = DEFAULT_MIN_VARIANCE,
):
    """Run clustering + session label propagation and export outputs."""
    logger.info("Loading session features from {}", features_path)
    session_features_df = pd.read_csv(features_path)
    logger.info(
        "Clustering config: k={}, correlation_threshold={}, drop_correlated={}, drop_low_variance={}, min_variance={}",
        DEFAULT_CLUSTER_COUNT,
        DEFAULT_CORRELATION_THRESHOLD,
        drop_correlated,
        drop_low_variance,
        min_variance,
    )

    clusterer = SessionClusterer(random_state=random_state)
    assignments = clusterer.fit_clustering(
        session_features_df=session_features_df,
        k=DEFAULT_CLUSTER_COUNT,
        drop_correlated=drop_correlated,
        correlation_threshold=DEFAULT_CORRELATION_THRESHOLD,
        drop_low_variance=drop_low_variance,
        min_variance=min_variance,
    )

    logger.info(
        "Cluster assignments created for {} sessions across {} clusters.",
        len(assignments),
        assignments["cluster_id"].nunique(),
    )
    logger.info(
        "Features used: {} (dropped_correlated={}, dropped_low_variance={}).",
        clusterer.metrics.get("n_features_used"),
        len(clusterer.metrics.get("dropped_correlated_features", [])),
        len(clusterer.metrics.get("dropped_low_variance_features", [])),
    )

    summary = clusterer.get_cluster_summary()
    logger.info("Cluster summary created with {} clusters.", len(summary))

    representatives = clusterer.get_representatives(top_n=representatives_top_n)
    logger.info(
        "Selected {} representative sessions using top_n={}",
        len(representatives),
        representatives_top_n,
    )

    if manual_labels_path is not None:
        manual_labels_df = pd.read_csv(manual_labels_path)
        if "cluster_id" not in manual_labels_df.columns:
            raise typer.BadParameter("manual labels file must include 'cluster_id'.")
        if label_column not in manual_labels_df.columns:
            raise typer.BadParameter(f"manual labels file must include '{label_column}'.")

        cluster_labels_df = manual_labels_df[["cluster_id", label_column]].rename(
            columns={label_column: "intent_label"}
        )
        labeled_sessions = clusterer.apply_manual_labels(cluster_labels_df)
        logger.info(
            "Applied manual labels from {} to {} sessions.",
            manual_labels_path,
            len(labeled_sessions),
        )
    elif use_auto_labels:
        labeled_sessions = clusterer.apply_auto_labels(
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
        logger.info("Applied automatic cluster labels to {} sessions.", len(labeled_sessions))
    else:
        fallback_labels = pd.DataFrame(
            {
                "cluster_id": summary["cluster_id"],
                "intent_label": "unlabeled",
            }
        )
        labeled_sessions = clusterer.apply_manual_labels(fallback_labels)
        logger.info(
            "No manual/auto labels provided. Exporting {} sessions with default 'unlabeled'.",
            len(labeled_sessions),
        )

    exports = clusterer.export_outputs(path=output_dir)
    logger.success("Exported clustering outputs: {}", exports)


if __name__ == "__main__":
    app()
