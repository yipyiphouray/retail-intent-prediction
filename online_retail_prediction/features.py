from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from online_retail_prediction.config import PROCESSED_DATA_DIR
from online_retail_prediction.modeling.feature_engineering import build_session_features
from online_retail_prediction.modeling.labeling import (
    ExternalPartialLabelStrategy,
    OverrideLabelStrategy,
    ProxyHybridIntentLabelStrategy,
    generate_session_labels,
)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "e-shop clothing 2008.csv",
    features_output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_output_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    delimiter: str = ",",
    n_clicks: int = 5,
    label_mode: str = "proxy_hybrid",
    min_session_clicks: int = 8,
    min_high_price_share: float = 0.5,
    manual_labels_path: Path | None = None,
    cluster_labels_path: Path | None = None,
    external_session_id_column: str = "session_id",
    external_label_column: str = "label",
    external_confidence_column: str | None = None,
):
    """Create first-N click session features and session-level labels."""

    logger.info(f"Loading clickstream from {input_path}...")
    clickstream = pd.read_csv(input_path, sep=delimiter)

    mode = label_mode.strip().lower()
    proxy_strategy = ProxyHybridIntentLabelStrategy(
        min_session_clicks=min_session_clicks,
        min_high_price_share=min_high_price_share,
    )

    if mode == "proxy_hybrid":
        label_strategy = proxy_strategy
    elif mode == "manual_only":
        if manual_labels_path is None:
            raise ValueError("manual_labels_path is required when label_mode='manual_only'.")
        label_strategy = ExternalPartialLabelStrategy(
            labels_path=manual_labels_path,
            label_source="manual",
            session_id_column=external_session_id_column,
            label_column=external_label_column,
            confidence_column=external_confidence_column,
        )
    elif mode == "cluster_only":
        if cluster_labels_path is None:
            raise ValueError("cluster_labels_path is required when label_mode='cluster_only'.")
        label_strategy = ExternalPartialLabelStrategy(
            labels_path=cluster_labels_path,
            label_source="cluster",
            session_id_column=external_session_id_column,
            label_column=external_label_column,
            confidence_column=external_confidence_column,
        )
    elif mode == "manual_over_proxy":
        if manual_labels_path is None:
            raise ValueError("manual_labels_path is required when label_mode='manual_over_proxy'.")
        manual_strategy = ExternalPartialLabelStrategy(
            labels_path=manual_labels_path,
            label_source="manual",
            session_id_column=external_session_id_column,
            label_column=external_label_column,
            confidence_column=external_confidence_column,
        )
        label_strategy = OverrideLabelStrategy(
            base_strategy=proxy_strategy,
            override_strategy=manual_strategy,
        )
    elif mode == "cluster_over_proxy":
        if cluster_labels_path is None:
            raise ValueError(
                "cluster_labels_path is required when label_mode='cluster_over_proxy'."
            )
        cluster_strategy = ExternalPartialLabelStrategy(
            labels_path=cluster_labels_path,
            label_source="cluster",
            session_id_column=external_session_id_column,
            label_column=external_label_column,
            confidence_column=external_confidence_column,
        )
        label_strategy = OverrideLabelStrategy(
            base_strategy=proxy_strategy,
            override_strategy=cluster_strategy,
        )
    else:
        raise ValueError(
            "Invalid label_mode. Use one of: "
            "proxy_hybrid, manual_only, cluster_only, manual_over_proxy, cluster_over_proxy."
        )

    features_df = build_session_features(clickstream=clickstream, n_clicks=n_clicks)
    labels_df = generate_session_labels(
        clickstream=clickstream,
        label_strategy=label_strategy,
        session_ids=features_df["session_id"],
    )

    features_output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_output_path, index=False)
    labels_df.to_csv(labels_output_path, index=False)

    logger.success(f"Features saved to {features_output_path}")
    logger.success(f"Labels saved to {labels_output_path}")


if __name__ == "__main__":
    app()
