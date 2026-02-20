from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from online_retail_prediction.modeling.feature_engineering import (
    _normalize_columns,
    _prepare_clickstream,
)


class IntentLabelStrategy(ABC):
    """Interface for session-level intent label generation from full sessions."""

    @abstractmethod
    def generate(self, clickstream: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with columns: session_id, label, label_source, label_confidence."""


class ProxyHybridIntentLabelStrategy(IntentLabelStrategy):
    """Proxy strategy based on full-session depth and high-price click share."""

    def __init__(self, min_session_clicks: int = 8, min_high_price_share: float = 0.5):
        self.min_session_clicks = min_session_clicks
        self.min_high_price_share = min_high_price_share

    def generate(self, clickstream: pd.DataFrame) -> pd.DataFrame:
        prepared = _prepare_clickstream(clickstream)
        grouped = prepared.groupby("session_id")

        session_clicks = grouped["order"].count()
        high_price_share = grouped["higher_than_average"].mean()

        labels = (
            (session_clicks >= self.min_session_clicks)
            & (high_price_share >= self.min_high_price_share)
        ).astype(int)

        return pd.DataFrame(
            {
                "session_id": labels.index,
                "label": labels.values,
                "label_source": "proxy_hybrid",
                "label_confidence": 1.0,
            }
        )


class ExternalPartialLabelStrategy(IntentLabelStrategy):
    """Load externally produced labels (manual or clustering outputs)."""

    def __init__(
        self,
        labels_path: Path,
        label_source: str,
        session_id_column: str = "session_id",
        label_column: str = "label",
        confidence_column: str | None = None,
    ):
        self.labels_path = Path(labels_path)
        self.label_source = label_source
        self.session_id_column = session_id_column
        self.label_column = label_column
        self.confidence_column = confidence_column

    def generate(self, clickstream: pd.DataFrame) -> pd.DataFrame:
        prepared = _prepare_clickstream(clickstream)
        sessions = pd.DataFrame(
            {"session_id": prepared["session_id"].drop_duplicates().sort_values()}
        )

        external = pd.read_csv(self.labels_path)
        external = _normalize_columns(external)

        if self.session_id_column not in external.columns:
            raise ValueError(
                f"Missing session id column '{self.session_id_column}' in {self.labels_path}"
            )
        if self.label_column not in external.columns:
            raise ValueError(f"Missing label column '{self.label_column}' in {self.labels_path}")

        external = external.rename(
            columns={
                self.session_id_column: "session_id",
                self.label_column: "label",
            }
        )

        external["session_id"] = pd.to_numeric(external["session_id"], errors="coerce")
        external = external.dropna(subset=["session_id"])
        external["session_id"] = external["session_id"].astype(int)

        selected_columns = ["session_id", "label"]
        if self.confidence_column:
            if self.confidence_column not in external.columns:
                raise ValueError(
                    f"Missing confidence column '{self.confidence_column}' in {self.labels_path}"
                )
            selected_columns.append(self.confidence_column)

        external = external[selected_columns].drop_duplicates(subset=["session_id"], keep="last")
        merged = sessions.merge(external, on="session_id", how="left")
        merged["label_source"] = np.where(merged["label"].isna(), "unlabeled", self.label_source)

        if self.confidence_column:
            merged = merged.rename(columns={self.confidence_column: "label_confidence"})
        else:
            merged["label_confidence"] = np.where(merged["label"].isna(), np.nan, 1.0)

        return merged[["session_id", "label", "label_source", "label_confidence"]]


class OverrideLabelStrategy(IntentLabelStrategy):
    """Override a base strategy with non-null labels from an override strategy."""

    def __init__(self, base_strategy: IntentLabelStrategy, override_strategy: IntentLabelStrategy):
        self.base_strategy = base_strategy
        self.override_strategy = override_strategy

    def generate(self, clickstream: pd.DataFrame) -> pd.DataFrame:
        base_labels = self.base_strategy.generate(clickstream).set_index("session_id")
        override_labels = self.override_strategy.generate(clickstream).set_index("session_id")

        aligned_override = override_labels.reindex(base_labels.index)
        has_override = aligned_override["label"].notna()

        merged = base_labels.copy()
        merged.loc[has_override, "label"] = aligned_override.loc[has_override, "label"]
        merged.loc[has_override, "label_source"] = aligned_override.loc[
            has_override, "label_source"
        ]
        merged.loc[has_override, "label_confidence"] = aligned_override.loc[
            has_override, "label_confidence"
        ]

        return merged.reset_index()


def generate_session_labels(
    clickstream: pd.DataFrame,
    label_strategy: IntentLabelStrategy | None = None,
    session_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Generate session-level labels from full-session clickstream.

    Args:
        clickstream: Click-level dataframe.
        label_strategy: Label generation strategy. Defaults to proxy-hybrid.
        session_ids: Optional subset of session_id values to align output with features.

    Returns:
        Dataframe with session_id, label, label_source, label_confidence.
    """

    strategy = label_strategy or ProxyHybridIntentLabelStrategy()
    labels_df = strategy.generate(clickstream=clickstream)
    labels_df = labels_df.drop_duplicates(subset=["session_id"], keep="last")

    if session_ids is not None:
        sessions = pd.DataFrame({"session_id": pd.Series(session_ids).drop_duplicates()})
        labels_df = sessions.merge(labels_df, on="session_id", how="left", validate="one_to_one")

    return labels_df
