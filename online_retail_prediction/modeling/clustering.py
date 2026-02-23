from __future__ import annotations

import json
from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from online_retail_prediction.config import DATA_DIR, MODELS_DIR
from online_retail_prediction.modeling.cluster_config import (
    ALLOWED_CLUSTER_FEATURES,
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_DROP_CORRELATED,
    DEFAULT_DROP_LOW_VARIANCE,
    DEFAULT_MIN_VARIANCE,
)


class SessionClusterer:
    """Cluster session-level feature rows and propagate cluster labels to sessions."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.kmeans: KMeans | None = None

        self.feature_columns: list[str] = []
        self.cluster_assignments_df: pd.DataFrame | None = None
        self.cluster_summary_df: pd.DataFrame | None = None
        self.cluster_representatives_df: pd.DataFrame | None = None
        self.labeled_sessions_df: pd.DataFrame | None = None
        self.metrics: dict[str, float | int | bool | list[int] | dict[str, int] | None] = {}

    @staticmethod
    def _validate_input(session_features_df: pd.DataFrame) -> None:
        if "session_id" not in session_features_df.columns:
            raise ValueError("Input dataframe must contain 'session_id'.")

        if session_features_df["session_id"].isna().any():
            raise ValueError("'session_id' contains missing values.")

        if session_features_df["session_id"].duplicated().any():
            raise ValueError("Input must contain one row per session_id.")

    @staticmethod
    def _drop_highly_correlated(
        X: pd.DataFrame, threshold: float
    ) -> tuple[pd.DataFrame, list[str]]:
        if X.empty:
            return X, []

        corr = X.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if (upper[col] > threshold).any()]
        if not drop_cols:
            return X, []

        return X.drop(columns=drop_cols), drop_cols

    @staticmethod
    def _drop_low_variance(X: pd.DataFrame, min_variance: float) -> tuple[pd.DataFrame, list[str]]:
        if X.empty:
            return X, []

        variances = X.var(numeric_only=True)
        drop_cols = variances[variances <= min_variance].index.tolist()
        if not drop_cols:
            return X, []

        return X.drop(columns=drop_cols), drop_cols

    def fit_clustering(
        self,
        session_features_df: pd.DataFrame,
        k: int = DEFAULT_CLUSTER_COUNT,
        *,
        use_pca: bool = False,
        pca_n_components: int | float | None = None,
        n_init: int = 10,
        drop_correlated: bool = DEFAULT_DROP_CORRELATED,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
        drop_low_variance: bool = DEFAULT_DROP_LOW_VARIANCE,
        min_variance: float = DEFAULT_MIN_VARIANCE,
    ) -> pd.DataFrame:
        """Fit KMeans on numeric session features and return cluster assignments."""

        if k < 2:
            raise ValueError("k must be at least 2.")
        if n_init < 10:
            raise ValueError("n_init must be >= 10.")

        self._validate_input(session_features_df)

        frame = session_features_df.copy()
        frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce")
        frame = frame.dropna(subset=["session_id"])
        frame["session_id"] = frame["session_id"].astype(int)

        numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        excluded_cols = {"session_id", "purchase_flag", "revenue"}
        candidate_numeric = [col for col in numeric_cols if col not in excluded_cols]

        whitelisted_features = [
            col for col in ALLOWED_CLUSTER_FEATURES if col in candidate_numeric
        ]
        ignored_extra_numeric_features = [
            col for col in candidate_numeric if col not in ALLOWED_CLUSTER_FEATURES
        ]
        missing_whitelisted_features = [
            col for col in ALLOWED_CLUSTER_FEATURES if col not in frame.columns
        ]

        if ignored_extra_numeric_features:
            logger.info(
                "Ignoring {} non-whitelisted numeric features: {}",
                len(ignored_extra_numeric_features),
                ignored_extra_numeric_features,
            )

        feature_columns = whitelisted_features

        if not feature_columns:
            raise ValueError("No numeric feature columns available for clustering.")

        X = frame[feature_columns].copy()

        dropped_correlated: list[str] = []
        dropped_low_variance: list[str] = []
        if drop_correlated:
            X, dropped_correlated = self._drop_highly_correlated(
                X, threshold=correlation_threshold
            )
        if drop_low_variance:
            X, dropped_low_variance = self._drop_low_variance(X, min_variance=min_variance)

        if X.empty or X.shape[1] == 0:
            raise ValueError("No features remain after optional feature pruning.")
        if len(X) < k:
            raise ValueError(f"k={k} exceeds number of sessions ({len(X)}).")

        self.feature_columns = X.columns.tolist()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if use_pca:
            self.pca = PCA(n_components=pca_n_components, random_state=self.random_state)
            X_for_clustering = self.pca.fit_transform(X_scaled)
        else:
            self.pca = None
            X_for_clustering = X_scaled

        self.kmeans = KMeans(random_state=self.random_state, n_clusters=k, n_init=n_init)
        cluster_ids = self.kmeans.fit_predict(X_for_clustering)

        assignments = pd.DataFrame(
            {
                "session_id": frame["session_id"].values,
                "cluster_id": cluster_ids,
            }
        )
        self.cluster_assignments_df = assignments

        silhouette: float | None = None
        calinski_harabasz: float | None = None
        davies_bouldin: float | None = None
        if len(np.unique(cluster_ids)) > 1 and len(X_for_clustering) > len(np.unique(cluster_ids)):
            silhouette = float(silhouette_score(X_for_clustering, cluster_ids))
            calinski_harabasz = float(calinski_harabasz_score(X_for_clustering, cluster_ids))
            davies_bouldin = float(davies_bouldin_score(X_for_clustering, cluster_ids))

        cluster_counts = assignments["cluster_id"].value_counts().sort_index()
        total_sessions = len(assignments)
        cluster_percentage = (cluster_counts / total_sessions) * 100

        small_clusters = cluster_percentage[cluster_percentage < 1.0].index.tolist()
        silhouette_low = silhouette is not None and silhouette < 0.1
        largest_cluster_share = (
            float(cluster_percentage.max()) if not cluster_percentage.empty else 0.0
        )
        imbalanced_distribution = largest_cluster_share > 30.0

        if small_clusters:
            logger.warning(
                "{} clusters are below 1% of total sessions: {}",
                len(small_clusters),
                small_clusters,
            )
        if silhouette_low:
            logger.warning("Silhouette score is low ({:.4f} < 0.1).", silhouette)
        if imbalanced_distribution:
            logger.warning(
                "Largest cluster share is high ({:.2f}% > 30%).",
                largest_cluster_share,
            )

        self.metrics = {
            "k": int(k),
            "random_state": int(self.random_state),
            "n_init": int(n_init),
            "n_sessions": int(total_sessions),
            "n_features_used": int(X.shape[1]),
            "feature_columns": self.feature_columns,
            "ignored_extra_numeric_features": ignored_extra_numeric_features,
            "missing_whitelisted_features": missing_whitelisted_features,
            "dropped_correlated_features": dropped_correlated,
            "dropped_low_variance_features": dropped_low_variance,
            "use_pca": bool(use_pca),
            "pca_n_components": pca_n_components,
            "pca_effective_components": int(self.pca.n_components_)
            if self.pca is not None
            else None,
            "inertia": float(self.kmeans.inertia_),
            "silhouette_score": silhouette,
            "calinski_harabasz_score": calinski_harabasz,
            "davies_bouldin_score": davies_bouldin,
            "small_cluster_ids_lt_1pct": [int(value) for value in small_clusters],
            "silhouette_below_0_1": bool(silhouette_low),
            "largest_cluster_share_pct": largest_cluster_share,
            "imbalanced_distribution_gt_30pct": bool(imbalanced_distribution),
            "cluster_size_distribution": {
                str(int(idx)): int(count) for idx, count in cluster_counts.items()
            },
        }

        summary_base = assignments.merge(frame, on="session_id", how="left")
        numeric_summary_columns = summary_base.select_dtypes(include=[np.number]).columns.tolist()
        numeric_summary_columns = [
            col for col in numeric_summary_columns if col not in {"session_id", "cluster_id"}
        ]

        summary = summary_base.groupby("cluster_id", as_index=False).agg(
            cluster_size=("session_id", "count")
        )
        summary["cluster_percentage"] = (summary["cluster_size"] / total_sessions) * 100

        if numeric_summary_columns:
            means = (
                summary_base.groupby("cluster_id", as_index=False)[numeric_summary_columns]
                .mean()
                .rename(columns={col: f"mean_{col}" for col in numeric_summary_columns})
            )
            summary = summary.merge(means, on="cluster_id", how="left")

        self.cluster_summary_df = summary.sort_values("cluster_id").reset_index(drop=True)

        distance_matrix = self.kmeans.transform(X_for_clustering)
        min_distances = distance_matrix[np.arange(len(assignments)), cluster_ids]

        representatives = assignments.copy()
        representatives["distance_to_centroid"] = min_distances
        representatives = representatives.sort_values(
            ["cluster_id", "distance_to_centroid", "session_id"]
        )
        representatives["rank_within_cluster"] = (
            representatives.groupby("cluster_id").cumcount() + 1
        )
        self.cluster_representatives_df = representatives.reset_index(drop=True)

        logger.info(
            "Finished clustering {} sessions into {} clusters. inertia={:.3f}, silhouette={}",
            total_sessions,
            k,
            self.metrics["inertia"],
            "None" if silhouette is None else f"{silhouette:.4f}",
        )

        return self.cluster_assignments_df.copy()

    def get_cluster_summary(self) -> pd.DataFrame:
        """Return cluster-level summary table."""
        if self.cluster_summary_df is None:
            raise ValueError("fit_clustering must be called before get_cluster_summary.")
        return self.cluster_summary_df.copy()

    def get_representatives(self, top_n: int = 5) -> pd.DataFrame:
        """Return top-N closest sessions to each cluster centroid."""
        if top_n <= 0:
            raise ValueError("top_n must be greater than 0.")
        if self.cluster_representatives_df is None:
            raise ValueError("fit_clustering must be called before get_representatives.")

        result = self.cluster_representatives_df[
            self.cluster_representatives_df["rank_within_cluster"] <= top_n
        ]
        return result.reset_index(drop=True)

    def apply_manual_labels(self, cluster_labels_df: pd.DataFrame) -> pd.DataFrame:
        """Apply manual labels by cluster_id and propagate them to all sessions."""
        if self.cluster_assignments_df is None:
            raise ValueError("fit_clustering must be called before apply_manual_labels.")

        required_columns = {"cluster_id", "intent_label"}
        if not required_columns.issubset(cluster_labels_df.columns):
            missing = required_columns.difference(cluster_labels_df.columns)
            raise ValueError(f"cluster_labels_df missing required columns: {sorted(missing)}")

        cluster_labels = cluster_labels_df[["cluster_id", "intent_label"]].drop_duplicates(
            subset=["cluster_id"], keep="last"
        )

        labeled = self.cluster_assignments_df.merge(cluster_labels, on="cluster_id", how="left")
        self.labeled_sessions_df = labeled[["session_id", "cluster_id", "intent_label"]].copy()
        return self.labeled_sessions_df.copy()

    def apply_auto_labels(self, high_threshold: float, low_threshold: float) -> pd.DataFrame:
        """Automatically label clusters using purchase_rate thresholds when available."""
        if high_threshold < low_threshold:
            raise ValueError("high_threshold must be greater than or equal to low_threshold.")
        if self.cluster_summary_df is None or self.cluster_assignments_df is None:
            raise ValueError("fit_clustering must be called before apply_auto_labels.")
        if "purchase_rate" not in self.cluster_summary_df.columns:
            raise ValueError(
                "purchase_rate is not available in cluster summary. "
                "Automatic labeling requires purchase_flag-derived purchase_rate."
            )

        summary = self.cluster_summary_df[["cluster_id", "purchase_rate"]].copy()
        summary["intent_label"] = "medium_intent"
        summary.loc[summary["purchase_rate"] >= high_threshold, "intent_label"] = "high_intent"
        summary.loc[summary["purchase_rate"] <= low_threshold, "intent_label"] = "low_intent"

        labeled = self.cluster_assignments_df.merge(
            summary[["cluster_id", "intent_label"]], on="cluster_id", how="left"
        )
        self.labeled_sessions_df = labeled[["session_id", "cluster_id", "intent_label"]].copy()
        return self.labeled_sessions_df.copy()

    def export_outputs(self, path: Path | str = DATA_DIR / "cluster_outputs") -> dict[str, Path]:
        """Export clustering tables to data outputs and metrics/models to models directory."""
        if self.cluster_assignments_df is None:
            raise ValueError("fit_clustering must be called before export_outputs.")

        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        assignments_path = output_dir / "cluster_assignments.csv"
        summary_path = output_dir / "cluster_summary.csv"
        representatives_path = output_dir / "cluster_representatives.csv"
        labeled_path = output_dir / "labeled_sessions.csv"
        labeled_for_rnn_path = output_dir / "labeled_sessions_for_rnn.csv"

        metrics_path = MODELS_DIR / "clustering_metrics.json"
        scaler_path = MODELS_DIR / "clustering_scaler.joblib"
        pca_path = MODELS_DIR / "clustering_pca.joblib"
        kmeans_path = MODELS_DIR / "kmeans_model.joblib"

        self.cluster_assignments_df.to_csv(assignments_path, index=False)
        if self.cluster_summary_df is not None:
            self.cluster_summary_df.to_csv(summary_path, index=False)
        if self.cluster_representatives_df is not None:
            self.cluster_representatives_df.to_csv(representatives_path, index=False)
        if self.labeled_sessions_df is not None:
            self.labeled_sessions_df.to_csv(labeled_path, index=False)
            self.labeled_sessions_df[["session_id", "intent_label"]].to_csv(
                labeled_for_rnn_path, index=False
            )

        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(self.metrics, handle, indent=2)

        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
        if self.pca is not None:
            joblib.dump(self.pca, pca_path)
        if self.kmeans is not None:
            joblib.dump(self.kmeans, kmeans_path)

        logger.info("Exported clustering outputs to {} and models to {}", output_dir, MODELS_DIR)

        return {
            "cluster_assignments": assignments_path,
            "cluster_summary": summary_path,
            "cluster_representatives": representatives_path,
            "labeled_sessions": labeled_path,
            "labeled_sessions_for_rnn": labeled_for_rnn_path,
            "clustering_metrics": metrics_path,
            "scaler": scaler_path,
            "pca": pca_path,
            "kmeans_model": kmeans_path,
        }


_DEFAULT_CLUSTERER = SessionClusterer()


def fit_clustering(
    session_features_df: pd.DataFrame,
    k: int = DEFAULT_CLUSTER_COUNT,
) -> pd.DataFrame:
    """Fit default clusterer and return session-to-cluster assignments."""
    return _DEFAULT_CLUSTERER.fit_clustering(session_features_df=session_features_df, k=k)


def get_cluster_summary() -> pd.DataFrame:
    """Return summary from default fitted clusterer."""
    return _DEFAULT_CLUSTERER.get_cluster_summary()


def get_representatives(top_n: int = 5) -> pd.DataFrame:
    """Return representative sessions from default fitted clusterer."""
    return _DEFAULT_CLUSTERER.get_representatives(top_n=top_n)


def apply_manual_labels(cluster_labels_df: pd.DataFrame) -> pd.DataFrame:
    """Apply cluster labels to sessions using default fitted clusterer."""
    return _DEFAULT_CLUSTERER.apply_manual_labels(cluster_labels_df=cluster_labels_df)


def apply_auto_labels(high_threshold: float, low_threshold: float) -> pd.DataFrame:
    """Apply rule-based labels using default fitted clusterer."""
    return _DEFAULT_CLUSTERER.apply_auto_labels(
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )


def export_outputs(path: Path | str) -> dict[str, Path]:
    """Export outputs from default fitted clusterer."""
    return _DEFAULT_CLUSTERER.export_outputs(path=path)
