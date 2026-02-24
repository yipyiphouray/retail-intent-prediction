# Cluster Labeling Module

## Purpose
This module clusters precomputed session features, summarizes clusters, supports manual cluster labeling, and propagates labels to sessions for downstream sequence modeling.

- Module: `online_retail_prediction/modeling/clustering.py`
- Primary class: `SessionClusterer`
- Required functional API:
  - `fit_clustering(session_features_df, k=DEFAULT_CLUSTER_COUNT)`
  - `get_cluster_summary()`
  - `get_representatives(top_n=5)`
  - `apply_manual_labels(cluster_labels_df)`
  - `apply_auto_labels(high_threshold, low_threshold)`
  - `export_outputs(path)`

This module is independent of model training and does not modify `train.py`.

## Input Contract
Input must be a precomputed session-level dataframe (`session_features_df`) with:

- exactly one row per `session_id`
- required column: `session_id`
- numeric feature columns for clustering
- optional non-numeric columns (ignored for clustering)

### Important constraints applied
- no feature engineering occurs in this module
- no additional first-N filtering occurs here
- clustering is performed on the provided session-level features
- `purchase_flag` and `revenue` are excluded from clustering features
- only whitelisted columns in `online_retail_prediction/modeling/cluster_config.py` are used
- new numeric columns added later are ignored unless added to the whitelist

## Cluster configuration

Core defaults are centralized in:

- `online_retail_prediction/modeling/cluster_config.py`

This includes:

- `DEFAULT_CLUSTER_COUNT`
- `DEFAULT_CORRELATION_THRESHOLD`
- `DEFAULT_DROP_CORRELATED`
- `DEFAULT_DROP_LOW_VARIANCE`
- `DEFAULT_MIN_VARIANCE`
- `ALLOWED_CLUSTER_FEATURES`

## Clustering Pipeline

### 1) Feature preprocessing
- Drops `session_id` and non-numeric columns.
- Excludes `purchase_flag` and `revenue` from clustering features.
- Standardizes numeric features with `StandardScaler`.
- Optional PCA is supported (`use_pca`, `pca_n_components`) for dimensionality reduction.

Optional noise-reduction switches:
- `drop_correlated=True` (threshold is config-driven)
- `drop_low_variance=True` with configurable `min_variance`

### 2) KMeans fitting
- Uses `KMeans` with fixed `random_state` and `n_init >= 10`.
- Primary output: `cluster_assignments` with columns:
  - `session_id`
  - `cluster_id`

### 3) Metrics
Stored in `models/clustering_metrics.json`:
- `inertia`
- `silhouette_score`
- cluster size distribution
- warning flags and diagnostic metadata

### 4) Cluster summary
`get_cluster_summary()` returns cluster-level table with:
- `cluster_id`
- `cluster_size`
- `cluster_percentage`
- mean of numeric input columns, prefixed with `mean_`

### 5) Representative sessions
`get_representatives(top_n=5)` computes distance to centroid and returns:
- `cluster_id`
- `session_id`
- `distance_to_centroid`
- `rank_within_cluster`

This supports:
- top-1 representative (required): call with `top_n=1`
- top-5 representatives (optional): call with `top_n=5`

## Labeling

### Manual labeling
`apply_manual_labels(cluster_labels_df)` expects:
- `cluster_id`
- `intent_label`

It propagates labels to all sessions in each cluster and returns:
- `session_id`
- `cluster_id`
- `intent_label`

## Export Paths and Files
`export_outputs(path)` writes session-level outputs to the provided output directory (recommended: `data/cluster_outputs/`) and metrics/models to `models/`.

### Data outputs (`data/cluster_outputs/`)
1. `cluster_assignments.csv`
2. `cluster_summary.csv`
3. `cluster_representatives.csv`
4. `labeled_sessions.csv`
5. `labeled_sessions_for_rnn.csv` (only `session_id`, `intent_label`)

### Model/metrics outputs (`models/`)
1. `clustering_metrics.json`
2. `clustering_scaler.joblib`
3. `clustering_pca.joblib` (if PCA enabled)
4. `kmeans_model.joblib`

## Validation Warnings
Warnings are logged when:
- any cluster has `<1%` of sessions
- silhouette score `<0.1`
- largest cluster share `>30%` of sessions

## Minimal usage example

```python
import pandas as pd

from online_retail_prediction.modeling.clustering import SessionClusterer

session_features_df = pd.read_csv("data/processed/features.csv")

clusterer = SessionClusterer(random_state=42)
clusterer.fit_clustering(session_features_df=session_features_df)

summary_df = clusterer.get_cluster_summary()
representatives_df = clusterer.get_representatives(top_n=5)

# Example manual labels from UI/analyst
cluster_labels_df = pd.DataFrame(
    {
        "cluster_id": summary_df["cluster_id"],
        "intent_label": "unlabeled",
    }
)

labeled_sessions_df = clusterer.apply_manual_labels(cluster_labels_df)
clusterer.export_outputs("data/cluster_outputs")
```

## CLI usage

You can run the full clustering + export pipeline from the command line:

```bash
uv run python -m online_retail_prediction.modeling.cluster_labeling_cli \
  --features-path data/processed/features.csv \
  --output-dir data/cluster_outputs \
  --representatives-top-n 5
```

To remove highly correlated features before clustering:

```bash
uv run python -m online_retail_prediction.modeling.cluster_labeling_cli \
  --features-path data/processed/features.csv \
  --drop-correlated \
  --output-dir data/cluster_outputs
```

Optional low-variance filtering is also available:

```bash
uv run python -m online_retail_prediction.modeling.cluster_labeling_cli \
  --features-path data/processed/features.csv \
  --drop-low-variance \
  --min-variance 1e-8 \
  --output-dir data/cluster_outputs
```

If no manual or auto labels are provided, the CLI exports `intent_label="unlabeled"`
for all sessions.

### Manual labels via file

Provide a labels file with columns:

- `cluster_id`
- `intent_label` (or another column name passed via `--label-column`)

Run:

```bash
uv run python -m online_retail_prediction.modeling.cluster_labeling_cli \
  --features-path data/processed/features.csv \
  --manual-labels-path data/cluster_outputs/cluster_labels.csv \
  --label-column intent_label \
  --output-dir data/cluster_outputs
```

## RNN handoff
Primary handoff file for sequence model integration:
- `data/cluster_outputs/labeled_sessions_for_rnn.csv`

Schema:
- `session_id`
- `intent_label`

The RNN teammate can join this on `session_id` to their sequence dataset.

## Correlation exploration notebook

Interactive correlation analysis is available in:

- `notebooks/clustering_correlation_exploration.ipynb`

Use it to inspect correlation heatmaps, review highly correlated feature pairs, and
decide which columns to drop before running clustering.

## Cluster interpretation notebook

Business-facing cluster interpretation is available in:

- `notebooks/cluster_interpretation.ipynb`

Use it to turn `cluster_summary.csv` into interpretable customer-segment descriptions using
CV-based feature selection and cluster profile visualizations.
