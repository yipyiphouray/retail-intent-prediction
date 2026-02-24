# Streamlit Cluster Labeling UI

## Purpose
This UI lets a subject matter expert manually label each cluster as either:

- `low-intent`
- `high-intent`

The app runs clustering on startup, displays cluster summary features, and saves labels to:

- `data/cluster_outputs/cluster_label.csv`

It also surfaces:

- selected-feature-focused cluster summary (from `selected_features.json` when available)
- business-facing cluster descriptions (from `cluster_interpretations.csv` when available)
- visual diagnostics (positioning map + engagement comparison)

## Prerequisites

1. From repo root, sync environment:

```bash
make sync
```

2. Ensure full-session features exist at:

- `data/processed/features_full_session.csv`

If it does not exist, generate it with:

```bash
uv run python -m online_retail_prediction.features \
  --input-path "<path-to-clickstream-csv>" \
  --delimiter ';'
```

For the e-shop dataset, the file is semicolon-delimited (`;`).

## Run The App

Use either command:

```bash
make cluster_ui
```

or

```bash
uv run streamlit run apps/cluster_labeling_app.py
```

Open:

- [http://localhost:8501](http://localhost:8501)

## How To Use

1. App starts and automatically runs clustering once.
2. Left panel shows cluster summary with selected features when available.
3. Right panel shows one label selector per cluster (`Cluster 0`, `Cluster 1`, ...).
4. Review the Cluster Descriptions cards and the two visual diagnostics.
5. Select `low-intent` or `high-intent` for every cluster.
6. Click **Save**.

If any cluster is unlabeled, Save stays disabled and the app shows which clusters are missing.

## Output

After clicking Save, the app writes:

- `data/cluster_outputs/cluster_label.csv`

Output schema:

- `cluster_id` (int)
- `intent_label` (`low-intent` or `high-intent`)

Rows are saved sorted by `cluster_id`.

## Optional Supporting Artifacts Read By The UI

The app reads these files when present and degrades gracefully when they are missing:

- `data/cluster_outputs/selected_features.json`
- `data/cluster_outputs/cluster_interpretations.csv`
- `data/cluster_outputs/cluster_positioning_map.png` (fallback image)
- `data/cluster_outputs/engagement_comparison.png` (fallback image)

## Refresh Clustering

Use **Refresh clustering** to rerun clustering and rebuild the displayed cluster summary.

## Troubleshooting

- `Session features file not found`
  - Create `data/processed/features_full_session.csv` first (see Prerequisites).

- Port already in use (`8501`)
  - Stop existing Streamlit process or run on another port:
  - `uv run streamlit run apps/cluster_labeling_app.py --server.port 8502`

- Need to stop the app from terminal
  - Press `Ctrl + C` in the terminal where Streamlit is running.
