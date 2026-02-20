# Labeling

## Purpose
This module generates **session-level intent labels** from the **full session history**.

- File: `online_retail_prediction/modeling/labeling.py`
- Main API: `generate_session_labels(clickstream, label_strategy=..., session_ids=...)`

This is intentionally separate from feature engineering so labels can evolve independently (proxy, manual, clustering, semi-supervised workflows).

## Core Design
Labeling is strategy-based via `IntentLabelStrategy`.

Available strategies:

1. `ProxyHybridIntentLabelStrategy`
   - Uses full-session aggregates:
     - total clicks in session
     - share of high-price clicks (`higher_than_average`)
   - Default rule:
     - `label = 1` if `session_clicks >= min_session_clicks`
     - and `high_price_share >= min_high_price_share`
   - Defaults in CLI:
     - `min_session_clicks = 8`
     - `min_high_price_share = 0.5`

2. `ExternalPartialLabelStrategy`
   - Loads labels from external CSV (manual labels, cluster labels, or pseudo-labels)
   - Supports partial coverage of sessions
   - Unlabeled sessions are retained with:
     - `label = NaN`
     - `label_source = "unlabeled"`

3. `OverrideLabelStrategy`
   - Combines two strategies
   - Uses base strategy labels unless override strategy provides a non-null label
   - Useful for workflows like `manual_over_proxy` or `cluster_over_proxy`

## Label Output Schema
All strategy outputs follow the same schema:

- `session_id`
- `label`
- `label_source`
- `label_confidence`

This standard schema makes downstream training and auditing consistent.

## Full-Session vs First-N Boundary
- Labeling: uses **full session** (all clicks available per session)
- Feature engineering: uses **first N clicks only**

This supports your intended training setup:
- Learn from early-session behavior
- Supervise against full-session intent outcome/proxy

## CLI Integration
`online_retail_prediction/features.py` supports these modes:

- `proxy_hybrid`
- `manual_only`
- `cluster_only`
- `manual_over_proxy`
- `cluster_over_proxy`

Outputs are saved to:

- `data/processed/labels.csv`

## External Label File Requirements
By default, external label files should contain:

- `session_id`
- `label`

Optional:

- confidence column (configured with `--external-confidence-column`)

You can also remap column names with:

- `--external-session-id-column`
- `--external-label-column`

## Example Commands
Proxy-only labeling:

```bash
uv run python -m online_retail_prediction.features \
  --label-mode proxy_hybrid \
  --min-session-clicks 8 \
  --min-high-price-share 0.5
```

Manual labels only:

```bash
uv run python -m online_retail_prediction.features \
  --label-mode manual_only \
  --manual-labels-path data/processed/manual_labels.csv
```

Manual labels overriding proxy labels:

```bash
uv run python -m online_retail_prediction.features \
  --label-mode manual_over_proxy \
  --manual-labels-path data/processed/manual_labels.csv
```
