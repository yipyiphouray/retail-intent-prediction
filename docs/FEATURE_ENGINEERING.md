# Feature Engineering

## Purpose
This module builds **session-level model features** using only the first `N` clicks of each session.

- File: `online_retail_prediction/modeling/feature_engineering.py`
- Main function: `build_session_features(clickstream, n_clicks=5)`
- Orchestration entrypoint: `online_retail_prediction/features.py`

This design prevents leakage from later clicks by restricting feature construction to early-session behavior.

## Input Data Expectations
The click-level input must include (after normalization):

- `session_id`
- `order`
- `price`
- `higher_than_average`
- `page_2_model`
- `main_category`
- `colour`
- `page`

### Column normalization supported
Raw dataset columns are normalized to snake_case and mapped when needed:

- `session ID` -> `session_id`
- `page 2 (clothing model)` -> `page_2_model`
- `price 2` -> `higher_than_average` using mapping `{1: 1, 2: 0}`
- `page 1 (main category)` -> `main_category` (ID-to-name mapping)

## Feature Construction Logic
1. Validate required columns and coerce key fields to numeric where needed.
2. Sort by `session_id` and `order`.
3. Keep first `N` rows per session.
4. Aggregate one row per session.

## Feature Families
Current feature families include:

- Basic behavior
  - `n_clicks_observed`
  - `n_unique_pages`, `n_unique_models`, `n_unique_categories`, `n_unique_colours`
- Price behavior (first `N` clicks)
  - `price_mean`, `price_min`, `price_max`, `price_std`
  - `high_price_share_first_n`, `high_price_count_first_n`
- Category mix/diversity
  - `category_entropy`
  - `category_share_*`
  - `top_category_share`
- Model frequency signals
  - `mean_model_frequency`, `max_model_frequency`, `min_model_frequency`
  - `first_model_frequency`, `last_model_frequency`
- Recency/sequence signals
  - `last_page`, `last_location`
  - `last_category_frequency`, `last_colour_frequency`
  - `category_transition_count`

## Output Schema
`build_session_features(...)` returns one row per `session_id`.

- Key column: `session_id`
- Remaining columns: engineered features listed above

When run through `online_retail_prediction/features.py`, features are saved to:

- `data/processed/features.csv`

## Leakage Boundary
Feature engineering uses only first `N` clicks by design. Any signal from clicks after `N` is excluded from feature values.

## Example Usage
From the project root:

```bash
uv run python -m online_retail_prediction.features \
  --input-path data/processed/e-shop\ clothing\ 2008.csv \
  --n-clicks 5
```

This generates:

- `data/processed/features.csv`
- `data/processed/labels.csv` (labels come from labeling module, not feature engineering)
