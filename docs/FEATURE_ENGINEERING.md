# Feature Engineering

## Purpose
This module builds **session-level model features** using either the first `N` clicks or the full session length.

- File: `online_retail_prediction/modeling/feature_engineering.py`
- Main function: `build_session_features(clickstream, n_clicks=5, aggregation_mode="first_n")`
- Orchestration entrypoint: `online_retail_prediction/features.py`

This design supports both leakage-safe training features (first-N) and full-session behavioral features (clustering/labeling workflows).

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
3. Select clicks per mode:
  - `aggregation_mode="first_n"`: keep first `N` rows per session.
  - `aggregation_mode="full_session"`: keep all rows per session.
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

- `data/processed/features_first_n.csv`
- `data/processed/features_full_session.csv`

## Leakage Boundary
- `first_n` mode uses only first `N` clicks by design. Any signal from clicks after `N` is excluded.
- `full_session` mode uses all available clicks per session.

## Example Usage
From the project root:

```bash
uv run python -m online_retail_prediction.features \
  --input-path data/processed/e-shop\ clothing\ 2008.csv \
  --n-clicks 5
```

This generates:

- `data/processed/features_first_n.csv`
- `data/processed/features_full_session.csv`
- `data/processed/baseline_labels.csv` (labels come from labeling module, not feature engineering)

## Detailed feature definitions (every engineered column)

Below are precise, implementation-aligned descriptions of each column produced by
`build_session_features(clickstream, n_clicks=N)` in
`online_retail_prediction/modeling/feature_engineering.py`.

- `session_id` : session identifier (copied from normalized input `session_id`).

- `n_clicks_observed` : integer count of clicks included in the aggregation for
  this session = min(N, session_length). Computed as count of rows per
  `session_id` after taking the first `N` rows by `order`.

- `n_unique_pages` : number of distinct `page` values among the first N clicks.
  Computed with `nunique(page)` on the truncated group.

- `n_unique_models` : number of distinct `page_2_model` values among first N clicks.

- `n_unique_categories` : `nunique(main_category)` among first N clicks.

- `n_unique_colours` : `nunique(colour)` among first N clicks.

- `price_mean` : arithmetic mean of `price` over the first N clicks for the session.
  If only one click is present, equals that click's `price`.

- `price_min` : minimum `price` among first N clicks.

- `price_max` : maximum `price` among first N clicks.

- `price_std` : sample standard deviation of `price` among first N clicks. When
  only one value exists, the code fills `NaN` with `0.0`.

- `high_price_share_first_n` : fraction (in [0,1]) of the first N clicks where
  `higher_than_average == 1`. Computed as `mean(higher_than_average)` on the
  truncated group. Example: 2 high-price clicks out of 5 -> 0.4.

- `high_price_count_first_n` : integer sum of `higher_than_average` in first N clicks.

- `category_entropy` : Shannon entropy (bits) of the `main_category` distribution
  within the first N clicks. Computed as
  `-sum(p_c * log2(p_c))` where `p_c` is the relative frequency of category `c`
  among first N clicks. Returns `0.0` for empty groups.

- `category_share_{X}` : for each distinct category value X observed within the
  first N clicks for a session, a column is produced named `category_share_X`.
  Each value is the relative frequency of category `X` within that session's
  first-N clicks (i.e., count(X)/n_clicks_observed). These columns are created
  dynamically and filled with `0.0` where category `X` did not occur for a
  given session.

- `top_category_share` : maximum of the `category_share_*` values for that session
  (the largest single-category proportion among first N clicks).

- `mean_model_frequency`, `max_model_frequency`, `min_model_frequency` :
  frequency-derived statistics using a global model-frequency mapping computed
  from the first-N truncated clicks across all sessions in the input batch.
  Steps:
  1. compute `model_freq_global[m] = count(m in first-N rows across all sessions) / total_firstN_rows`
  2. map each click's `page_2_model` to `model_freq_global` and compute per-session
     mean/max/min of these mapped values.

- `first_model_frequency` : for each session, look up the `page_2_model` value at
  the first click (within first N) and map it to the global `model_freq_global`
  (0.0 if model not seen in global map). This captures how common the first
  model is across the dataset.

- `last_model_frequency` : same as `first_model_frequency` but for the last
  click included in first N (i.e., the most recent click within truncated window).

- `last_page` : the `page` value of the last click included in the first-N window
  for the session (taken from grouped.tail(1) on the truncated group).

- `last_location` : `location` for the last click included in the truncated window.

- `last_category_frequency` : frequency of the `main_category` value that appeared
  in the last included click, computed relative to the first-N distribution for
  that session (i.e., `category_share_{last_category}`). Implemented by
  computing global per-session category frequencies among first N and mapping
  the last-click's category to that frequency (0.0 if absent).

- `last_colour_frequency` : same as `last_category_frequency` but for `colour`.

- `category_transition_count` : number of category-to-category transitions within
  the first-N sequence. Implementation detail: computed as
  `int(values.ne(values.shift()).sum() - 1)` on the ordered `main_category`
  sequence per session, clipped to a minimum of `0`. Intuitively it counts how
  many times the user switched categories during the observed window (with a
  floor of 0).


### Notes on edge cases and data types

- Sessions with fewer than `N` clicks are retained; all aggregates are computed
  over the available clicks (no padding). `n_clicks_observed` reflects this.
- Numeric coercion: `session_id`, `order`, `price`, and `higher_than_average`
  are coerced to numeric and rows with missing critical values are dropped prior
  to aggregation.
- Column naming: raw input columns are normalized (snake_case) and common raw
  names are mapped (see the normalization section in the file). The feature
  generator expects normalized inputs but also supports common raw names.
