# Demo Inference Simulation (API + UI)

This guide explains how to run and verify the local demo deployment that connects:

1. React storefront (user browsing simulation)
2. FastAPI session API (click capture + trigger logic)
3. SQLite persistence (session/click storage)
4. Inference pipeline (`InferencePipeline`) with stacking model

## What the Demo Does

1. A new session is created.
2. Each product interaction sends one click payload to the API.
3. The API persists each click to SQLite.
4. On click number 5 (exactly once per session), inference runs.
5. The prediction (`low-intent` or `high-intent`) is stored and returned.
6. The UI can show a promo/ad when prediction is `high-intent`.

## Model Artifact Distribution

The model file is intentionally not stored in git (`models/*.pkl` is ignored).

- Expected runtime file: `models/stacking_ensemble_model.pkl`
- Manifest used for download: `models/model_manifest.json`

`make api` starts FastAPI, and API startup calls `ensure_model(...)`:

- if model exists locally, startup proceeds
- if model is missing, it downloads from `url` in manifest and verifies `sha256`
- if URL/checksum is invalid, startup fails with an actionable error

Manual prefetch option:

```bash
uv run python -m online_retail_prediction.modeling.fetch_model
```

## First-Time Setup

From a fresh clone (on a branch that includes this demo code):

```bash
make requirements
make api
```

In a second terminal:

```bash
make demo_ui
```

## API Endpoints

- `GET /health`
- `POST /api/v1/sessions`
- `GET /api/v1/sessions/{session_id}`
- `POST /api/v1/sessions/{session_id}/clicks`
- `GET /api/v1/sessions/{session_id}/clicks`

## Verify Click Format and Pipeline Trigger

### 1) Health check

```bash
curl -s http://127.0.0.1:8000/health
```

Confirm:

- `"status": "ok"`
- `"model_ready": true`

### 2) Create a session

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/sessions
```

Capture `session_id` from the response.

### 3) Send clicks and confirm trigger on click 5

Replace `<SESSION_ID>` and run 5 requests. Example payload:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/sessions/<SESSION_ID>/clicks" \
  -H 'Content-Type: application/json' \
  -d '{
    "year": 2008,
    "month": 4,
    "day": 1,
    "country": 29,
    "page_1_main_category": 1,
    "page_2_clothing_model": "A13",
    "colour": 5,
    "location": 2,
    "model_photography": 1,
    "price": 39.0,
    "price_2": 1,
    "page": 2
  }'
```

Expected behavior:

- clicks 1-4: `"triggered": false`, `"prediction": null`
- click 5: `"triggered": true`, prediction returned and stored
- click 6+: `"triggered": false` (no automatic re-predict)

### 4) Confirm raw dataset-compatible row format

```bash
curl -s "http://127.0.0.1:8000/api/v1/sessions/<SESSION_ID>/clicks"
```

Each row uses the raw schema keys:

- `year`, `month`, `day`, `order`, `country`, `session ID`
- `page 1 (main category)`, `page 2 (clothing model)`
- `colour`, `location`, `model photography`, `price`, `price 2`, `page`

### 5) Confirm data persisted in SQLite

```bash
sqlite3 data/interim/demo_sessions.db ".schema sessions"
sqlite3 data/interim/demo_sessions.db ".schema clicks"

sqlite3 data/interim/demo_sessions.db \
  "SELECT session_id, predicted, prediction_label, prediction_probability FROM sessions ORDER BY session_id DESC LIMIT 5;"

sqlite3 data/interim/demo_sessions.db \
  "SELECT session_id, order_idx, year, month, day, country, page_1_main_category, page_2_clothing_model, colour, location, model_photography, price, price_2, page FROM clicks WHERE session_id = <SESSION_ID> ORDER BY order_idx;"
```

## Demo Walkthrough (Presentation Flow)

1. Start API (`make api`) and UI (`make demo_ui`).
2. Open storefront and start a session.
3. Click products five times across categories/pages.
4. Show prediction appears on click 5.
5. If prediction is high intent, show promo/ad behavior.
6. Continue clicking to show no second auto-trigger.
7. Start a new session and repeat.

## Troubleshooting

### Model download fails

- Verify `models/model_manifest.json` has a valid public GitHub Release asset URL.
- Verify checksum matches exactly:

```bash
shasum -a 256 models/stacking_ensemble_model.pkl
```

### Checksum mismatch error

- Recompute SHA256 from the exact uploaded artifact.
- Update `sha256` in `models/model_manifest.json`.

### Users cannot download model

- Ensure repository/release visibility allows asset download for intended users.
- If private, users need authenticated access to the release asset.
