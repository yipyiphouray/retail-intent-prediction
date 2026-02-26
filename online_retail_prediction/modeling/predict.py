"""End-to-end inference pipeline for the stacking ensemble classifier."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from online_retail_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from online_retail_prediction.modeling.feature_engineering import (
    _normalize_columns,
    build_session_features,
)

app = typer.Typer()

LABEL_MAP = {0: "low-intent", 1: "high-intent"}
DEFAULT_FREQ_PATH = PROCESSED_DATA_DIR / "training_frequencies.json"


@functools.lru_cache(maxsize=1)
def load_model(model_path: str):
    """Load and cache the stacking ensemble model."""
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.success("Model loaded")
    return model


@functools.lru_cache(maxsize=1)
def _load_frequencies(freq_path: str) -> dict:
    """Load pre-computed training frequency tables."""
    with open(freq_path) as f:
        return json.load(f)


def _correct_frequency_features(
    feature_row: pd.DataFrame,
    clicks: pd.DataFrame,
    freq_path: Path,
    n_clicks: int = 5,
) -> pd.DataFrame:
    """Replace within-session frequency features with training-set global frequencies.

    During training, model/category/colour frequencies are computed across all
    sessions. At inference with a single session, build_session_features computes
    within-session frequencies instead. This function corrects that mismatch.
    """
    freqs = _load_frequencies(str(freq_path))
    model_freq = freqs["model_freq"]
    category_freq = freqs["category_freq"]
    colour_freq = freqs["colour_freq"]

    prepared = _normalize_columns(clicks)
    prepared["order"] = pd.to_numeric(prepared["order"], errors="coerce")
    prepared = prepared.sort_values("order")
    first_n = prepared.head(n_clicks)

    if first_n.empty:
        return feature_row

    # Map each click's model to its global frequency
    click_model_freqs = first_n["page_2_model"].map(model_freq).fillna(0.0)
    feature_row["mean_model_frequency"] = click_model_freqs.mean()
    feature_row["max_model_frequency"] = click_model_freqs.max()
    feature_row["min_model_frequency"] = click_model_freqs.min()

    # First and last model frequencies
    feature_row["first_model_frequency"] = model_freq.get(
        str(first_n["page_2_model"].iloc[0]), 0.0
    )
    feature_row["last_model_frequency"] = model_freq.get(
        str(first_n["page_2_model"].iloc[-1]), 0.0
    )

    # Last click's category and colour frequencies
    last_click = first_n.iloc[-1]
    feature_row["last_category_frequency"] = category_freq.get(
        str(last_click["main_category"]), 0.0
    )
    feature_row["last_colour_frequency"] = colour_freq.get(
        str(last_click["colour"]), 0.0
    )

    return feature_row


def predict_session(
    clicks: pd.DataFrame,
    model_path: Path = MODELS_DIR / "stacking_ensemble_model.pkl",
    freq_path: Path = DEFAULT_FREQ_PATH,
    n_clicks: int = 5,
) -> dict:
    """Run the full inference pipeline for a single session.

    Args:
        clicks: Raw click-level DataFrame for one session.
        model_path: Path to the trained stacking ensemble .pkl file.
        freq_path: Path to pre-computed training frequency tables.
        n_clicks: Number of initial clicks to use for features.

    Returns:
        Dict with keys: label, probability, features.
    """
    if clicks.empty:
        raise ValueError("clicks DataFrame is empty — at least one click is required.")

    model = load_model(str(model_path))
    expected_features = [str(f) for f in model.feature_names_in_]

    features_df = build_session_features(
        clicks, n_clicks=n_clicks, aggregation_mode="first_n"
    )

    # Drop session_id — not a model feature
    feature_row = features_df.drop(columns=["session_id"], errors="ignore")

    # Align columns: add any missing features as 0.0, reorder to match model
    for col in expected_features:
        if col not in feature_row.columns:
            feature_row[col] = 0.0
    feature_row = feature_row[expected_features]

    # Correct frequency features using training-set global frequencies
    feature_row = _correct_frequency_features(feature_row, clicks, freq_path, n_clicks)

    proba = model.predict_proba(feature_row)[0, 1]
    pred = model.predict(feature_row)[0]

    return {
        "label": LABEL_MAP[int(pred)],
        "probability": round(float(proba), 4),
        "features": feature_row.iloc[0].to_dict(),
    }


@app.command()
def main(
    raw_data_path: Path = RAW_DATA_DIR / "e-shop clothing 2008.csv",
    model_path: Path = MODELS_DIR / "stacking_ensemble_model.pkl",
    session_id: int = 1,
):
    """Run inference on a single session from the raw dataset."""
    logger.info(f"Loading raw data from {raw_data_path}")
    raw = pd.read_csv(raw_data_path, sep=";")

    session_clicks = raw[raw["session ID"] == session_id]
    if session_clicks.empty:
        logger.error(f"Session {session_id} not found in raw data.")
        raise typer.Exit(code=1)

    logger.info(f"Session {session_id}: {len(session_clicks)} clicks")
    result = predict_session(session_clicks, model_path=model_path)

    logger.success(f"Prediction: {result['label']} (probability={result['probability']})")


if __name__ == "__main__":
    app()
