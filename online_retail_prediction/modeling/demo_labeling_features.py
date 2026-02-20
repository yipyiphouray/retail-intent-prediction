"""Demo script to demonstrate labeling and feature engineering modules."""

import pandas as pd
from pathlib import Path

from online_retail_prediction.modeling.feature_engineering import build_session_features
from online_retail_prediction.modeling.labeling import (
    generate_session_labels,
    ProxyHybridIntentLabelStrategy,
)

# Load the preprocessed data
data_path = Path("data/raw/e-shop clothing 2008.csv")
print(f"Loading data from {data_path}...")
clickstream = pd.read_csv(data_path, sep=";")

print(f"\nOriginal data shape: {clickstream.shape}")
print(f"First few rows:\n{clickstream.head()}\n")

# 1. FEATURE ENGINEERING
print("=" * 60)
print("1. FEATURE ENGINEERING")
print("=" * 60)
print("Building session features from first 5 clicks per session...")
features = build_session_features(clickstream, n_clicks=5)

print(f"\nFeature matrix shape: {features.shape}")
print(f"Number of features: {len(features.columns)}")
print(f"\nFeature columns:\n{list(features.columns)}\n")
print(f"Sample features for first 3 sessions:\n{features.head(3)}\n")

# 2. LABELING
print("=" * 60)
print("2. LABELING (Intent Prediction)")
print("=" * 60)
print("Generating labels using ProxyHybridIntentLabelStrategy...")
print("(Sessions with 8+ clicks AND 50%+ high-price items = intent to purchase)")

strategy = ProxyHybridIntentLabelStrategy(
    min_session_clicks=8,
    min_high_price_share=0.5
)
labels = generate_session_labels(
    clickstream,
    label_strategy=strategy,
    session_ids=features["session_id"]
)

print(f"\nLabels shape: {labels.shape}")
print(f"\nLabel distribution:")
print(labels["label"].value_counts())
print(f"\nLabel sources:")
print(labels["label_source"].value_counts())
print(f"\nSample labels:\n{labels.head(10)}\n")

# 3. COMBINED DATASET
print("=" * 60)
print("3. COMBINED FEATURES + LABELS")
print("=" * 60)
dataset = features.merge(labels, on="session_id", how="left")
print(f"Combined dataset shape: {dataset.shape}")
print(f"\nSample combined data:\n{dataset.head()}\n")

print("✓ Demo completed successfully!")
