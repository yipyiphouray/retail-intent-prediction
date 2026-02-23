from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from online_retail_prediction.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from online_retail_prediction.modeling.feature_engineering import build_session_features
from online_retail_prediction.modeling.labeling import (
    ProxyHybridIntentLabelStrategy,
    generate_session_labels,
)

app = typer.Typer()

SUPPORTED_MODEL_TYPES = [
    "logistic_regression",
    "random_forest",
    "knn",
    "xgboost",
    "lightgbm",
    "all",
]


def load_and_prepare_data(
    raw_data_path: Path,
    n_clicks: int = 5,
    min_session_clicks: int = 8,
    min_high_price_share: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data and generate features and labels."""
    logger.info(f"Loading raw data from {raw_data_path}...")
    clickstream = pd.read_csv(raw_data_path, sep=";")
    logger.info(f"Loaded {len(clickstream)} clicks")

    logger.info(f"Building session features using first {n_clicks} clicks...")
    features = build_session_features(clickstream, n_clicks=n_clicks)
    logger.info(f"Generated features for {len(features)} sessions")

    logger.info("Generating labels using ProxyHybridIntentLabelStrategy...")
    strategy = ProxyHybridIntentLabelStrategy(
        min_session_clicks=min_session_clicks, min_high_price_share=min_high_price_share
    )
    labels = generate_session_labels(
        clickstream, label_strategy=strategy, session_ids=features["session_id"]
    )
    logger.info(
        f"Label distribution: {labels['label'].value_counts().to_dict()}"
    )

    return features, labels


def prepare_train_test_split(
    features: pd.DataFrame, labels: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Merge features and labels, split into train/test sets."""
    logger.info("Merging features and labels...")
    dataset = features.merge(labels[["session_id", "label"]], on="session_id", how="inner")

    dataset = dataset.dropna(subset=["label"])
    logger.info(f"Dataset shape after removing unlabeled sessions: {dataset.shape}")

    feature_cols = [col for col in dataset.columns if col not in ["session_id", "label"]]
    X = dataset[feature_cols]
    y = dataset["label"].astype(int)

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    logger.info(f"Using {len(feature_cols)} features: {list(X.columns)}")
    logger.info(f"Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logistic_regression",
    random_state: int = 42,
):
    """Train a baseline classifier."""
    logger.info(f"Training {model_type} model...")

    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight="balanced"
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif model_type == "knn":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=15, weights="distance")),
            ]
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)
    logger.success(f"{model_type} model trained successfully")
    return model


def train_tuned_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logistic_regression",
    scoring: str = "roc_auc",
    cv: int = 5,
    random_state: int = 42,
):
    """Train a model with hyperparameter tuning using GridSearchCV.

    Args:
        X_train: Training features.
        y_train: Training labels.
        model_type: Type of model ('logistic_regression' or 'random_forest').
        scoring: Metric to optimize ('roc_auc', 'f1', 'accuracy', 'precision', 'recall').
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (best_model, best_params, cv_results).
    """
    logger.info(f"Tuning {model_type} model with GridSearchCV (scoring={scoring}, cv={cv})...")

    if model_type == "logistic_regression":
        base_model = LogisticRegression(
            random_state=random_state, class_weight="balanced", max_iter=2000
        )
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["lbfgs", "saga"],
            "penalty": ["l2"],
        }
    elif model_type == "random_forest":
        base_model = RandomForestClassifier(
            random_state=random_state, class_weight="balanced", n_jobs=-1
        )
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    logger.success(f"Best {scoring}: {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, split_name: str = "test") -> dict:
    """Evaluate model performance on a given split."""
    logger.info(f"Evaluating model on {split_name} set...")

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_pred_proba),
    }

    logger.info(f"{split_name.upper()} SET METRICS")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name.upper()}: {metric_value:.4f}")

    logger.info(f"\n{split_name.title()} Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    logger.info(f"\n{cm}")

    logger.info(f"\n{split_name.title()} Classification Report:")
    logger.info(f"\n{classification_report(y, y_pred)}")

    return metrics


def run_model_comparison(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    model_types = [
        "logistic_regression",
        "random_forest",
        "knn",
        "xgboost",
        "lightgbm",
    ]
    rows: list[dict[str, float | str]] = []

    for model_type in model_types:
        model = train_baseline_model(X_train, y_train, model_type=model_type, random_state=random_state)
        train_metrics = evaluate_model(model, X_train, y_train, split_name=f"{model_type}_train")
        test_metrics = evaluate_model(model, X_test, y_test, split_name=f"{model_type}_test")

        rows.append(
            {
                "model": model_type,
                "train_accuracy": train_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "test_precision": test_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "test_recall": test_metrics["recall"],
                "train_f1": train_metrics["f1_score"],
                "test_f1": test_metrics["f1_score"],
                "train_roc_auc": train_metrics["roc_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
        )

        model_output_path = MODELS_DIR / f"{model_type}_model.pkl"
        joblib.dump(model, model_output_path)
        logger.info(f"Saved model artifact to {model_output_path}")

    results = pd.DataFrame(rows)
    results = results.sort_values("test_roc_auc", ascending=False).reset_index(drop=True)

    logger.info("MODEL COMPARISON (sorted by test_roc_auc)")
    logger.info("\n" + results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    comparison_csv = REPORTS_DIR / "model_comparison_baseline.csv"
    results.to_csv(comparison_csv, index=False)
    logger.success(f"Saved comparison table to {comparison_csv}")

    plt.figure(figsize=(10, 6))
    plt.plot(results["model"], results["test_roc_auc"], marker="o", label="Test ROC-AUC")
    plt.plot(results["model"], results["test_f1"], marker="s", label="Test F1")
    plt.plot(results["model"], results["test_accuracy"], marker="^", label="Test Accuracy")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.title("Baseline Model Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    comparison_png = FIGURES_DIR / "model_comparison_baseline.png"
    plt.savefig(comparison_png, dpi=180)
    plt.close()
    logger.success(f"Saved comparison plot to {comparison_png}")

    return results


@app.command()
def main(
    raw_data_path: Path = RAW_DATA_DIR / "e-shop clothing 2008.csv",
    model_path: Path = MODELS_DIR / "baseline_model.pkl",
    model_type: str = "logistic_regression",
    n_clicks: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    save_features: bool = True,
    tune: bool = False,
    scoring: str = "roc_auc",
    cv: int = 5,
):
    """
    Train a model to predict purchase intent.

    Args:
        raw_data_path: Path to raw clickstream data
        model_path: Path to save trained model
        model_type: Type of model ('logistic_regression', 'random_forest', 'knn',
            'xgboost', 'lightgbm', or 'all' for full comparison)
        n_clicks: Number of initial clicks to use for features
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        save_features: Whether to save features and labels to processed data directory
        tune: Enable hyperparameter tuning with GridSearchCV
        scoring: Metric to optimize when tuning ('roc_auc', 'f1', 'accuracy', 'precision', 'recall')
        cv: Number of cross-validation folds for tuning
    """
    logger.info("MODEL TRAINING PIPELINE" + (" (with tuning)" if tune else " (baseline)"))

    model_type = model_type.strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type: {model_type}. Use one of {SUPPORTED_MODEL_TYPES}.")

    features, labels = load_and_prepare_data(raw_data_path, n_clicks=n_clicks)

    if save_features:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        features_path = PROCESSED_DATA_DIR / "features.csv"
        labels_path = PROCESSED_DATA_DIR / "labels.csv"
        features.to_csv(features_path, index=False)
        labels.to_csv(labels_path, index=False)
        logger.info(f"Saved features to {features_path}")
        logger.info(f"Saved labels to {labels_path}")

    X_train, X_test, y_train, y_test = prepare_train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    if model_type == "all":
        run_model_comparison(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            random_state=random_state,
        )
        logger.success("All-model comparison complete!")
        return

    best_params = None
    if tune:
        if model_type not in {"logistic_regression", "random_forest"}:
            raise ValueError("Tuning is currently supported only for logistic_regression and random_forest")
        model, best_params, cv_results = train_tuned_model(
            X_train, y_train,
            model_type=model_type,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
        )
    else:
        model = train_baseline_model(X_train, y_train, model_type=model_type, random_state=random_state)

    train_metrics = evaluate_model(model, X_train, y_train, split_name="train")
    test_metrics = evaluate_model(model, X_test, y_test, split_name="test")

    logger.info("TRAIN vs TEST COMPARISON")
    for metric_name in train_metrics:
        train_val = train_metrics[metric_name]
        test_val = test_metrics[metric_name]
        diff = train_val - test_val
        logger.info(f"{metric_name.upper()}: Train={train_val:.4f}, Test={test_val:.4f}, Diff={diff:+.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.success(f"Model saved to {model_path}")

    metrics_path = model_path.parent / f"{model_path.stem}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("MODEL PERFORMANCE METRICS\n\n")
        if best_params:
            f.write(f"TUNING: scoring={scoring}, cv={cv}\n")
            f.write("BEST PARAMETERS\n")
            for param_name, param_value in best_params.items():
                f.write(f"{param_name}: {param_value}\n")
            f.write("\n")
        f.write("TRAIN SET\n")
        for metric_name, metric_value in train_metrics.items():
            f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")
        f.write("\nTEST SET\n")
        for metric_name, metric_value in test_metrics.items():
            f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")
    logger.info(f"Metrics saved to {metrics_path}")

    logger.success("Training pipeline complete!")


if __name__ == "__main__":
    app()
