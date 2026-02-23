from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import typer

from online_retail_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def _create_bagging_classifier(
    base_estimator,
    n_estimators: int = 50,
    max_samples: float = 0.8,
    max_features: float = 1.0,
    bootstrap: bool = True,
    random_state: int = 42,
) -> BaggingClassifier:
    """Create a scikit-learn version compatible BaggingClassifier."""
    params = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "random_state": random_state,
        "n_jobs": 1,
    }

    try:
        return BaggingClassifier(estimator=base_estimator, **params)
    except TypeError:
        return BaggingClassifier(base_estimator=base_estimator, **params)


def load_processed_features_and_labels(
    features_path: Path,
    labels_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and merge processed features and labels."""
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    required_feature_columns = {"session_id"}
    required_label_columns = {"session_id", "label"}

    if not required_feature_columns.issubset(features.columns):
        missing = required_feature_columns - set(features.columns)
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    if not required_label_columns.issubset(labels.columns):
        missing = required_label_columns - set(labels.columns)
        raise ValueError(f"Missing required label columns: {sorted(missing)}")

    dataset = features.merge(labels[["session_id", "label"]], on="session_id", how="inner")
    dataset = dataset.dropna(subset=["label"])

    if dataset.empty:
        raise ValueError("Merged dataset is empty after dropping missing labels.")

    y = dataset["label"].astype(int)
    feature_cols = [col for col in dataset.columns if col not in ["session_id", "label"]]
    X = dataset[feature_cols]
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    if X.empty:
        raise ValueError("No numeric feature columns available for model training.")

    if y.nunique() < 2:
        raise ValueError("Label column must contain at least two classes.")

    logger.info(f"Loaded dataset with {len(X)} rows and {X.shape[1]} numeric features")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into stratified train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def build_model_configs(
    random_state: int = 42,
) -> dict[str, tuple[object, dict[str, list[object]]]]:
    """Build bagging base-estimator configs and tuning grids."""
    logistic_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    class_weight="balanced",
                    max_iter=3000,
                ),
            ),
        ]
    )

    knn_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(algorithm="kd_tree", leaf_size=30, n_jobs=1),
            ),
        ]
    )

    decision_tree = DecisionTreeClassifier(random_state=random_state, class_weight="balanced")

    configs: dict[str, tuple[object, dict[str, list[object]]]] = {
        "logistic_regression": (
            logistic_pipeline,
            {
                "n_estimators": [25, 50, 100],
                "max_samples": [0.6, 0.8, 1.0],
                "max_features": [0.7, 1.0],
                "bootstrap": [True],
                "estimator__classifier__C": [0.1, 1.0, 10.0],
            },
        ),
        "decision_tree": (
            decision_tree,
            {
                "n_estimators": [25, 50, 100],
                "max_samples": [0.6, 0.8, 1.0],
                "max_features": [0.7, 1.0],
                "bootstrap": [True, False],
                "estimator__max_depth": [5, 10, None],
                "estimator__min_samples_split": [2, 5, 10],
                "estimator__min_samples_leaf": [1, 2, 4],
            },
        ),
        "knn": (
            knn_pipeline,
            {
                # Reduced grid to limit total CV fits for KNN (faster tuning)
                "n_estimators": [25, 50],
                "max_samples": [0.6, 1.0],
                "max_features": [0.7, 1.0],
                "bootstrap": [True],
                "estimator__classifier__n_neighbors": [3, 5],
                "estimator__classifier__weights": ["uniform"],
                "estimator__classifier__p": [2],
            },
        ),
    }

    return configs


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Evaluate model on required metrics."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    return {
        "roc_auc": roc_auc_score(y, y_pred_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "cohens_kappa": cohen_kappa_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
    }


def tune_and_train_bagging_model(
    model_name: str,
    base_estimator,
    param_grid: dict[str, list[object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[BaggingClassifier, dict[str, object], float]:
    """Tune and train one bagging model family with ROC-AUC optimization."""
    base_model = _create_bagging_classifier(
        clone(base_estimator),
        random_state=random_state,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )

    logger.info(f"Starting tuning for bagging_{model_name}")
    grid_search.fit(X_train, y_train)
    logger.success(f"Best ROC-AUC for bagging_{model_name}: {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters for bagging_{model_name}: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def save_results(
    model_name: str,
    model,
    best_params: dict[str, object],
    best_cv_roc_auc: float,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Save trained model and evaluation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_stem = f"bagging_{model_name}"
    model_path = output_dir / f"{model_stem}.pkl"
    metrics_path = output_dir / f"{model_stem}_metrics.txt"

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(f"MODEL: {model_stem}\n")
        file.write("OPTIMIZATION_METRIC: roc_auc\n")
        file.write(f"BEST_CV_ROC_AUC: {best_cv_roc_auc:.6f}\n\n")

        file.write("BEST_PARAMS\n")
        for param_name, param_value in best_params.items():
            file.write(f"{param_name}: {param_value}\n")

        file.write("\nTRAIN_METRICS\n")
        file.write(f"ROC_AUC: {train_metrics['roc_auc']:.6f}\n")
        file.write(f"PRECISION: {train_metrics['precision']:.6f}\n")
        file.write(f"RECALL: {train_metrics['recall']:.6f}\n")
        file.write(f"F1: {train_metrics['f1']:.6f}\n")
        file.write(f"MACRO_F1: {train_metrics['macro_f1']:.6f}\n")
        file.write(f"COHENS_KAPPA: {train_metrics['cohens_kappa']:.6f}\n")
        file.write(f"ACCURACY: {train_metrics['accuracy']:.6f}\n")

        file.write("\nTEST_METRICS\n")
        file.write(f"ROC_AUC: {test_metrics['roc_auc']:.6f}\n")
        file.write(f"PRECISION: {test_metrics['precision']:.6f}\n")
        file.write(f"RECALL: {test_metrics['recall']:.6f}\n")
        file.write(f"F1: {test_metrics['f1']:.6f}\n")
        file.write(f"MACRO_F1: {test_metrics['macro_f1']:.6f}\n")
        file.write(f"COHENS_KAPPA: {test_metrics['cohens_kappa']:.6f}\n")
        file.write(f"ACCURACY: {test_metrics['accuracy']:.6f}\n")

    logger.success(f"Saved model: {model_path}")
    logger.success(f"Saved metrics: {metrics_path}")


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    output_dir: Path = MODELS_DIR,
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 42,
):
    """Train, tune, and evaluate bagging models with LR, DT, and KNN base estimators."""
    logger.info("BAGGING TRAINING PIPELINE")
    logger.info("Fine-tuning objective: roc_auc")

    X, y = load_processed_features_and_labels(features_path=features_path, labels_path=labels_path)
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    configs = build_model_configs(random_state=random_state)
    summary_rows: list[dict[str, object]] = []

    for model_name, (base_estimator, param_grid) in configs.items():
        model, best_params, best_cv_roc_auc = tune_and_train_bagging_model(
            model_name=model_name,
            base_estimator=base_estimator,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            random_state=random_state,
        )

        train_metrics = evaluate_model(model, X_train, y_train)
        test_metrics = evaluate_model(model, X_test, y_test)

        logger.info(
            f"bagging_{model_name} TEST | "
            f"ROC_AUC={test_metrics['roc_auc']:.4f}, "
            f"PRECISION={test_metrics['precision']:.4f}, "
            f"RECALL={test_metrics['recall']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, "
            f"MACRO_F1={test_metrics['macro_f1']:.4f}, "
            f"COHENS_KAPPA={test_metrics['cohens_kappa']:.4f}, "
            f"ACCURACY={test_metrics['accuracy']:.4f}"
        )

        save_results(
            model_name=model_name,
            model=model,
            best_params=best_params,
            best_cv_roc_auc=best_cv_roc_auc,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            output_dir=output_dir,
        )

        summary_rows.append(
            {
                "model": f"bagging_{model_name}",
                "best_cv_roc_auc": best_cv_roc_auc,
                "test_roc_auc": test_metrics["roc_auc"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_cohens_kappa": test_metrics["cohens_kappa"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(by="test_roc_auc", ascending=False)
    summary_path = output_dir / "bagging_model_comparison.csv"
    summary.to_csv(summary_path, index=False)
    logger.success(f"Saved model comparison summary: {summary_path}")
    logger.info(f"Top model by test ROC-AUC: {summary.iloc[0]['model']}")


if __name__ == "__main__":
    app()
