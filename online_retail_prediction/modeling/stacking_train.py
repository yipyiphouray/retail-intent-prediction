"""Stacking ensemble classifier with XGBoost, LightGBM, Random Forest, and Logistic Regression."""

from __future__ import annotations

from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
from loguru import logger
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer
from xgboost import XGBClassifier

from online_retail_prediction.config import MODELS_DIR, PROJ_ROOT, PROCESSED_DATA_DIR
from online_retail_prediction.modeling.bagging_train import (
    evaluate_model,
    load_processed_features_and_labels,
    split_data,
)

app = typer.Typer()


def build_base_estimator_configs(
    random_state: int = 42,
) -> dict[str, tuple[object, dict[str, list[object]]]]:
    """Build base estimator configs and tuning grids for stacking."""
    configs: dict[str, tuple[object, dict[str, list[object]]]] = {
        "logistic_regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            {
                "classifier__C": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 25.0, 50.0, 75.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2500.0, 5000.0, 7500.0, 10000.0, 25000.0, 50000.0, 100000.0],
                "classifier__solver": ["saga"],
                "classifier__penalty": ["l1", "l2"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "xgboost": (
            XGBClassifier(
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
        "lightgbm": (
            LGBMClassifier(
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
    }
    return configs


def tune_base_estimator(
    name: str,
    estimator,
    param_grid: dict[str, list[object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> tuple[object, dict[str, object], float, pd.DataFrame]:
    """Tune a single base estimator via GridSearchCV on ROC-AUC.

    Returns:
        Tuple of (best_estimator, best_params, best_score, cv_results).
    """
    logger.info(f"Tuning {name}...")

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    logger.success(f"Best ROC-AUC for {name}: {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters for {name}: {grid_search.best_params_}")

    cv_results = pd.DataFrame(grid_search.cv_results_)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, cv_results


def build_stacking_classifier(
    tuned_estimators: list[tuple[str, object]],
    cv: int = 5,
    random_state: int = 42,
) -> StackingClassifier:
    """Build a StackingClassifier from pre-tuned base estimators."""
    return StackingClassifier(
        estimators=tuned_estimators,
        final_estimator=LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1,
    )


def tune_meta_learner(
    stacking_model: StackingClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> tuple[StackingClassifier, dict[str, object], float, pd.DataFrame]:
    """Tune the meta-learner C via GridSearchCV on the assembled stacking classifier.

    Returns:
        Tuple of (best_stacking_model, best_params, best_score, cv_results).
    """
    param_grid = {
        "final_estimator__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    }

    logger.info("Tuning meta-learner (final_estimator C)...")
    grid_search = GridSearchCV(
        estimator=stacking_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    logger.success(f"Best meta-learner ROC-AUC: {grid_search.best_score_:.4f}")
    logger.info(f"Best meta-learner params: {grid_search.best_params_}")

    cv_results = pd.DataFrame(grid_search.cv_results_)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, cv_results


def save_results(
    model,
    tuning_summary: list[dict[str, object]],
    cv_results: dict[str, pd.DataFrame],
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Save stacking model, metrics, cv_results, and individual best base estimators."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "stacking_ensemble_model.pkl"
    metrics_path = output_dir / "stacking_ensemble_metrics.txt"
    comparison_path = output_dir / "stacking_ensemble_base_comparison.csv"

    joblib.dump(model, model_path)

    # Save individual best base estimators for downstream analysis
    # model.estimators_ = fitted estimators (list), model.estimators = named tuples (list of (name, est))
    for (name, _), fitted_estimator in zip(model.estimators, model.estimators_):
        estimator_path = output_dir / f"stacking_base_{name}.pkl"
        joblib.dump(fitted_estimator, estimator_path)
        logger.success(f"Saved best {name}: {estimator_path}")

    # Save cv_results for each estimator and meta-learner (validation/learning curves)
    cv_results_dir = output_dir / "stacking_cv_results"
    cv_results_dir.mkdir(parents=True, exist_ok=True)
    for name, cv_df in cv_results.items():
        cv_path = cv_results_dir / f"stacking_{name}_cv_results.csv"
        cv_df.to_csv(cv_path, index=False)
        logger.success(f"Saved cv_results for {name}: {cv_path}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("MODEL: stacking_ensemble\n")
        f.write("META_LEARNER: LogisticRegression(balanced)\n\n")

        f.write("BASE_ESTIMATOR_TUNING\n")
        for row in tuning_summary:
            f.write(f"\n{row['name']}\n")
            f.write(f"  best_cv_roc_auc: {row['best_cv_roc_auc']:.6f}\n")
            f.write(f"  best_params: {row['best_params']}\n")

        f.write("\nTRAIN_METRICS\n")
        for metric_name, metric_value in train_metrics.items():
            f.write(f"{metric_name.upper()}: {metric_value:.6f}\n")

        f.write("\nTEST_METRICS\n")
        for metric_name, metric_value in test_metrics.items():
            f.write(f"{metric_name.upper()}: {metric_value:.6f}\n")

    summary_df = pd.DataFrame(tuning_summary).sort_values(by="best_cv_roc_auc", ascending=False)
    summary_df.to_csv(comparison_path, index=False)

    logger.success(f"Saved model: {model_path}")
    logger.success(f"Saved metrics: {metrics_path}")
    logger.success(f"Saved base estimator comparison: {comparison_path}")


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features_first_n.csv",
    cluster_assignments_path: Path = PROJ_ROOT / "data" / "cluster_outputs" / "cluster_assignments.csv",
    cluster_labels_path: Path = PROJ_ROOT / "data" / "cluster_outputs" / "cluster_label.csv",
    output_dir: Path = MODELS_DIR,
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 42,
):
    """Train a stacking ensemble with tuned LR, RF, XGBoost, and LightGBM base estimators."""
    logger.info("STACKING ENSEMBLE TRAINING PIPELINE")

    X, y = load_processed_features_and_labels(
        features_path=features_path,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
    )
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    configs = build_base_estimator_configs(random_state=random_state)
    tuned_estimators: list[tuple[str, object]] = []
    tuning_summary: list[dict[str, object]] = []
    cv_results: dict[str, pd.DataFrame] = {}

    for name, (estimator, param_grid) in configs.items():
        best_model, best_params, best_score, cv_df = tune_base_estimator(
            name=name,
            estimator=estimator,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )
        tuned_estimators.append((name, best_model))
        tuning_summary.append(
            {"name": name, "best_cv_roc_auc": best_score, "best_params": best_params}
        )
        cv_results[name] = cv_df

    logger.info("Building stacking classifier from tuned base estimators...")
    stacking_model = build_stacking_classifier(
        tuned_estimators=tuned_estimators, cv=cv, random_state=random_state
    )

    stacking_model, meta_params, meta_score, meta_cv_df = tune_meta_learner(
        stacking_model=stacking_model,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
    )
    tuning_summary.append(
        {"name": "meta_learner", "best_cv_roc_auc": meta_score, "best_params": meta_params}
    )
    cv_results["meta_learner"] = meta_cv_df

    train_metrics = evaluate_model(stacking_model, X_train, y_train)
    test_metrics = evaluate_model(stacking_model, X_test, y_test)

    logger.info("STACKING ENSEMBLE RESULTS")
    for metric_name in train_metrics:
        train_val = train_metrics[metric_name]
        test_val = test_metrics[metric_name]
        diff = train_val - test_val
        logger.info(
            f"{metric_name.upper()}: Train={train_val:.4f}, Test={test_val:.4f}, Diff={diff:+.4f}"
        )

    save_results(
        model=stacking_model,
        tuning_summary=tuning_summary,
        cv_results=cv_results,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        output_dir=output_dir,
    )

    logger.success("Stacking pipeline complete!")


if __name__ == "__main__":
    app()
