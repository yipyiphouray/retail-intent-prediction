"""Feature importance analysis for the stacking ensemble classifier."""

from __future__ import annotations

from pathlib import Path

from joblib import load
from loguru import logger
import pandas as pd
from sklearn.pipeline import Pipeline
import typer

from online_retail_prediction.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
)
from online_retail_prediction.modeling.bagging_train import (
    load_processed_features_and_labels,
    split_data,
)
from online_retail_prediction.modeling.feature_importance import (
    get_coefficient_importance,
    get_model_feature_importance,
    get_permutation_feature_importance,
    plot_feature_importance,
)

app = typer.Typer()


@app.command()
def main(
    model_path: Path = MODELS_DIR / "stacking_ensemble_model.pkl",
    features_path: Path = PROCESSED_DATA_DIR / "features_first_n.csv",
    cluster_assignments_path: Path = PROJ_ROOT / "data" / "cluster_outputs" / "cluster_assignments.csv",
    cluster_labels_path: Path = PROJ_ROOT / "data" / "cluster_outputs" / "cluster_label.csv",
    output_dir: Path = FIGURES_DIR,
    csv_dir: Path = PROCESSED_DATA_DIR,
    top_n: int = 10,
    n_repeats: int = 30,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Compute and plot feature importance for the stacking ensemble and its base estimators."""
    logger.info("STACKING FEATURE IMPORTANCE ANALYSIS")

    logger.info("Loading stacking model from {}", model_path)
    stacking_model = load(model_path)

    X, y = load_processed_features_and_labels(
        features_path=features_path,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
    )
    _, X_test, _, y_test = split_data(X, y, test_size=test_size, random_state=random_state)

    feature_names = X_test.columns.tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # --- Permutation importance on full stacking ensemble ---
    logger.info("Computing permutation importance on full stacking ensemble...")
    perm_df = get_permutation_feature_importance(
        model=stacking_model,
        X=X_test,
        y=y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )
    perm_csv = csv_dir / "stacking_permutation_importance.csv"
    perm_df.to_csv(perm_csv, index=False)
    logger.success("Saved permutation importance CSV: {}", perm_csv)

    plot_feature_importance(
        importance_df=perm_df,
        top_n=top_n,
        title=f"Stacking Ensemble — Top {top_n} Features (Permutation Importance)",
        output_path=output_dir / "stacking_permutation_importance.png",
    )
    logger.success("Saved permutation importance plot")

    # --- Per-base-estimator importance ---
    for (name, _), fitted_estimator in zip(stacking_model.estimators, stacking_model.estimators_):
        logger.info("Computing feature importance for base estimator: {}", name)

        if isinstance(fitted_estimator, Pipeline):
            # LR is inside a Pipeline(scaler, classifier) — extract the classifier
            classifier = fitted_estimator.named_steps["classifier"]
            importance_df = get_coefficient_importance(
                model=classifier, feature_names=feature_names
            )
        else:
            importance_df = get_model_feature_importance(
                model=fitted_estimator, feature_names=feature_names
            )

        base_csv = csv_dir / f"stacking_{name}_importance.csv"
        importance_df.to_csv(base_csv, index=False)
        logger.success("Saved {} importance CSV: {}", name, base_csv)

        plot_feature_importance(
            importance_df=importance_df,
            top_n=top_n,
            title=f"Stacking Base — {name.replace('_', ' ').title()} (Top {top_n} Features)",
            output_path=output_dir / f"stacking_{name}_importance.png",
        )
        logger.success("Saved {} importance plot", name)

    logger.success("Stacking feature importance analysis complete!")


if __name__ == "__main__":
    app()
