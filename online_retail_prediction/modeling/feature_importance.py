from pathlib import Path

import pandas as pd
import typer
from joblib import load
from loguru import logger
from sklearn.inspection import permutation_importance

from online_retail_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def get_model_feature_importance(
    model, feature_names: list[str], normalize: bool = True
) -> pd.DataFrame:
    """Return feature importance for estimators exposing `feature_importances_`."""
    if not hasattr(model, "feature_importances_"):
        model_name = type(model).__name__
        raise ValueError(
            f"{model_name} does not expose feature_importances_. "
            "Use permutation importance instead."
        )

    importances = pd.Series(model.feature_importances_, index=feature_names, dtype=float)
    if normalize and importances.sum() > 0:
        importances = importances / importances.sum()

    result = (
        importances.sort_values(ascending=False)
        .rename("importance")
        .reset_index(names="feature")
    )
    return result


def get_permutation_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str | None = None,
) -> pd.DataFrame:
    """Return permutation importance with mean and std across shuffles."""
    importance_result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )

    result = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": importance_result.importances_mean,
            "importance_std": importance_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return result.reset_index(drop=True)


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "feature_importance.csv",
    method: str = "auto",
    n_repeats: int = 10,
):
    """
    Compute and save feature importance.

    `method` options:
    - `auto`: use model-based importances when available, otherwise permutation.
    - `model`: force model-based importances.
    - `permutation`: force permutation importances.
    """
    logger.info("Loading features, labels, and model...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze("columns")
    model = load(model_path)

    chosen_method = method.lower().strip()
    valid_methods = {"auto", "model", "permutation"}
    if chosen_method not in valid_methods:
        raise typer.BadParameter(f"Invalid method '{method}'. Use one of {valid_methods}.")

    if chosen_method == "auto":
        chosen_method = "model" if hasattr(model, "feature_importances_") else "permutation"

    logger.info("Computing feature importance using '{}' method...", chosen_method)
    if chosen_method == "model":
        importance_df = get_model_feature_importance(model=model, feature_names=X.columns.tolist())
    else:
        importance_df = get_permutation_feature_importance(
            model=model,
            X=X,
            y=y,
            n_repeats=n_repeats,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    logger.success("Saved feature importance to {}", output_path)


if __name__ == "__main__":
    app()
