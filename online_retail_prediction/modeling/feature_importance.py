from pathlib import Path

from joblib import load
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
import typer

from online_retail_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

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
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return result


def get_coefficient_importance(
    model, feature_names: list[str], normalize: bool = True
) -> pd.DataFrame:
    """Return feature importance from logistic regression coefficients.

    Uses absolute values of coefficients as importance measure.
    For binary classification, extracts coef_[0].

    Returns DataFrame with columns: feature, importance, coefficient (signed value).
    """
    if not hasattr(model, "coef_"):
        model_name = type(model).__name__
        raise ValueError(
            f"{model_name} does not expose coef_. "
            "Use permutation importance or model-based importance instead."
        )

    coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
    abs_coefficients = np.abs(coefficients)

    if normalize and abs_coefficients.sum() > 0:
        importances = abs_coefficients / abs_coefficients.sum()
    else:
        importances = abs_coefficients

    result = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
                "coefficient": coefficients,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return result


def get_shap_importance(
    model,
    X: pd.DataFrame,
    sample_size: int | None = 100,
) -> pd.DataFrame:
    """Compute SHAP values for linear models (e.g., logistic regression).

    Uses shap.LinearExplainer for efficiency with linear models.

    Args:
        model: A fitted linear model with coef_ attribute.
        X: Feature data for computing SHAP values.
        sample_size: Number of samples to use. None uses all data.

    Returns DataFrame with columns: feature, importance, mean_abs_shap.
    """
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    result = (
        pd.DataFrame(
            {
                "feature": X.columns.tolist(),
                "importance": mean_abs_shap / mean_abs_shap.sum()
                if mean_abs_shap.sum() > 0
                else mean_abs_shap,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
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


def plot_shap_beeswarm(
    model,
    X: pd.DataFrame,
    sample_size: int | None = None,
    max_display: int = 10,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Create SHAP beeswarm plot showing feature impact distribution.

    Args:
        model: A fitted linear model with coef_ attribute.
        X: Feature data for computing SHAP values.
        sample_size: Number of samples to use. None uses all data.
        max_display: Maximum number of features to display.
        output_path: Path to save the figure. If None, displays interactively.
        figsize: Figure size as (width, height).

    Returns:
        matplotlib Figure object.
    """
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Beeswarm Plot", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved SHAP beeswarm plot to {}", output_path)
        plt.close()

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Feature Importance",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot horizontal bar chart of top N feature importances.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' (or 'importance_mean') columns.
        top_n: Number of top features to display.
        title: Plot title.
        output_path: Path to save the figure. If None, displays interactively.
        figsize: Figure size as (width, height).

    Returns:
        matplotlib Figure object.
    """
    importance_col = "importance" if "importance" in importance_df.columns else "importance_mean"

    df = importance_df.head(top_n).copy()
    df = df.iloc[::-1]  # Reverse for horizontal bar (top feature at top)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(df)))[::-1]
    bars = ax.barh(
        df["feature"],
        df[importance_col],
        height=0.5,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    max_val = df[importance_col].max()
    offset = max_val * 0.02  # 2% of max value for proportional spacing

    for bar, val in zip(bars, df[importance_col]):
        ax.text(
            val + offset, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance plot to {}", output_path)

    return fig


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features_first_n.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "baseline_labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "feature_importance.csv",
    method: str = "auto",
    n_repeats: int = 50,
    sample_size: int | None = None,
    plot: bool = False,
    plot_path: Path = REPORTS_DIR / "figures" / "feature_importance.png",
    top_n: int = 10,
    beeswarm: bool = False,
    beeswarm_path: Path = REPORTS_DIR / "figures" / "shap_beeswarm.png",
):
    """
    Compute and save feature importance.

    `method` options:
    - `auto`: use coefficients for linear models, model-based for trees, else permutation.
    - `coefficients`: use logistic regression coefficients (requires coef_ attribute).
    - `shap`: use SHAP values via LinearExplainer (for linear models).
    - `model`: force model-based importances (requires feature_importances_).
    - `permutation`: force permutation importances (works with any model).

    Use `--beeswarm` to generate SHAP beeswarm plot showing feature impact distribution.
    """
    logger.info("Loading features, labels, and model...")
    X = pd.read_csv(features_path)
    if "session_id" in X.columns:
        X = X.drop(columns=["session_id"])
    labels_df = pd.read_csv(labels_path)
    y = labels_df["label"] if "label" in labels_df.columns else labels_df.iloc[:, 0]
    model = load(model_path)

    chosen_method = method.lower().strip()
    valid_methods = {"auto", "coefficients", "shap", "model", "permutation"}
    if chosen_method not in valid_methods:
        raise typer.BadParameter(f"Invalid method '{method}'. Use one of {valid_methods}.")

    if chosen_method == "auto":
        if hasattr(model, "coef_"):
            chosen_method = "coefficients"
        elif hasattr(model, "feature_importances_"):
            chosen_method = "model"
        else:
            chosen_method = "permutation"

    logger.info("Computing feature importance using '{}' method...", chosen_method)

    if chosen_method == "coefficients":
        importance_df = get_coefficient_importance(model=model, feature_names=X.columns.tolist())
    elif chosen_method == "shap":
        importance_df = get_shap_importance(model=model, X=X, sample_size=sample_size)
    elif chosen_method == "model":
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

    if plot:
        title = f"Top {top_n} Features ({chosen_method.title()} Method)"
        plot_feature_importance(
            importance_df=importance_df,
            top_n=top_n,
            title=title,
            output_path=plot_path,
        )
        logger.success("Saved feature importance plot to {}", plot_path)

    if beeswarm:
        logger.info("Generating SHAP beeswarm plot...")
        plot_shap_beeswarm(
            model=model,
            X=X,
            sample_size=sample_size,
            max_display=top_n,
            output_path=beeswarm_path,
        )
        logger.success("Saved SHAP beeswarm plot to {}", beeswarm_path)


if __name__ == "__main__":
    app()
