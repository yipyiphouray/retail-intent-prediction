from pathlib import Path

from joblib import load
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
import typer

from online_retail_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()


def plot_learning_curve(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    train_sizes: np.ndarray | None = None,
    scoring: str = "accuracy",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot learning curve showing train/validation scores vs training set size.

    Args:
        model: Estimator instance (unfitted).
        X: Feature data.
        y: Target labels.
        cv: Number of cross-validation folds.
        train_sizes: Array of training set sizes to evaluate.
        scoring: Scoring metric.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="blue")
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color="orange")

    ax.plot(train_sizes_abs, train_mean, "o-", color="blue", label="Training Score")
    ax.plot(train_sizes_abs, val_mean, "o-", color="orange", label="Validation Score")

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel(f"{scoring.title()}", fontsize=12)
    ax.set_title("Learning Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved learning curve to {}", output_path)

    return fig


def plot_validation_curve(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    param_name: str,
    param_range: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (10, 6),
    log_scale: bool = True,
) -> plt.Figure:
    """Plot validation curve showing train/validation scores vs hyperparameter.

    Args:
        model: Estimator instance (unfitted).
        X: Feature data.
        y: Target labels.
        param_name: Name of hyperparameter to vary.
        param_range: Array of hyperparameter values to evaluate.
        cv: Number of cross-validation folds.
        scoring: Scoring metric.
        output_path: Path to save the figure.
        figsize: Figure size.
        log_scale: Whether to use log scale for x-axis.

    Returns:
        matplotlib Figure object.
    """
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="blue")
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color="orange")

    ax.plot(param_range, train_mean, "o-", color="blue", label="Training Score")
    ax.plot(param_range, val_mean, "o-", color="orange", label="Validation Score")

    if log_scale:
        ax.set_xscale("log")

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel(f"{scoring.title()}", fontsize=12)
    ax.set_title(f"Validation Curve ({param_name})", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved validation curve to {}", output_path)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (8, 6),
    normalize: bool = False,
) -> plt.Figure:
    """Plot confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class labels for axes.
        output_path: Path to save the figure.
        figsize: Figure size.
        normalize: Whether to normalize by row (true class).

    Returns:
        matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    if labels is None:
        labels = ["No Purchase", "Purchase"]

    fig, ax = plt.subplots(figsize=figsize)

    fmt = ".2f" if normalize else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, cbar_kws={"shrink": 0.8})

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved confusion matrix to {}", output_path)

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot ROC curve with AUC score.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for positive class.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")

    ax.fill_between(fpr, tpr, alpha=0.2, color="blue")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved ROC curve to {}", output_path)

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot Precision-Recall curve with AUC score.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for positive class.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    # Baseline is the proportion of positive class
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, color="blue", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    ax.axhline(y=baseline, color="gray", lw=1, linestyle="--", label=f"Baseline ({baseline:.3f})")

    ax.fill_between(recall, precision, alpha=0.2, color="blue")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved precision-recall curve to {}", output_path)

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot calibration curve (reliability diagram).

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for positive class.
        n_bins: Number of bins for calibration.
        output_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(prob_pred, prob_true, "o-", color="blue", lw=2, label="Logistic Regression")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Perfectly Calibrated")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved calibration curve to {}", output_path)

    return fig


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    output_dir: Path = REPORTS_DIR / "figures",
    cv: int = 5,
    scoring: str = "accuracy",
):
    """Generate learning curve and validation curves for logistic regression."""
    logger.info("Loading features and labels...")
    X = pd.read_csv(features_path)
    if "session_id" in X.columns:
        X = X.drop(columns=["session_id"])
    labels_df = pd.read_csv(labels_path)
    y = labels_df["label"] if "label" in labels_df.columns else labels_df.iloc[:, 0]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Learning Curve
    logger.info("Generating learning curve...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    plot_learning_curve(
        model=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        output_path=output_dir / "learning_curve.png",
    )
    logger.success("Saved learning curve")

    # Validation Curve for C (regularization)
    logger.info("Generating validation curve for C (regularization)...")
    c_range = np.array([0.001, 0.01, 0.1, 1, 10, 100])
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    plot_validation_curve(
        model=model,
        X=X,
        y=y,
        param_name="C",
        param_range=c_range,
        cv=cv,
        scoring=scoring,
        output_path=output_dir / "validation_curve_C.png",
        log_scale=True,
    )
    logger.success("Saved validation curve for C")

    # Validation Curve for max_iter
    logger.info("Generating validation curve for max_iter...")
    iter_range = np.array([50, 100, 200, 500, 1000, 2000])
    model = LogisticRegression(class_weight="balanced", random_state=42)
    plot_validation_curve(
        model=model,
        X=X,
        y=y,
        param_name="max_iter",
        param_range=iter_range,
        cv=cv,
        scoring=scoring,
        output_path=output_dir / "validation_curve_max_iter.png",
        log_scale=False,
    )
    logger.success("Saved validation curve for max_iter")

    # Train model for classification plots
    logger.info("Training model for classification plots...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    logger.info("Generating confusion matrix...")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        output_path=output_dir / "confusion_matrix.png",
    )
    logger.success("Saved confusion matrix")

    # Normalized Confusion Matrix
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        output_path=output_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    logger.success("Saved normalized confusion matrix")

    # ROC Curve
    logger.info("Generating ROC curve...")
    plot_roc_curve(
        y_true=y_test,
        y_prob=y_prob,
        output_path=output_dir / "roc_curve.png",
    )
    logger.success("Saved ROC curve")

    # Precision-Recall Curve
    logger.info("Generating precision-recall curve...")
    plot_precision_recall_curve(
        y_true=y_test,
        y_prob=y_prob,
        output_path=output_dir / "precision_recall_curve.png",
    )
    logger.success("Saved precision-recall curve")

    # Calibration Curve
    logger.info("Generating calibration curve...")
    plot_calibration_curve(
        y_true=y_test,
        y_prob=y_prob,
        output_path=output_dir / "calibration_curve.png",
    )
    logger.success("Saved calibration curve")

    logger.success("All plots generated in {}", output_dir)


if __name__ == "__main__":
    app()
