from __future__ import annotations

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import typer

from online_retail_prediction.config import FIGURES_DIR, PROJ_ROOT, RAW_DATA_DIR, REPORTS_DIR
from online_retail_prediction.modeling.RNN_train import (
<<<<<<< HEAD
	cross_validate_rnn_model,
	prepare_rnn_training_data,
=======
	prepare_rnn_training_data,
	train_rnn_model,
>>>>>>> origin
)

app = typer.Typer()
CLUSTER_OUTPUTS_DIR = PROJ_ROOT / "data" / "cluster_outputs"


def _plot_results(results: pd.DataFrame, output_path: Path) -> None:
	plt.figure(figsize=(10, 6))
<<<<<<< HEAD
	plot_metrics = [
		("cv_mean_test_roc_auc", "CV Mean Test ROC-AUC", "o"),
		("cv_mean_test_precision", "CV Mean Test Precision", "s"),
		("cv_mean_test_recall", "CV Mean Test Recall", "^"),
		("cv_mean_test_f1", "CV Mean Test F1", "v"),
		("cv_mean_test_macro_f1", "CV Mean Test Macro F1", "D"),
		("cv_mean_test_cohen_kappa", "CV Mean Test Cohen's Kappa", "P"),
		("cv_mean_test_accuracy", "CV Mean Test Accuracy", "X"),
	]
	for column, label, marker in plot_metrics:
		plt.plot(results["n_clicks"], results[column], marker=marker, label=label)
	plt.xlabel("Number of first clicks used (n)")
	plt.ylabel("Metric value")
	plt.title("RNN cross-validated test metrics by first-n-click window")
=======
	plt.plot(results["n_clicks"], results["test_roc_auc"], marker="o", label="Test ROC-AUC")
	plt.plot(results["n_clicks"], results["train_roc_auc"], marker="s", label="Train ROC-AUC")
	plt.plot(results["n_clicks"], results["test_accuracy"], marker="^", label="Test Accuracy")
	plt.plot(results["n_clicks"], results["train_accuracy"], marker="v", label="Train Accuracy")
	plt.xlabel("Number of first clicks used (n)")
	plt.ylabel("Metric value")
	plt.title("RNN performance by first-n-click window")
>>>>>>> origin
	plt.ylim(0.0, 1.0)
	plt.grid(alpha=0.25)
	plt.legend()
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=180)
	plt.close()


@app.command()
def main(
	raw_data_path: Path = RAW_DATA_DIR / "e-shop clothing 2008.csv",
	cluster_assignments_path: Path = CLUSTER_OUTPUTS_DIR / "cluster_assignments.csv",
	cluster_labels_path: Path = CLUSTER_OUTPUTS_DIR / "cluster_label.csv",
	start_n: int = 2,
	end_n: int = 10,
	hidden_size: int = 8,
	learning_rate: float = 0.01,
	epochs: int = 1,
<<<<<<< HEAD
	n_splits: int = 5,
=======
	test_size: float = 0.2,
>>>>>>> origin
	random_state: int = 42,
	metrics_output_path: Path = REPORTS_DIR / "rnn_n_clicks_metrics.csv",
	plot_output_path: Path = FIGURES_DIR / "rnn_n_clicks_performance.png",
) -> None:
<<<<<<< HEAD
	"""Run session-level CV RNN for n in [start_n, end_n], optimize by ROC-AUC, and save outputs."""
=======
	"""Run RNN training for n in [start_n, end_n], visualize metrics, and save outputs."""
>>>>>>> origin

	if start_n <= 0 or end_n <= 0:
		raise ValueError("start_n and end_n must be greater than 0")
	if start_n > end_n:
		raise ValueError("start_n must be less than or equal to end_n")

	logger.info(f"Loading raw clickstream from {raw_data_path}...")
	clickstream = pd.read_csv(raw_data_path, sep=";")

	all_results: list[dict[str, float | int]] = []
	for n_clicks in range(start_n, end_n + 1):
		logger.info(f"Training RNN with first {n_clicks} clicks per session...")
		dataset = prepare_rnn_training_data(
			clickstream=clickstream,
			n_clicks=n_clicks,
			cluster_assignments_path=cluster_assignments_path,
			cluster_labels_path=cluster_labels_path,
		)
<<<<<<< HEAD
		_, metrics = cross_validate_rnn_model(
=======
		_, metrics, _ = train_rnn_model(
>>>>>>> origin
			dataset=dataset,
			hidden_size=hidden_size,
			learning_rate=learning_rate,
			epochs=epochs,
<<<<<<< HEAD
			n_splits=n_splits,
=======
			test_size=test_size,
>>>>>>> origin
			random_state=random_state,
		)
		all_results.append({"n_clicks": n_clicks, **metrics})
		logger.info(
<<<<<<< HEAD
			"n={} | cv_mean_test_roc_auc={:.4f} (+/-{:.4f}), cv_mean_test_precision={:.4f}, "
			"cv_mean_test_recall={:.4f}, cv_mean_test_f1={:.4f}, cv_mean_test_macro_f1={:.4f}, "
			"cv_mean_test_cohen_kappa={:.4f}, cv_mean_test_accuracy={:.4f}",
			n_clicks,
			metrics["cv_mean_test_roc_auc"],
			metrics["cv_std_test_roc_auc"],
			metrics["cv_mean_test_precision"],
			metrics["cv_mean_test_recall"],
			metrics["cv_mean_test_f1"],
			metrics["cv_mean_test_macro_f1"],
			metrics["cv_mean_test_cohen_kappa"],
			metrics["cv_mean_test_accuracy"],
		)

	results_df = pd.DataFrame(all_results).sort_values("n_clicks").reset_index(drop=True)
	best_idx = results_df["cv_mean_test_roc_auc"].idxmax()
	best_result = results_df.loc[best_idx]

=======
			"n={} | test_accuracy={:.4f}, test_roc_auc={:.4f}",
			n_clicks,
			metrics["test_accuracy"],
			metrics["test_roc_auc"],
		)

	results_df = pd.DataFrame(all_results).sort_values("n_clicks").reset_index(drop=True)
>>>>>>> origin
	metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
	results_df.to_csv(metrics_output_path, index=False)

	_plot_results(results=results_df, output_path=plot_output_path)

<<<<<<< HEAD
	logger.success(
		"Best n_clicks by ROC-AUC: n_clicks={} with cv_mean_test_roc_auc={:.4f} (+/-{:.4f})",
		int(best_result["n_clicks"]),
		best_result["cv_mean_test_roc_auc"],
		best_result["cv_std_test_roc_auc"],
	)
=======
>>>>>>> origin
	logger.success(f"Saved tuning metrics to {metrics_output_path}")
	logger.success(f"Saved tuning plot to {plot_output_path}")


if __name__ == "__main__":
	app()
