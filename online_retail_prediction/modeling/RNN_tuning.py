from __future__ import annotations

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import typer

from online_retail_prediction.config import FIGURES_DIR, PROJ_ROOT, RAW_DATA_DIR, REPORTS_DIR
from online_retail_prediction.modeling.RNN_train import (
	prepare_rnn_training_data,
	train_rnn_model,
)

app = typer.Typer()
CLUSTER_OUTPUTS_DIR = PROJ_ROOT / "data" / "cluster_outputs"


def _plot_results(results: pd.DataFrame, output_path: Path) -> None:
	plt.figure(figsize=(10, 6))
	plt.plot(results["n_clicks"], results["test_roc_auc"], marker="o", label="Test ROC-AUC")
	plt.plot(results["n_clicks"], results["train_roc_auc"], marker="s", label="Train ROC-AUC")
	plt.plot(results["n_clicks"], results["test_accuracy"], marker="^", label="Test Accuracy")
	plt.plot(results["n_clicks"], results["train_accuracy"], marker="v", label="Train Accuracy")
	plt.xlabel("Number of first clicks used (n)")
	plt.ylabel("Metric value")
	plt.title("RNN performance by first-n-click window")
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
	test_size: float = 0.2,
	random_state: int = 42,
	metrics_output_path: Path = REPORTS_DIR / "rnn_n_clicks_metrics.csv",
	plot_output_path: Path = FIGURES_DIR / "rnn_n_clicks_performance.png",
) -> None:
	"""Run RNN training for n in [start_n, end_n], visualize metrics, and save outputs."""

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
		_, metrics, _ = train_rnn_model(
			dataset=dataset,
			hidden_size=hidden_size,
			learning_rate=learning_rate,
			epochs=epochs,
			test_size=test_size,
			random_state=random_state,
		)
		all_results.append({"n_clicks": n_clicks, **metrics})
		logger.info(
			"n={} | test_accuracy={:.4f}, test_roc_auc={:.4f}",
			n_clicks,
			metrics["test_accuracy"],
			metrics["test_roc_auc"],
		)

	results_df = pd.DataFrame(all_results).sort_values("n_clicks").reset_index(drop=True)
	metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
	results_df.to_csv(metrics_output_path, index=False)

	_plot_results(results=results_df, output_path=plot_output_path)

	logger.success(f"Saved tuning metrics to {metrics_output_path}")
	logger.success(f"Saved tuning plot to {plot_output_path}")


if __name__ == "__main__":
	app()
