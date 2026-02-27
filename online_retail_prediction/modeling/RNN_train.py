from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
<<<<<<< HEAD
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
=======
<<<<<<< HEAD
from sklearn.metrics import (
	accuracy_score,
	cohen_kappa_score,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
=======
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
>>>>>>> origin
>>>>>>> origin/dev
import typer

from online_retail_prediction.config import MODELS_DIR, PROJ_ROOT, RAW_DATA_DIR
from online_retail_prediction.modeling.clicks_feature_engineering import (
	build_click_level_features,
)

app = typer.Typer()
CLUSTER_OUTPUTS_DIR = PROJ_ROOT / "data" / "cluster_outputs"


def _sigmoid(values: np.ndarray) -> np.ndarray:
	clipped = np.clip(values, -30.0, 30.0)
	return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class SequenceDataset:
	sequences: list[np.ndarray]
	labels: np.ndarray
	session_ids: np.ndarray
	feature_names: list[str]


class SimpleSessionRNN:
	def __init__(
		self,
		input_size: int,
		hidden_size: int = 32,
		learning_rate: float = 0.01,
		random_state: int = 42,
	):
		rng = np.random.default_rng(random_state)
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.learning_rate = learning_rate

		self.w_xh = rng.normal(0.0, 0.1, size=(hidden_size, input_size))
		self.w_hh = rng.normal(0.0, 0.1, size=(hidden_size, hidden_size))
		self.b_h = np.zeros(hidden_size)

		self.w_hy = rng.normal(0.0, 0.1, size=(hidden_size,))
		self.b_y = 0.0

	def _forward(self, sequence: np.ndarray) -> tuple[float, dict[str, list[np.ndarray] | float]]:
		hidden_state = np.zeros(self.hidden_size)
		hidden_states: list[np.ndarray] = []
		pre_activations: list[np.ndarray] = []

		for row in sequence:
			pre_activation = self.w_xh @ row + self.w_hh @ hidden_state + self.b_h
			hidden_state = np.tanh(pre_activation)
			pre_activations.append(pre_activation)
			hidden_states.append(hidden_state.copy())

		logit = float(self.w_hy @ hidden_state + self.b_y)
		probability = float(_sigmoid(np.array([logit]))[0])
		cache = {
			"hidden_states": hidden_states,
			"pre_activations": pre_activations,
			"logit": logit,
		}
		return probability, cache

	def _backward(
		self,
		sequence: np.ndarray,
		label: float,
		cache: dict[str, list[np.ndarray] | float],
	) -> None:
		hidden_states = cache["hidden_states"]
		logit = cache["logit"]

		probability = float(_sigmoid(np.array([float(logit)]))[0])
		d_logit = probability - label

		grad_w_hy = d_logit * hidden_states[-1]
		grad_b_y = d_logit

		grad_w_xh = np.zeros_like(self.w_xh)
		grad_w_hh = np.zeros_like(self.w_hh)
		grad_b_h = np.zeros_like(self.b_h)

		grad_next_hidden = d_logit * self.w_hy
		for t in reversed(range(len(sequence))):
			hidden_t = hidden_states[t]
			hidden_prev = np.zeros(self.hidden_size) if t == 0 else hidden_states[t - 1]
			input_t = sequence[t]

			d_pre_activation = grad_next_hidden * (1.0 - hidden_t * hidden_t)
			grad_w_xh += np.outer(d_pre_activation, input_t)
			grad_w_hh += np.outer(d_pre_activation, hidden_prev)
			grad_b_h += d_pre_activation

			grad_next_hidden = self.w_hh.T @ d_pre_activation

		clip_value = 5.0
		grad_w_xh = np.clip(grad_w_xh, -clip_value, clip_value)
		grad_w_hh = np.clip(grad_w_hh, -clip_value, clip_value)
		grad_b_h = np.clip(grad_b_h, -clip_value, clip_value)
		grad_w_hy = np.clip(grad_w_hy, -clip_value, clip_value)
		grad_b_y = float(np.clip(grad_b_y, -clip_value, clip_value))

		self.w_xh -= self.learning_rate * grad_w_xh
		self.w_hh -= self.learning_rate * grad_w_hh
		self.b_h -= self.learning_rate * grad_b_h
		self.w_hy -= self.learning_rate * grad_w_hy
		self.b_y -= self.learning_rate * grad_b_y

	def fit(self, sequences: list[np.ndarray], labels: np.ndarray, epochs: int = 30) -> list[float]:
		losses: list[float] = []
		rng = np.random.default_rng(42)

		for epoch in range(epochs):
			indices = rng.permutation(len(sequences))
			epoch_losses: list[float] = []
			for idx in indices:
				probability, cache = self._forward(sequences[idx])
				label = float(labels[idx])
				loss = -(label * np.log(probability + 1e-12) + (1 - label) * np.log(1 - probability + 1e-12))
				epoch_losses.append(float(loss))
				self._backward(sequences[idx], label=label, cache=cache)

			mean_loss = float(np.mean(epoch_losses))
			losses.append(mean_loss)
			logger.info(f"Epoch {epoch + 1}/{epochs} - loss: {mean_loss:.4f}")

		return losses

	def predict_proba(self, sequences: list[np.ndarray]) -> np.ndarray:
		probabilities = [self._forward(sequence)[0] for sequence in sequences]
		return np.array(probabilities)


def _extract_numeric_sequence_columns(click_features: pd.DataFrame) -> list[str]:
	numeric_columns = click_features.select_dtypes(include=[np.number]).columns.tolist()
	excluded_columns = {
		"session_id",
		"click_position",
	}
	return [column for column in numeric_columns if column not in excluded_columns]


def _normalize_intent_labels(raw_labels: pd.Series) -> pd.Series:
	if pd.api.types.is_numeric_dtype(raw_labels):
		normalized = raw_labels.astype(int)
	else:
		normalized = (
			raw_labels.astype(str)
			.str.strip()
			.str.lower()
			.map(
				{
					"low-intent": 0,
					"low_intent": 0,
					"low intent": 0,
					"0": 0,
					"high-intent": 1,
					"high_intent": 1,
					"high intent": 1,
					"1": 1,
				}
			)
		)

	if normalized.isna().any():
		invalid_values = sorted(raw_labels[normalized.isna()].astype(str).unique().tolist())
		raise ValueError(f"Unsupported intent labels found in cluster labels: {invalid_values}")

	unique_values = set(normalized.unique().tolist())
	if not unique_values.issubset({0, 1}):
		raise ValueError(
			f"Cluster intent labels must map to binary values 0/1. Found: {sorted(unique_values)}"
		)

	return normalized.astype(int)


def _load_session_labels_from_cluster_files(
	cluster_assignments_path: Path,
	cluster_labels_path: Path,
) -> pd.DataFrame:
	if not cluster_assignments_path.exists():
		raise FileNotFoundError(f"Cluster assignments file not found: {cluster_assignments_path}")
	if not cluster_labels_path.exists():
		raise FileNotFoundError(f"Cluster labels file not found: {cluster_labels_path}")

	cluster_assignments = pd.read_csv(cluster_assignments_path)
	cluster_labels = pd.read_csv(cluster_labels_path)

	required_assignment_columns = {"session_id", "cluster_id"}
	if not required_assignment_columns.issubset(cluster_assignments.columns):
		missing = required_assignment_columns - set(cluster_assignments.columns)
		raise ValueError(f"Missing required cluster assignment columns: {sorted(missing)}")

	if "intent_label" in cluster_labels.columns:
		cluster_label_column = "intent_label"
	elif "label" in cluster_labels.columns:
		cluster_label_column = "label"
	else:
		raise ValueError("Cluster labels file must contain either 'intent_label' or 'label'.")

	labeled_assignments = cluster_assignments.merge(
		cluster_labels[["cluster_id", cluster_label_column]],
		on="cluster_id",
		how="inner",
	)

	if labeled_assignments.empty:
		raise ValueError("Cluster assignments and cluster labels produced an empty merge.")

	raw_labels = labeled_assignments[cluster_label_column].copy()
	labeled_assignments["label"] = _normalize_intent_labels(raw_labels)

	session_label_conflicts = labeled_assignments.groupby("session_id")["label"].nunique()
	conflicting_sessions = session_label_conflicts[session_label_conflicts > 1]
	if not conflicting_sessions.empty:
		sample_conflicts = conflicting_sessions.index.tolist()[:10]
		raise ValueError(
			"Found session_id values with conflicting cluster-derived labels. "
			f"Sample session_ids: {sample_conflicts}"
		)

	return labeled_assignments[["session_id", "label"]].drop_duplicates(
		subset=["session_id"], keep="first"
	)


def _build_session_sequences(
	click_features: pd.DataFrame,
	feature_columns: list[str],
) -> tuple[list[int], list[np.ndarray]]:
	session_ids: list[int] = []
	sequences: list[np.ndarray] = []

	for session_id, group in click_features.groupby("session_id"):
		ordered = group.sort_values("click_position")
		session_ids.append(int(session_id))
		sequences.append(ordered[feature_columns].to_numpy(dtype=float))

	return session_ids, sequences


def _fit_standardization(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
	stacked = np.vstack(sequences)
	mean = stacked.mean(axis=0)
	std = stacked.std(axis=0)
	std[std == 0.0] = 1.0
	return mean, std


def _apply_standardization(
	sequences: list[np.ndarray],
	mean: np.ndarray,
	std: np.ndarray,
) -> list[np.ndarray]:
	return [(sequence - mean) / std for sequence in sequences]


<<<<<<< HEAD
=======
<<<<<<< HEAD
def _compute_split_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_prob: np.ndarray,
	prefix: str,
) -> dict[str, float]:
	return {
		f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
		f"{prefix}_roc_auc": float(roc_auc_score(y_true, y_prob)),
		f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
		f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
		f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
		f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
		f"{prefix}_cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
	}


=======
>>>>>>> origin
>>>>>>> origin/dev
def prepare_rnn_training_data(
	clickstream: pd.DataFrame,
	n_clicks: int = 5,
	cluster_assignments_path: Path = CLUSTER_OUTPUTS_DIR / "cluster_assignments.csv",
	cluster_labels_path: Path = CLUSTER_OUTPUTS_DIR / "cluster_label.csv",
) -> SequenceDataset:
	"""
	Create sequence data trimmed to first n clicks per session.
	"""
	click_features = build_click_level_features(clickstream=clickstream, n_clicks=n_clicks)

	logger.info(
		"Trimmed clickstream to first {} clicks per session before training.",
		n_clicks,
	)

	labels = _load_session_labels_from_cluster_files(
		cluster_assignments_path=cluster_assignments_path,
		cluster_labels_path=cluster_labels_path,
	)

	valid_session_ids = set(labels["session_id"].astype(int).tolist())
	click_features = click_features[click_features["session_id"].isin(valid_session_ids)].copy()
	if click_features.empty:
		raise ValueError("No sessions remain after joining first-N click features with cluster labels.")

	feature_columns = _extract_numeric_sequence_columns(click_features)
	session_ids, sequences = _build_session_sequences(click_features, feature_columns)

	label_map = labels.set_index("session_id")["label"].to_dict()
	y = np.array([int(label_map[session_id]) for session_id in session_ids], dtype=float)

	return SequenceDataset(
		sequences=sequences,
		labels=y,
		session_ids=np.array(session_ids, dtype=int),
		feature_names=feature_columns,
	)


def train_rnn_model(
	dataset: SequenceDataset,
	hidden_size: int = 32,
	learning_rate: float = 0.01,
	epochs: int = 30,
	test_size: float = 0.2,
	random_state: int = 42,
) -> tuple[SimpleSessionRNN, dict[str, float], dict[str, np.ndarray | list[str]]]:
	indices = np.arange(len(dataset.sequences))
	train_idx, test_idx = train_test_split(
		indices,
		test_size=test_size,
		random_state=random_state,
		stratify=dataset.labels.astype(int),
	)

	train_sequences = [dataset.sequences[index] for index in train_idx]
	test_sequences = [dataset.sequences[index] for index in test_idx]
	y_train = dataset.labels[train_idx]
	y_test = dataset.labels[test_idx]

	mean, std = _fit_standardization(train_sequences)
	train_sequences_scaled = _apply_standardization(train_sequences, mean=mean, std=std)
	test_sequences_scaled = _apply_standardization(test_sequences, mean=mean, std=std)

	model = SimpleSessionRNN(
		input_size=len(dataset.feature_names),
		hidden_size=hidden_size,
		learning_rate=learning_rate,
		random_state=random_state,
	)
	model.fit(train_sequences_scaled, y_train, epochs=epochs)

	train_probabilities = model.predict_proba(train_sequences_scaled)
	test_probabilities = model.predict_proba(test_sequences_scaled)

	train_predictions = (train_probabilities >= 0.5).astype(int)
	test_predictions = (test_probabilities >= 0.5).astype(int)

	metrics = {
<<<<<<< HEAD
=======
<<<<<<< HEAD
		**_compute_split_metrics(y_train, train_predictions, train_probabilities, "train"),
		**_compute_split_metrics(y_test, test_predictions, test_probabilities, "test"),
=======
>>>>>>> origin/dev
		"train_accuracy": float(accuracy_score(y_train, train_predictions)),
		"test_accuracy": float(accuracy_score(y_test, test_predictions)),
		"train_roc_auc": float(roc_auc_score(y_train, train_probabilities)),
		"test_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
<<<<<<< HEAD
=======
>>>>>>> origin
>>>>>>> origin/dev
	}

	artifacts = {
		"scaler_mean": mean,
		"scaler_std": std,
		"feature_names": dataset.feature_names,
	}
	return model, metrics, artifacts


<<<<<<< HEAD
=======
<<<<<<< HEAD
def cross_validate_rnn_model(
	dataset: SequenceDataset,
	hidden_size: int = 32,
	learning_rate: float = 0.01,
	epochs: int = 30,
	n_splits: int = 5,
	random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
	if n_splits < 2:
		raise ValueError("n_splits must be at least 2 for cross-validation")

	labels = dataset.labels.astype(int)
	class_counts = np.bincount(labels)
	if len(class_counts) < 2 or class_counts.min() < n_splits:
		raise ValueError(
			"Insufficient samples per class for stratified CV. "
			f"Minimum class count is {class_counts.min() if len(class_counts) > 0 else 0}, "
			f"but n_splits={n_splits}."
		)

	indices = np.arange(len(dataset.sequences))
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

	fold_results: list[dict[str, float | int]] = []
	for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):
		train_sequences = [dataset.sequences[index] for index in train_idx]
		test_sequences = [dataset.sequences[index] for index in test_idx]
		y_train = dataset.labels[train_idx]
		y_test = dataset.labels[test_idx]

		mean, std = _fit_standardization(train_sequences)
		train_sequences_scaled = _apply_standardization(train_sequences, mean=mean, std=std)
		test_sequences_scaled = _apply_standardization(test_sequences, mean=mean, std=std)

		model = SimpleSessionRNN(
			input_size=len(dataset.feature_names),
			hidden_size=hidden_size,
			learning_rate=learning_rate,
			random_state=random_state + fold,
		)
		model.fit(train_sequences_scaled, y_train, epochs=epochs)

		train_probabilities = model.predict_proba(train_sequences_scaled)
		test_probabilities = model.predict_proba(test_sequences_scaled)

		train_predictions = (train_probabilities >= 0.5).astype(int)
		test_predictions = (test_probabilities >= 0.5).astype(int)

		fold_metrics = {
			"fold": fold,
			**_compute_split_metrics(y_train, train_predictions, train_probabilities, "train"),
			**_compute_split_metrics(y_test, test_predictions, test_probabilities, "test"),
		}
		fold_results.append(fold_metrics)

	fold_results_df = pd.DataFrame(fold_results).sort_values("fold").reset_index(drop=True)

	summary_metrics: dict[str, float] = {}
	metric_columns = [column for column in fold_results_df.columns if column != "fold"]
	for column in metric_columns:
		summary_metrics[f"cv_mean_{column}"] = float(fold_results_df[column].mean())
		summary_metrics[f"cv_std_{column}"] = float(fold_results_df[column].std(ddof=0))

	return fold_results_df, summary_metrics


=======
>>>>>>> origin
>>>>>>> origin/dev
def save_rnn_artifacts(
	model: SimpleSessionRNN,
	artifacts: dict[str, np.ndarray | list[str]],
	output_path: Path,
) -> None:
	feature_names = np.array(artifacts["feature_names"], dtype=object)
	np.savez(
		output_path,
		w_xh=model.w_xh,
		w_hh=model.w_hh,
		b_h=model.b_h,
		w_hy=model.w_hy,
		b_y=np.array([model.b_y]),
		scaler_mean=artifacts["scaler_mean"],
		scaler_std=artifacts["scaler_std"],
		feature_names=feature_names,
	)


@app.command()
def main(
	raw_data_path: Path = RAW_DATA_DIR / "e-shop clothing 2008.csv",
	cluster_assignments_path: Path = CLUSTER_OUTPUTS_DIR / "cluster_assignments.csv",
	cluster_labels_path: Path = CLUSTER_OUTPUTS_DIR / "cluster_label.csv",
	model_output_path: Path = MODELS_DIR / "rnn_first_n_clicks_model.npz",
	n_clicks: int = 5,
	hidden_size: int = 32,
	learning_rate: float = 0.01,
	epochs: int = 30,
	test_size: float = 0.2,
	random_state: int = 42,
) -> None:
	"""Train a simple RNN on first-n-click trimmed sessions to predict purchase-intent probability."""

	logger.info(f"Loading raw clickstream from {raw_data_path}...")
	clickstream = pd.read_csv(raw_data_path, sep=";")

	dataset = prepare_rnn_training_data(
		clickstream=clickstream,
		n_clicks=n_clicks,
		cluster_assignments_path=cluster_assignments_path,
		cluster_labels_path=cluster_labels_path,
	)
	logger.info(
		"Prepared {} sessions with sequence length <= {} and {} features per click.",
		len(dataset.sequences),
		n_clicks,
		len(dataset.feature_names),
	)

	model, metrics, artifacts = train_rnn_model(
		dataset=dataset,
		hidden_size=hidden_size,
		learning_rate=learning_rate,
		epochs=epochs,
		test_size=test_size,
		random_state=random_state,
	)

	model_output_path.parent.mkdir(parents=True, exist_ok=True)
	save_rnn_artifacts(model=model, artifacts=artifacts, output_path=model_output_path)

	logger.success(f"Saved RNN artifacts to {model_output_path}")
	logger.info(
<<<<<<< HEAD
=======
<<<<<<< HEAD
		"Metrics: train_accuracy={:.4f}, train_roc_auc={:.4f}, train_precision={:.4f}, "
		"train_recall={:.4f}, train_f1={:.4f}, train_macro_f1={:.4f}, train_cohen_kappa={:.4f}, "
		"test_accuracy={:.4f}, test_roc_auc={:.4f}, test_precision={:.4f}, test_recall={:.4f}, "
		"test_f1={:.4f}, test_macro_f1={:.4f}, test_cohen_kappa={:.4f}",
		metrics["train_accuracy"],
		metrics["train_roc_auc"],
		metrics["train_precision"],
		metrics["train_recall"],
		metrics["train_f1"],
		metrics["train_macro_f1"],
		metrics["train_cohen_kappa"],
		metrics["test_accuracy"],
		metrics["test_roc_auc"],
		metrics["test_precision"],
		metrics["test_recall"],
		metrics["test_f1"],
		metrics["test_macro_f1"],
		metrics["test_cohen_kappa"],
=======
>>>>>>> origin/dev
		"Metrics: train_accuracy={:.4f}, test_accuracy={:.4f}, train_roc_auc={:.4f}, test_roc_auc={:.4f}",
		metrics["train_accuracy"],
		metrics["test_accuracy"],
		metrics["train_roc_auc"],
		metrics["test_roc_auc"],
<<<<<<< HEAD
=======
>>>>>>> origin
>>>>>>> origin/dev
	)


if __name__ == "__main__":
	app()
