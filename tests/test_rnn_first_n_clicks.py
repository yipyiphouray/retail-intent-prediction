"""Tests for RNN data preparation using only first-N clicks per session."""

from pathlib import Path

import pandas as pd

<<<<<<< HEAD
from online_retail_prediction.modeling.RNN_train import prepare_rnn_training_data
=======
<<<<<<< HEAD
from online_retail_prediction.modeling.RNN_train import (
    cross_validate_rnn_model,
    prepare_rnn_training_data,
    train_rnn_model,
)
=======
from online_retail_prediction.modeling.RNN_train import prepare_rnn_training_data
>>>>>>> origin
>>>>>>> origin/dev


def test_prepare_rnn_data_trims_sequences_to_first_n_clicks(tmp_path: Path) -> None:
    clickstream = pd.DataFrame(
        {
            "session ID": [1, 1, 1, 2],
            "order": [1, 2, 3, 1],
            "price": [10, 20, 999, 30],
            "price 2": [2, 1, 1, 2],
            "page 2 (clothing model)": ["A1", "A2", "A3", "B1"],
            "main_category": ["trousers", "skirts", "sale", "blouses"],
            "colour": ["black", "red", "white", "blue"],
            "page": [1, 2, 3, 1],
            "location": [1, 2, 3, 4],
        }
    )

    cluster_assignments = pd.DataFrame(
        {
            "session_id": [1, 2],
            "cluster_id": [0, 1],
        }
    )
    cluster_labels = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "intent_label": ["low-intent", "high-intent"],
        }
    )
    cluster_assignments_path = tmp_path / "cluster_assignments.csv"
    cluster_labels_path = tmp_path / "cluster_label.csv"
    cluster_assignments.to_csv(cluster_assignments_path, index=False)
    cluster_labels.to_csv(cluster_labels_path, index=False)

    dataset = prepare_rnn_training_data(
        clickstream=clickstream,
        n_clicks=2,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
    )

    assert len(dataset.sequences) == 2
    assert all(sequence.shape[0] <= 2 for sequence in dataset.sequences)

    session_to_sequence = dict(zip(dataset.session_ids.tolist(), dataset.sequences, strict=False))
    price_index = dataset.feature_names.index("price")

    assert session_to_sequence[1][:, price_index].tolist() == [10.0, 20.0]
<<<<<<< HEAD
=======
<<<<<<< HEAD


def test_train_rnn_model_reports_expanded_metrics(tmp_path: Path) -> None:
    clickstream = pd.DataFrame(
        {
            "session ID": [1, 1, 2, 2, 3, 3, 4, 4],
            "order": [1, 2, 1, 2, 1, 2, 1, 2],
            "price": [10, 15, 12, 18, 25, 30, 28, 32],
            "price 2": [1, 1, 2, 2, 1, 1, 2, 2],
            "page 2 (clothing model)": ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"],
            "main_category": [
                "trousers",
                "skirts",
                "trousers",
                "skirts",
                "sale",
                "sale",
                "blouses",
                "blouses",
            ],
            "colour": ["black", "red", "black", "red", "white", "white", "blue", "blue"],
            "page": [1, 2, 1, 2, 1, 2, 1, 2],
            "location": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    cluster_assignments = pd.DataFrame(
        {
            "session_id": [1, 2, 3, 4],
            "cluster_id": [0, 0, 1, 1],
        }
    )
    cluster_labels = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "intent_label": ["low-intent", "high-intent"],
        }
    )
    cluster_assignments_path = tmp_path / "cluster_assignments.csv"
    cluster_labels_path = tmp_path / "cluster_label.csv"
    cluster_assignments.to_csv(cluster_assignments_path, index=False)
    cluster_labels.to_csv(cluster_labels_path, index=False)

    dataset = prepare_rnn_training_data(
        clickstream=clickstream,
        n_clicks=2,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
    )

    _, metrics, _ = train_rnn_model(
        dataset=dataset,
        hidden_size=4,
        learning_rate=0.01,
        epochs=1,
        test_size=0.5,
        random_state=42,
    )

    expected_metric_keys = {
        "train_roc_auc",
        "train_precision",
        "train_recall",
        "train_f1",
        "train_macro_f1",
        "train_cohen_kappa",
        "train_accuracy",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_macro_f1",
        "test_cohen_kappa",
        "test_accuracy",
    }

    assert expected_metric_keys.issubset(metrics.keys())


def test_cross_validate_rnn_model_reports_cv_mean_and_std_metrics(tmp_path: Path) -> None:
    clickstream = pd.DataFrame(
        {
            "session ID": [1, 1, 2, 2, 3, 3, 4, 4],
            "order": [1, 2, 1, 2, 1, 2, 1, 2],
            "price": [10, 15, 12, 18, 25, 30, 28, 32],
            "price 2": [1, 1, 2, 2, 1, 1, 2, 2],
            "page 2 (clothing model)": ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"],
            "main_category": [
                "trousers",
                "skirts",
                "trousers",
                "skirts",
                "sale",
                "sale",
                "blouses",
                "blouses",
            ],
            "colour": ["black", "red", "black", "red", "white", "white", "blue", "blue"],
            "page": [1, 2, 1, 2, 1, 2, 1, 2],
            "location": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    cluster_assignments = pd.DataFrame(
        {
            "session_id": [1, 2, 3, 4],
            "cluster_id": [0, 0, 1, 1],
        }
    )
    cluster_labels = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "intent_label": ["low-intent", "high-intent"],
        }
    )
    cluster_assignments_path = tmp_path / "cluster_assignments.csv"
    cluster_labels_path = tmp_path / "cluster_label.csv"
    cluster_assignments.to_csv(cluster_assignments_path, index=False)
    cluster_labels.to_csv(cluster_labels_path, index=False)

    dataset = prepare_rnn_training_data(
        clickstream=clickstream,
        n_clicks=2,
        cluster_assignments_path=cluster_assignments_path,
        cluster_labels_path=cluster_labels_path,
    )

    fold_results, summary_metrics = cross_validate_rnn_model(
        dataset=dataset,
        hidden_size=4,
        learning_rate=0.01,
        epochs=1,
        n_splits=2,
        random_state=42,
    )

    assert len(fold_results) == 2
    assert "fold" in fold_results.columns

    expected_cv_summary_keys = {
        "cv_mean_test_roc_auc",
        "cv_std_test_roc_auc",
        "cv_mean_test_precision",
        "cv_std_test_precision",
        "cv_mean_test_recall",
        "cv_std_test_recall",
        "cv_mean_test_f1",
        "cv_std_test_f1",
        "cv_mean_test_macro_f1",
        "cv_std_test_macro_f1",
        "cv_mean_test_cohen_kappa",
        "cv_std_test_cohen_kappa",
        "cv_mean_test_accuracy",
        "cv_std_test_accuracy",
    }

    assert expected_cv_summary_keys.issubset(summary_metrics.keys())
=======
>>>>>>> origin
>>>>>>> origin/dev
