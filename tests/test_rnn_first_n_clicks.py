"""Tests for RNN data preparation using only first-N clicks per session."""

import pandas as pd

from online_retail_prediction.modeling.RNN_train import prepare_rnn_training_data


def test_prepare_rnn_data_trims_sequences_to_first_n_clicks() -> None:
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

    dataset = prepare_rnn_training_data(
        clickstream=clickstream,
        n_clicks=2,
        min_session_clicks=1,
        min_high_price_share=0.0,
    )

    assert len(dataset.sequences) == 2
    assert all(sequence.shape[0] <= 2 for sequence in dataset.sequences)

    session_to_sequence = dict(zip(dataset.session_ids.tolist(), dataset.sequences, strict=False))
    price_index = dataset.feature_names.index("price")

    assert session_to_sequence[1][:, price_index].tolist() == [10.0, 20.0]
