"""Tests for demo inference pipeline and FastAPI session endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from online_retail_prediction.api.app import create_app
from online_retail_prediction.api.service import InferenceService
from online_retail_prediction.api.session_store import RAW_ROW_COLUMN_ORDER, SessionStore
from online_retail_prediction.modeling.predict import InferencePipeline


class _FakeModel:
    """Simple deterministic model used to test inference plumbing."""

    feature_names_in_ = np.array(
        [
            "n_clicks_observed",
            "mean_model_frequency",
            "max_model_frequency",
            "min_model_frequency",
            "first_model_frequency",
            "last_model_frequency",
            "last_category_frequency",
            "last_colour_frequency",
        ],
        dtype=object,
    )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        score = np.clip(features["n_clicks_observed"].to_numpy(dtype=float) / 5.0, 0.0, 1.0)
        return np.column_stack([1.0 - score, score])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


class _FakePipeline:
    """Service-level fake pipeline for API tests."""

    def __init__(self) -> None:
        self.call_count = 0

    def predict_rows(self, rows: list[dict]) -> dict:
        self.call_count += 1
        return {
            "label": "high-intent",
            "probability": 0.8123,
            "features": {"n_clicks_observed": len(rows)},
        }


@pytest.fixture()
def sample_click_rows() -> pd.DataFrame:
    """Build five click rows matching the raw dataset schema."""

    return pd.DataFrame(
        {
            "year": [2008, 2008, 2008, 2008, 2008],
            "month": [4, 4, 4, 4, 4],
            "day": [1, 1, 1, 1, 1],
            "order": [1, 2, 3, 4, 5],
            "country": [29, 29, 29, 29, 29],
            "session ID": [1001, 1001, 1001, 1001, 1001],
            "page 1 (main category)": [1, 1, 2, 2, 3],
            "page 2 (clothing model)": ["A13", "A16", "B4", "B17", "C1"],
            "colour": [1, 1, 10, 6, 14],
            "location": [5, 6, 2, 6, 3],
            "model photography": [1, 1, 1, 2, 2],
            "price": [28.0, 33.0, 52.0, 38.0, 65.0],
            "price 2": [2, 2, 1, 2, 1],
            "page": [1, 1, 1, 2, 2],
        }
    )


@pytest.fixture()
def pipeline_with_fakes(monkeypatch: pytest.MonkeyPatch) -> InferencePipeline:
    """Inference pipeline with fake model and frequencies."""

    monkeypatch.setattr(
        "online_retail_prediction.modeling.predict.load_model",
        lambda _path: _FakeModel(),
    )
    monkeypatch.setattr(
        "online_retail_prediction.modeling.predict._load_frequencies",
        lambda _path: {
            "model_freq": {"A13": 0.2, "A16": 0.1, "B4": 0.3, "B17": 0.15, "C1": 0.25},
            "category_freq": {"trousers": 0.5, "skirts": 0.3, "blouses": 0.2},
            "colour_freq": {"1": 0.2, "10": 0.2, "6": 0.2, "14": 0.2},
        },
    )
    return InferencePipeline(
        model_path=Path("unused.pkl"), freq_path=Path("unused.json"), n_clicks=5
    )


def test_inference_pipeline_predicts_single_session(
    sample_click_rows: pd.DataFrame,
    pipeline_with_fakes: InferencePipeline,
) -> None:
    result = pipeline_with_fakes.predict_session(sample_click_rows)

    assert result["label"] == "high-intent"
    assert result["probability"] == 1.0
    assert "n_clicks_observed" in result["features"]


def test_inference_pipeline_rejects_multi_session_input(
    sample_click_rows: pd.DataFrame,
    pipeline_with_fakes: InferencePipeline,
) -> None:
    multi_session = sample_click_rows.copy()
    multi_session.loc[multi_session.index[-1], "session ID"] = 2002

    with pytest.raises(ValueError, match="exactly one session_id"):
        pipeline_with_fakes.predict_session(multi_session)


def _sample_click_payload(index: int) -> dict:
    model_codes = ["A13", "A16", "B4", "B17", "C1", "D2"]
    category_codes = [1, 1, 2, 2, 3, 4]
    colour_codes = [1, 1, 10, 6, 14, 2]
    page_codes = [1, 1, 1, 2, 2, 3]

    return {
        "year": 2008,
        "month": 4,
        "day": 1,
        "country": 29,
        "page_1_main_category": category_codes[index - 1],
        "page_2_clothing_model": model_codes[index - 1],
        "colour": colour_codes[index - 1],
        "location": (index % 6) + 1,
        "model_photography": 1 if index % 2 == 0 else 2,
        "price": float(20 + index * 5),
        "price_2": 1 if index >= 3 else 2,
        "page": page_codes[index - 1],
    }


def test_api_triggers_prediction_on_fifth_click_and_only_once(tmp_path: Path) -> None:
    db_path = tmp_path / "demo_sessions.db"
    store = SessionStore(db_path=db_path)
    store.initialize()

    fake_pipeline = _FakePipeline()
    service = InferenceService(session_store=store, pipeline=fake_pipeline, trigger_click_count=5)
    app = create_app(service=service, db_path=db_path, model_path=tmp_path / "model.pkl")

    with TestClient(app) as client:
        session_response = client.post("/api/v1/sessions")
        assert session_response.status_code == 200

        session = session_response.json()
        session_id = session["session_id"]
        assert session["click_count"] == 0
        assert session["status"] == "collecting"
        assert session["prediction"] is None

        for i in range(1, 5):
            response = client.post(
                f"/api/v1/sessions/{session_id}/clicks",
                json=_sample_click_payload(i),
            )
            assert response.status_code == 200
            body = response.json()
            assert body["click_count"] == i
            assert body["triggered"] is False
            assert body["prediction"] is None
            assert body["show_ad"] is False

        fifth = client.post(
            f"/api/v1/sessions/{session_id}/clicks",
            json=_sample_click_payload(5),
        )
        assert fifth.status_code == 200

        fifth_body = fifth.json()
        assert fifth_body["click_count"] == 5
        assert fifth_body["triggered"] is True
        assert fifth_body["prediction"] == {"label": "high-intent", "probability": 0.8123}
        assert fifth_body["show_ad"] is True
        assert list(fifth_body["raw_row"].keys()) == RAW_ROW_COLUMN_ORDER

        sixth = client.post(
            f"/api/v1/sessions/{session_id}/clicks",
            json=_sample_click_payload(6),
        )
        assert sixth.status_code == 200

        sixth_body = sixth.json()
        assert sixth_body["click_count"] == 6
        assert sixth_body["triggered"] is False
        assert sixth_body["prediction"] == {"label": "high-intent", "probability": 0.8123}
        assert fake_pipeline.call_count == 1

        clicks_response = client.get(f"/api/v1/sessions/{session_id}/clicks")
        assert clicks_response.status_code == 200
        clicks_payload = clicks_response.json()
        assert len(clicks_payload["clicks"]) == 6
        assert list(clicks_payload["clicks"][0].keys()) == RAW_ROW_COLUMN_ORDER


def test_click_storage_persists_across_service_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "demo_sessions.db"
    initial_store = SessionStore(db_path=db_path)
    initial_store.initialize()

    first_pipeline = _FakePipeline()
    first_service = InferenceService(
        session_store=initial_store,
        pipeline=first_pipeline,
        trigger_click_count=5,
    )
    first_app = create_app(
        service=first_service, db_path=db_path, model_path=tmp_path / "model.pkl"
    )

    with TestClient(first_app) as first_client:
        session_id = first_client.post("/api/v1/sessions").json()["session_id"]
        for i in range(1, 7):
            first_client.post(
                f"/api/v1/sessions/{session_id}/clicks", json=_sample_click_payload(i)
            )

    restarted_store = SessionStore(db_path=db_path)
    restarted_store.initialize()

    second_pipeline = _FakePipeline()
    second_service = InferenceService(
        session_store=restarted_store,
        pipeline=second_pipeline,
        trigger_click_count=5,
    )
    second_app = create_app(
        service=second_service, db_path=db_path, model_path=tmp_path / "model.pkl"
    )

    with TestClient(second_app) as second_client:
        session_response = second_client.get(f"/api/v1/sessions/{session_id}")
        assert session_response.status_code == 200
        state = session_response.json()

        assert state["click_count"] == 6
        assert state["status"] == "predicted"
        assert state["prediction"] == {"label": "high-intent", "probability": 0.8123}
