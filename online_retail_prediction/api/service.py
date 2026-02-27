"""Business logic for collecting clicks and triggering session inference."""

from __future__ import annotations

from typing import Any

from online_retail_prediction.api.session_store import SessionStore
from online_retail_prediction.modeling.predict import InferencePipeline


class InferenceService:
    """Coordinates session storage and model inference trigger behavior."""

    def __init__(
        self,
        session_store: SessionStore,
        pipeline: InferencePipeline,
        trigger_click_count: int = 5,
    ) -> None:
        self.session_store = session_store
        self.pipeline = pipeline
        self.trigger_click_count = trigger_click_count

    def create_session(self) -> dict[str, Any]:
        session_id = self.session_store.create_session()
        return self.get_session(session_id)

    def get_session(self, session_id: int) -> dict[str, Any]:
        session = self.session_store.get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} does not exist.")

        prediction = self._prediction_from_session_state(session)
        status = "predicted" if session["predicted"] else "collecting"

        return {
            "session_id": session["session_id"],
            "click_count": session["click_count"],
            "status": status,
            "prediction": prediction,
        }

    def list_clicks(self, session_id: int) -> list[dict[str, Any]]:
        return self.session_store.list_clicks_raw_rows(session_id)

    def record_click(self, session_id: int, click_payload: dict[str, Any]) -> dict[str, Any]:
        session_before_click = self.session_store.get_session(session_id)
        if session_before_click is None:
            raise KeyError(f"Session {session_id} does not exist.")

        raw_row = self.session_store.add_click(session_id=session_id, click_payload=click_payload)
        click_count = self.session_store.get_click_count(session_id)

        triggered = False
        prediction = self._prediction_from_session_state(session_before_click)

        if click_count == self.trigger_click_count and not session_before_click["predicted"]:
            rows = self.session_store.list_clicks_raw_rows(session_id)
            result = self.pipeline.predict_rows(rows)
            self.session_store.save_prediction(
                session_id=session_id,
                label=result["label"],
                probability=result["probability"],
            )
            prediction = {
                "label": result["label"],
                "probability": float(result["probability"]),
            }
            triggered = True
        elif session_before_click["predicted"]:
            prediction = self._prediction_from_session_state(session_before_click)

        show_ad = bool(prediction and prediction["label"] == "high-intent")

        return {
            "session_id": session_id,
            "click_count": click_count,
            "triggered": triggered,
            "prediction": prediction,
            "show_ad": show_ad,
            "raw_row": raw_row,
        }

    @staticmethod
    def _prediction_from_session_state(session_state: dict[str, Any]) -> dict[str, Any] | None:
        label = session_state.get("prediction_label")
        probability = session_state.get("prediction_probability")
        if label is None or probability is None:
            return None
        return {
            "label": str(label),
            "probability": float(probability),
        }
