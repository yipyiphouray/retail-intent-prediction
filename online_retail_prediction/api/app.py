"""FastAPI app exposing demo session endpoints for clickstream inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from online_retail_prediction.api.schemas import (
    ClickEventOut,
    ClickIn,
    HealthOut,
    SessionClicksOut,
    SessionStateOut,
)
from online_retail_prediction.api.service import InferenceService
from online_retail_prediction.api.session_store import SessionStore
from online_retail_prediction.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from online_retail_prediction.modeling.fetch_model import (
    DEFAULT_MANIFEST_PATH,
    ensure_model,
)
from online_retail_prediction.modeling.predict import InferencePipeline

DEFAULT_DB_PATH = DATA_DIR / "interim" / "demo_sessions.db"
DEFAULT_MODEL_PATH = MODELS_DIR / "stacking_ensemble_model.pkl"
DEFAULT_FREQ_PATH = PROCESSED_DATA_DIR / "training_frequencies.json"


def _resolve_path(env_var: str, default: Path) -> Path:
    raw = os.getenv(env_var)
    return Path(raw) if raw else default


def build_default_service() -> tuple[InferenceService, Path, Path]:
    db_path = _resolve_path("DEMO_DB_PATH", DEFAULT_DB_PATH)
    model_path = _resolve_path("DEMO_MODEL_PATH", DEFAULT_MODEL_PATH)
    freq_path = _resolve_path("DEMO_FREQ_PATH", DEFAULT_FREQ_PATH)
    manifest_path = _resolve_path("DEMO_MODEL_MANIFEST_PATH", DEFAULT_MANIFEST_PATH)

    ensure_model(model_path=model_path, manifest_path=manifest_path, force_download=False)

    pipeline = InferencePipeline(model_path=model_path, freq_path=freq_path, n_clicks=5)
    store = SessionStore(db_path=db_path)
    store.initialize()

    return (
        InferenceService(session_store=store, pipeline=pipeline, trigger_click_count=5),
        db_path,
        model_path,
    )


def create_app(
    service: InferenceService | None = None,
    db_path: Path | None = None,
    model_path: Path | None = None,
) -> FastAPI:
    def _initialize_state(application: FastAPI) -> None:
        if application.state.service is not None:
            if application.state.db_path is None:
                application.state.db_path = application.state.service.session_store.db_path
            if application.state.model_path is None:
                application.state.model_path = _resolve_path("DEMO_MODEL_PATH", DEFAULT_MODEL_PATH)
            return

        try:
            built_service, built_db_path, built_model_path = build_default_service()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to initialize inference service. "
                "Ensure model artifact download is configured via models/model_manifest.json "
                "or place models/stacking_ensemble_model.pkl locally."
            ) from exc

        application.state.service = built_service
        application.state.db_path = built_db_path
        application.state.model_path = built_model_path

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        _initialize_state(application)
        yield

    app = FastAPI(title="Retail Intent Demo API", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.service = service
    app.state.db_path = db_path
    app.state.model_path = model_path

    def _service() -> InferenceService:
        service_obj = app.state.service
        if service_obj is None:
            raise RuntimeError("Inference service has not been initialized.")
        return service_obj

    @app.get("/health", response_model=HealthOut)
    def health() -> HealthOut:
        model_path_str = str(app.state.model_path) if app.state.model_path is not None else ""
        db_path_str = str(app.state.db_path) if app.state.db_path is not None else ""
        return HealthOut(
            status="ok",
            model_ready=bool(model_path_str and Path(model_path_str).exists()),
            model_path=model_path_str,
            db_path=db_path_str,
        )

    @app.post("/api/v1/sessions", response_model=SessionStateOut)
    def create_session() -> SessionStateOut:
        return SessionStateOut.model_validate(_service().create_session())

    @app.get("/api/v1/sessions/{session_id}", response_model=SessionStateOut)
    def get_session(session_id: int) -> SessionStateOut:
        try:
            state = _service().get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SessionStateOut.model_validate(state)

    @app.get("/api/v1/sessions/{session_id}/clicks", response_model=SessionClicksOut)
    def get_session_clicks(session_id: int) -> SessionClicksOut:
        try:
            rows = _service().list_clicks(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return SessionClicksOut(session_id=session_id, clicks=rows)

    @app.post("/api/v1/sessions/{session_id}/clicks", response_model=ClickEventOut)
    def post_click(session_id: int, click: ClickIn) -> ClickEventOut:
        try:
            event = _service().record_click(
                session_id=session_id, click_payload=click.model_dump()
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ClickEventOut.model_validate(event)

    return app


app = create_app()
