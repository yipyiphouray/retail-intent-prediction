"""Pydantic schemas for the demo inference API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ClickIn(BaseModel):
    """Input click payload from the demo storefront."""

    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    country: int = Field(ge=1)
    page_1_main_category: int = Field(ge=1, le=4)
    page_2_clothing_model: str = Field(min_length=1)
    colour: int = Field(ge=1, le=14)
    location: int = Field(ge=1, le=6)
    model_photography: int = Field(ge=1, le=2)
    price: float = Field(gt=0)
    price_2: int = Field(ge=1, le=2)
    page: int = Field(ge=1, le=5)


class PredictionOut(BaseModel):
    """Model prediction payload."""

    label: Literal["low-intent", "high-intent"]
    probability: float


class SessionStateOut(BaseModel):
    """Current session status and prediction state."""

    session_id: int
    click_count: int
    status: Literal["collecting", "predicted"]
    prediction: PredictionOut | None


class ClickEventOut(BaseModel):
    """Response payload for a click event insertion."""

    session_id: int
    click_count: int
    triggered: bool
    prediction: PredictionOut | None
    show_ad: bool
    raw_row: dict[str, Any]


class SessionClicksOut(BaseModel):
    """All click rows captured for a given session."""

    session_id: int
    clicks: list[dict[str, Any]]


class HealthOut(BaseModel):
    """API health and dependency readiness."""

    status: Literal["ok"]
    model_ready: bool
    model_path: str
    db_path: str
