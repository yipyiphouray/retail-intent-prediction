"""SQLite persistence for demo sessions and clickstream rows."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any, Mapping

CLICK_COLUMNS = [
    "year",
    "month",
    "day",
    "country",
    "page_1_main_category",
    "page_2_clothing_model",
    "colour",
    "location",
    "model_photography",
    "price",
    "price_2",
    "page",
]

RAW_ROW_COLUMN_ORDER = [
    "year",
    "month",
    "day",
    "order",
    "country",
    "session ID",
    "page 1 (main category)",
    "page 2 (clothing model)",
    "colour",
    "location",
    "model photography",
    "price",
    "price 2",
    "page",
]


class SessionStore:
    """Persistence adapter for session state and click rows."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    predicted INTEGER NOT NULL DEFAULT 0,
                    prediction_label TEXT,
                    prediction_probability REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clicks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    order_idx INTEGER NOT NULL,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    day INTEGER NOT NULL,
                    country INTEGER NOT NULL,
                    page_1_main_category INTEGER NOT NULL,
                    page_2_clothing_model TEXT NOT NULL,
                    colour INTEGER NOT NULL,
                    location INTEGER NOT NULL,
                    model_photography INTEGER NOT NULL,
                    price REAL NOT NULL,
                    price_2 INTEGER NOT NULL,
                    page INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    UNIQUE(session_id, order_idx)
                )
                """
            )

    def create_session(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("INSERT INTO sessions DEFAULT VALUES")
            return int(cursor.lastrowid)

    def get_click_count(self, session_id: int) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) AS click_count FROM clicks WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            return int(row["click_count"]) if row else 0

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.predicted,
                    s.prediction_label,
                    s.prediction_probability,
                    (
                        SELECT COUNT(*)
                        FROM clicks c
                        WHERE c.session_id = s.session_id
                    ) AS click_count
                FROM sessions s
                WHERE s.session_id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "session_id": int(row["session_id"]),
            "predicted": bool(row["predicted"]),
            "prediction_label": row["prediction_label"],
            "prediction_probability": row["prediction_probability"],
            "click_count": int(row["click_count"]),
        }

    def add_click(self, session_id: int, click_payload: Mapping[str, Any]) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} does not exist.")

        order_idx = session["click_count"] + 1
        values = [click_payload[column] for column in CLICK_COLUMNS]

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO clicks (
                    session_id,
                    order_idx,
                    year,
                    month,
                    day,
                    country,
                    page_1_main_category,
                    page_2_clothing_model,
                    colour,
                    location,
                    model_photography,
                    price,
                    price_2,
                    page
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [session_id, order_idx, *values],
            )
            conn.execute(
                """
                UPDATE sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (session_id,),
            )

        return self._payload_to_raw_row(
            session_id=session_id, order_idx=order_idx, payload=click_payload
        )

    def list_clicks_raw_rows(self, session_id: int) -> list[dict[str, Any]]:
        if self.get_session(session_id) is None:
            raise KeyError(f"Session {session_id} does not exist.")

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    session_id,
                    order_idx,
                    year,
                    month,
                    day,
                    country,
                    page_1_main_category,
                    page_2_clothing_model,
                    colour,
                    location,
                    model_photography,
                    price,
                    price_2,
                    page
                FROM clicks
                WHERE session_id = ?
                ORDER BY order_idx ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()

        return [self._db_row_to_raw_row(row) for row in rows]

    def save_prediction(self, session_id: int, label: str, probability: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET
                    predicted = 1,
                    prediction_label = ?,
                    prediction_probability = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (label, probability, session_id),
            )

    @staticmethod
    def _payload_to_raw_row(
        session_id: int,
        order_idx: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        row = {
            "year": payload["year"],
            "month": payload["month"],
            "day": payload["day"],
            "order": order_idx,
            "country": payload["country"],
            "session ID": session_id,
            "page 1 (main category)": payload["page_1_main_category"],
            "page 2 (clothing model)": payload["page_2_clothing_model"],
            "colour": payload["colour"],
            "location": payload["location"],
            "model photography": payload["model_photography"],
            "price": payload["price"],
            "price 2": payload["price_2"],
            "page": payload["page"],
        }
        return {column: row[column] for column in RAW_ROW_COLUMN_ORDER}

    @staticmethod
    def _db_row_to_raw_row(row: sqlite3.Row) -> dict[str, Any]:
        data = {
            "year": row["year"],
            "month": row["month"],
            "day": row["day"],
            "order": row["order_idx"],
            "country": row["country"],
            "session ID": row["session_id"],
            "page 1 (main category)": row["page_1_main_category"],
            "page 2 (clothing model)": row["page_2_clothing_model"],
            "colour": row["colour"],
            "location": row["location"],
            "model photography": row["model_photography"],
            "price": row["price"],
            "price 2": row["price_2"],
            "page": row["page"],
        }
        return {column: data[column] for column in RAW_ROW_COLUMN_ORDER}
