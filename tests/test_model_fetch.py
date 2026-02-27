"""Tests for model artifact download and checksum validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from online_retail_prediction.modeling.fetch_model import ensure_model


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_ensure_model_skips_download_when_model_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "stacking_ensemble_model.pkl"
    model_path.write_bytes(b"already here")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "url": "https://example.com/model.pkl",
                "sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "online_retail_prediction.modeling.fetch_model._download_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("should not download")),
    )

    resolved = ensure_model(model_path=model_path, manifest_path=manifest_path)
    assert resolved == model_path


def test_ensure_model_downloads_and_verifies_checksum(tmp_path: Path) -> None:
    source_path = tmp_path / "source_model.pkl"
    source_bytes = b"model-bytes-for-download"
    source_path.write_bytes(source_bytes)

    model_path = tmp_path / "models" / "stacking_ensemble_model.pkl"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "url": source_path.as_uri(),
                "sha256": _sha256(source_bytes),
            }
        ),
        encoding="utf-8",
    )

    resolved = ensure_model(model_path=model_path, manifest_path=manifest_path)

    assert resolved == model_path
    assert model_path.read_bytes() == source_bytes


def test_ensure_model_raises_on_checksum_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "source_model.pkl"
    source_path.write_bytes(b"invalid-checksum-content")

    model_path = tmp_path / "models" / "stacking_ensemble_model.pkl"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "url": source_path.as_uri(),
                "sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ensure_model(model_path=model_path, manifest_path=manifest_path)
