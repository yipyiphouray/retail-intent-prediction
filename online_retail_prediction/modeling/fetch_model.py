"""Model artifact fetch utility for demo deployment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request

from loguru import logger
import typer

from online_retail_prediction.config import MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "stacking_ensemble_model.pkl"
DEFAULT_MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

app = typer.Typer()


@dataclass(frozen=True)
class ModelManifest:
    """Download metadata for the model artifact."""

    url: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as output:  # noqa: S310
        shutil.copyfileobj(response, output)


def load_manifest(manifest_path: Path) -> ModelManifest:
    if not manifest_path.exists():
        raise RuntimeError(
            f"Model manifest not found: {manifest_path}. "
            "Add models/model_manifest.json with release URL and sha256."
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in model manifest: {manifest_path}") from exc

    url = str(payload.get("url", "")).strip()
    sha256 = str(payload.get("sha256", "")).strip().lower()

    if not url:
        raise RuntimeError(
            "Model manifest is missing `url`. "
            "Set it to the GitHub Release asset URL for stacking_ensemble_model.pkl."
        )

    if not SHA256_PATTERN.fullmatch(sha256):
        raise RuntimeError(
            "Model manifest has invalid `sha256`. "
            "Provide a 64-character SHA256 checksum for the model asset."
        )

    return ModelManifest(url=url, sha256=sha256)


def ensure_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    force_download: bool = False,
) -> Path:
    """Ensure model exists locally, downloading from manifest when missing."""
    if model_path.exists() and not force_download:
        logger.info(f"Model already available at {model_path}; skipping download.")
        return model_path

    manifest = load_manifest(manifest_path)

    temporary_path = model_path.with_suffix(f"{model_path.suffix}.download")
    logger.info(f"Downloading model artifact from {manifest.url}")

    try:
        _download_file(manifest.url, temporary_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to download model from {manifest.url}. Check network access and manifest URL."
        ) from exc

    downloaded_sha = _sha256(temporary_path)
    if downloaded_sha != manifest.sha256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded model checksum mismatch. "
            f"Expected {manifest.sha256}, got {downloaded_sha}."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.replace(model_path)
    logger.success(f"Model saved to {model_path}")
    return model_path


@app.command()
def main(
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    force_download: bool = False,
) -> None:
    """Download the inference model artifact if it is missing locally."""
    try:
        ensure_model(
            model_path=model_path,
            manifest_path=manifest_path,
            force_download=force_download,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
