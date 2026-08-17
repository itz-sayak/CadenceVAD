"""Paths to CadenceVAD's bundled inference artifacts."""

from __future__ import annotations

from pathlib import Path

from .artifact_validation import validate_bundled_artifacts


def _bundled_model_dir() -> Path:
    packaged = Path(__file__).resolve().parent / "_models"
    if packaged.is_dir():
        return packaged
    repository = Path(__file__).resolve().parents[2] / "models" / "cadencevad-v0.1"
    if repository.is_dir():
        return repository
    raise RuntimeError("CadenceVAD's bundled ONNX model is missing from this installation")


def bundled_model_path() -> Path:
    """Return the self-contained streaming ONNX model shipped with CadenceVAD."""

    path = _bundled_model_dir() / "cadencevad-stream.onnx"
    if not path.is_file():
        raise RuntimeError("CadenceVAD's bundled ONNX model is missing from this installation")
    return validate_bundled_artifacts(path)


__all__ = ["bundled_model_path"]
