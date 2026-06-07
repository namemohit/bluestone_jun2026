"""Config loading with ${ENV_VAR} expansion.

Secrets (NVR password, DB DSN) should live in environment variables and be referenced
in the YAML as ${VAR}. This keeps credentials out of the committed config.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    """Load a YAML config and expand ${ENV_VAR} placeholders."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Copy configs/cameras.example.yaml to "
            f"configs/cameras.yaml and edit it."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _expand(raw)


def get_camera(config: dict, camera_id: str) -> dict:
    """Return the camera block for a given id, or raise."""
    for cam in config.get("cameras", []):
        if cam.get("id") == camera_id:
            return cam
    ids = [c.get("id") for c in config.get("cameras", [])]
    raise KeyError(f"Camera '{camera_id}' not in config. Available: {ids}")
