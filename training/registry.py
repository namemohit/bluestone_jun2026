"""Minimal model registry — versioned params + metrics, with an active pointer (promote/rollback)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelVersion:
    version: int
    params: Any
    metrics: dict


class ModelRegistry:
    def __init__(self):
        self.versions: list[ModelVersion] = []
        self.active_version: Optional[int] = None

    def register(self, params, metrics) -> ModelVersion:
        mv = ModelVersion(version=len(self.versions) + 1, params=params, metrics=metrics)
        self.versions.append(mv)
        return mv

    def promote(self, version: int) -> None:
        self.active_version = version

    def active(self) -> Optional[ModelVersion]:
        for mv in self.versions:
            if mv.version == self.active_version:
                return mv
        return None
