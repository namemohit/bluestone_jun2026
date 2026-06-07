"""Human corrections + an append-only, round-versioned label store.

GUARDRAIL: only human-verified corrections enter the store. Models are never retrained on
their own unverified predictions (that causes feedback bias / collapse).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class Correction:
    item_id: str
    field: str  # "is_employee" | "demographics" | "group"
    before: Any
    after: Any
    embedding: Optional[np.ndarray] = None
    employee_id: Optional[int] = None
    subject: Optional[int] = None  # identity/global-id the correction applies to
    reviewer: str = "human"


class LabelStore:
    """Accumulated human-verified corrections, grouped by review round (for dataset versioning)."""

    def __init__(self):
        self.rounds: list[list[Correction]] = []

    def add_round(self, corrections) -> None:
        self.rounds.append(list(corrections))

    def all(self) -> list[Correction]:
        return [c for r in self.rounds for c in r]

    def for_field(self, field: str) -> list[Correction]:
        return [c for c in self.all() if c.field == field]

    def __len__(self) -> int:
        return len(self.all())
