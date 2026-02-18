"""Policy model for QA gating."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Policy:
    max_corrupt_pct: float = 1.0
    max_overexposed_pct: float = 5.0
    max_underexposed_pct: float = 5.0
    max_duplicate_pct: float = 10.0
    min_images_total: int = 100
