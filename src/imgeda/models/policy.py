"""Policy model for QA gating."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Policy:
    max_corrupt_pct: float = 1.0
    max_overexposed_pct: float = 5.0
    max_underexposed_pct: float = 5.0
    max_duplicate_pct: float = 10.0
    max_blurry_pct: float = 100.0  # disabled by default
    max_artifact_pct: float = 100.0  # disabled by default
    min_images_total: int = 100
    min_width: int = 0
    min_height: int = 0
    allowed_formats: list[str] = field(default_factory=list)
    max_aspect_ratio: float = 0.0  # 0 = disabled
