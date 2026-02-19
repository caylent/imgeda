"""Configuration models with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class ScanConfig:
    workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    checkpoint_every: int = 500
    include_hashes: bool = True
    hash_size: int = 16
    skip_pixel_stats: bool = False
    skip_exif: bool = False
    extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",
    )
    dark_threshold: float = 40.0
    overexposed_threshold: float = 220.0
    artifact_threshold: float = 50.0
    corner_patch_fraction: float = 0.1
    max_image_dimension: int = 2048
    duplicate_hamming_threshold: int = 8
    resume: bool = True
    force: bool = False


@dataclass(slots=True)
class PlotConfig:
    output_dir: str = "."
    format: str = "png"
    dpi: int = 150
    sample: int | None = None
    figsize: tuple[float, float] = (9.0, 6.0)
    artifact_threshold: float = 50.0
    seed: int = 42
