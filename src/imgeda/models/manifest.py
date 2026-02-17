"""Data models for image analysis records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PixelStats:
    mean_r: float = 0.0
    mean_g: float = 0.0
    mean_b: float = 0.0
    std_r: float = 0.0
    std_g: float = 0.0
    std_b: float = 0.0
    mean_brightness: float = 0.0
    min_val: int = 0
    max_val: int = 255


@dataclass(slots=True)
class CornerStats:
    corner_mean: float = 0.0
    center_mean: float = 0.0
    border_mean: float = 0.0
    delta: float = 0.0


@dataclass(slots=True)
class ImageRecord:
    # Identity
    path: str = ""
    filename: str = ""
    file_size_bytes: int = 0
    mtime: float = 0.0

    # Metadata
    width: int = 0
    height: int = 0
    format: str = ""
    color_mode: str = ""
    num_channels: int = 0
    aspect_ratio: float = 0.0

    # Pixel stats (optional)
    pixel_stats: PixelStats | None = None

    # Corner stats (optional)
    corner_stats: CornerStats | None = None

    # Hashes (optional)
    phash: str | None = None
    dhash: str | None = None

    # Flags
    is_corrupt: bool = False
    is_dark: bool = False
    is_overexposed: bool = False
    has_border_artifact: bool = False

    # Meta
    analyzed_at: str = ""
    analyzer_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageRecord:
        ps = data.get("pixel_stats")
        cs = data.get("corner_stats")
        rec = cls(
            **{
                k: v
                for k, v in data.items()
                if k in cls.__dataclass_fields__ and k not in ("pixel_stats", "corner_stats")
            }
        )
        if ps and isinstance(ps, dict):
            rec.pixel_stats = PixelStats(**ps)
        if cs and isinstance(cs, dict):
            rec.corner_stats = CornerStats(**cs)
        return rec


MANIFEST_META_KEY = "__manifest_meta__"


@dataclass(slots=True)
class ManifestMeta:
    is_meta: bool = field(default=True, repr=False)
    input_dir: str = ""
    schema_version: int = 1
    settings: dict[str, Any] = field(default_factory=dict)
    total_files: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("is_meta", None)
        d[MANIFEST_META_KEY] = True
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestMeta:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)
