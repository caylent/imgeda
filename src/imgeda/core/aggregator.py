"""Dataset-level stat aggregation from manifest records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from imgeda.models.manifest import ImageRecord


@dataclass
class DatasetSummary:
    total_images: int = 0
    total_size_bytes: int = 0
    corrupt_count: int = 0
    dark_count: int = 0
    overexposed_count: int = 0
    artifact_count: int = 0
    blurry_count: int = 0

    min_width: int = 0
    max_width: int = 0
    min_height: int = 0
    max_height: int = 0

    format_counts: dict[str, int] = field(default_factory=dict)
    mode_counts: dict[str, int] = field(default_factory=dict)
    extension_counts: dict[str, int] = field(default_factory=dict)

    # EXIF summary
    camera_make_counts: dict[str, int] = field(default_factory=dict)
    camera_model_counts: dict[str, int] = field(default_factory=dict)
    high_distortion_risk_count: int = 0
    gps_flagged_count: int = 0
    rotated_count: int = 0
    exif_present_count: int = 0


def aggregate(records: list[ImageRecord]) -> DatasetSummary:
    """Compute dataset-level summary statistics."""
    if not records:
        return DatasetSummary()

    valid = [r for r in records if not r.is_corrupt]

    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    camera_makes: Counter[str] = Counter()
    camera_models: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []

    exif_present = 0
    high_distortion = 0
    gps_flagged = 0
    rotated = 0

    for r in records:
        if r.format:
            formats[r.format] += 1
        if r.color_mode:
            modes[r.color_mode] += 1
        ext = r.filename.rsplit(".", 1)[-1].lower() if "." in r.filename else ""
        if ext:
            extensions[ext] += 1
        if r.camera_make:
            camera_makes[r.camera_make] += 1
            exif_present += 1
        if r.camera_model:
            camera_models[r.camera_model] += 1
        if r.distortion_risk == "high":
            high_distortion += 1
        if r.has_gps_data:
            gps_flagged += 1
        if r.orientation_tag is not None and r.orientation_tag != 1:
            rotated += 1

    for r in valid:
        widths.append(r.width)
        heights.append(r.height)

    return DatasetSummary(
        total_images=len(records),
        total_size_bytes=sum(r.file_size_bytes for r in records),
        corrupt_count=sum(1 for r in records if r.is_corrupt),
        dark_count=sum(1 for r in records if r.is_dark),
        overexposed_count=sum(1 for r in records if r.is_overexposed),
        artifact_count=sum(1 for r in records if r.has_border_artifact),
        blurry_count=sum(1 for r in records if r.is_blurry),
        min_width=min(widths) if widths else 0,
        max_width=max(widths) if widths else 0,
        min_height=min(heights) if heights else 0,
        max_height=max(heights) if heights else 0,
        format_counts=dict(formats.most_common()),
        mode_counts=dict(modes.most_common()),
        extension_counts=dict(extensions.most_common()),
        camera_make_counts=dict(camera_makes.most_common()),
        camera_model_counts=dict(camera_models.most_common()),
        high_distortion_risk_count=high_distortion,
        gps_flagged_count=gps_flagged,
        rotated_count=rotated,
        exif_present_count=exif_present,
    )
