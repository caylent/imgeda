"""CSV export for manifest records."""

from __future__ import annotations

import csv

from imgeda.models.manifest import ImageRecord


def _flatten_record(rec: ImageRecord) -> dict[str, object]:
    """Flatten an ImageRecord into a flat dict for CSV."""
    d: dict[str, object] = {
        "path": rec.path,
        "filename": rec.filename,
        "file_size_bytes": rec.file_size_bytes,
        "mtime": rec.mtime,
        "width": rec.width,
        "height": rec.height,
        "format": rec.format,
        "color_mode": rec.color_mode,
        "num_channels": rec.num_channels,
        "aspect_ratio": rec.aspect_ratio,
        "camera_make": rec.camera_make or "",
        "camera_model": rec.camera_model or "",
        "lens_model": rec.lens_model or "",
        "focal_length_mm": rec.focal_length_mm if rec.focal_length_mm is not None else "",
        "focal_length_35mm": rec.focal_length_35mm if rec.focal_length_35mm is not None else "",
        "iso_speed": rec.iso_speed if rec.iso_speed is not None else "",
        "f_number": rec.f_number if rec.f_number is not None else "",
        "exposure_time_sec": rec.exposure_time_sec if rec.exposure_time_sec is not None else "",
        "datetime_original": rec.datetime_original or "",
        "has_gps_data": rec.has_gps_data,
        "distortion_risk": rec.distortion_risk or "",
        "blur_score": rec.blur_score if rec.blur_score is not None else "",
        "is_blurry": rec.is_blurry,
        "phash": rec.phash or "",
        "dhash": rec.dhash or "",
        "is_corrupt": rec.is_corrupt,
        "is_dark": rec.is_dark,
        "is_overexposed": rec.is_overexposed,
        "has_border_artifact": rec.has_border_artifact,
        "analyzed_at": rec.analyzed_at,
    }

    ps = rec.pixel_stats
    d["mean_r"] = ps.mean_r if ps else ""
    d["mean_g"] = ps.mean_g if ps else ""
    d["mean_b"] = ps.mean_b if ps else ""
    d["std_r"] = ps.std_r if ps else ""
    d["std_g"] = ps.std_g if ps else ""
    d["std_b"] = ps.std_b if ps else ""
    d["mean_brightness"] = ps.mean_brightness if ps else ""

    cs = rec.corner_stats
    d["corner_mean"] = cs.corner_mean if cs else ""
    d["center_mean"] = cs.center_mean if cs else ""
    d["border_mean"] = cs.border_mean if cs else ""
    d["corner_delta"] = cs.delta if cs else ""

    return d


def records_to_csv(records: list[ImageRecord], output_path: str) -> int:
    """Write ImageRecords to a CSV file. Returns row count."""
    if not records:
        return 0

    rows = [_flatten_record(r) for r in records]
    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)
