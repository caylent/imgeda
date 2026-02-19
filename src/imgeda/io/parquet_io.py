"""Streaming JSONL to Parquet conversion."""

from __future__ import annotations

from imgeda.models.manifest import ImageRecord


def _flatten_record(rec: ImageRecord) -> dict[str, object]:
    """Flatten an ImageRecord into a dict with nested fields expanded."""
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
        "focal_length_mm": rec.focal_length_mm if rec.focal_length_mm is not None else 0.0,
        "focal_length_35mm": rec.focal_length_35mm if rec.focal_length_35mm is not None else 0,
        "iso_speed": rec.iso_speed if rec.iso_speed is not None else 0,
        "f_number": rec.f_number if rec.f_number is not None else 0.0,
        "exposure_time_sec": (rec.exposure_time_sec if rec.exposure_time_sec is not None else 0.0),
        "datetime_original": rec.datetime_original or "",
        "orientation_tag": rec.orientation_tag if rec.orientation_tag is not None else 0,
        "has_gps_data": rec.has_gps_data,
        "distortion_risk": rec.distortion_risk or "",
        "blur_score": rec.blur_score if rec.blur_score is not None else 0.0,
        "is_blurry": rec.is_blurry,
        "phash": rec.phash or "",
        "dhash": rec.dhash or "",
        "is_corrupt": rec.is_corrupt,
        "is_dark": rec.is_dark,
        "is_overexposed": rec.is_overexposed,
        "has_border_artifact": rec.has_border_artifact,
        "analyzed_at": rec.analyzed_at,
    }

    # Flatten pixel_stats
    ps = rec.pixel_stats
    d["pixel_stats.mean_r"] = ps.mean_r if ps else 0.0
    d["pixel_stats.mean_g"] = ps.mean_g if ps else 0.0
    d["pixel_stats.mean_b"] = ps.mean_b if ps else 0.0
    d["pixel_stats.std_r"] = ps.std_r if ps else 0.0
    d["pixel_stats.std_g"] = ps.std_g if ps else 0.0
    d["pixel_stats.std_b"] = ps.std_b if ps else 0.0
    d["pixel_stats.mean_brightness"] = ps.mean_brightness if ps else 0.0
    d["pixel_stats.min_val"] = ps.min_val if ps else 0
    d["pixel_stats.max_val"] = ps.max_val if ps else 255

    # Flatten corner_stats
    cs = rec.corner_stats
    d["corner_stats.corner_mean"] = cs.corner_mean if cs else 0.0
    d["corner_stats.center_mean"] = cs.center_mean if cs else 0.0
    d["corner_stats.border_mean"] = cs.border_mean if cs else 0.0
    d["corner_stats.delta"] = cs.delta if cs else 0.0

    return d


def records_to_parquet(records: list[ImageRecord], output_path: str) -> int:
    """Convert ImageRecords to a Parquet file. Returns row count.

    Requires pyarrow: install with `pip install imgeda[parquet]`.
    """
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as e:
        msg = "pyarrow is required for Parquet export. Install with: pip install imgeda[parquet]"
        raise ImportError(msg) from e

    if not records:
        # Write empty parquet with schema
        schema = pa.schema(
            [
                ("path", pa.string()),
                ("filename", pa.string()),
                ("file_size_bytes", pa.int64()),
            ]
        )
        table = pa.table({f.name: [] for f in schema}, schema=schema)
        pq.write_table(table, output_path)
        return 0

    # Process in chunks to bound memory
    chunk_size = 10000
    writer = None
    total_rows = 0

    try:
        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            rows = [_flatten_record(r) for r in chunk]

            # Build columnar data
            columns: dict[str, list[object]] = {}
            for key in rows[0]:
                columns[key] = [row[key] for row in rows]

            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    return total_rows
