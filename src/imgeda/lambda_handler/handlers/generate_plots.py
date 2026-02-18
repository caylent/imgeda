"""Read a manifest from S3, generate all plots, and upload PNGs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import orjson

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import MANIFEST_META_KEY, ImageRecord
from imgeda.plotting.artifacts import plot_artifacts
from imgeda.plotting.aspect_ratio import plot_aspect_ratio
from imgeda.plotting.dimensions import plot_dimensions
from imgeda.plotting.duplicates import plot_duplicates
from imgeda.plotting.file_size import plot_file_size
from imgeda.plotting.pixel_stats import plot_brightness, plot_channels

ALL_PLOT_FUNCTIONS = [
    plot_dimensions,
    plot_file_size,
    plot_aspect_ratio,
    plot_brightness,
    plot_channels,
    plot_artifacts,
    plot_duplicates,
]


def handle(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Generate all plots from a manifest and upload to S3.

    Input event:
        bucket: S3 bucket containing the manifest
        manifest_key: S3 key of the merged manifest JSONL
        output_prefix: S3 prefix for uploading plot PNGs

    Returns:
        plots: List of S3 keys for uploaded plot PNGs
    """
    import boto3  # type: ignore[import-untyped]

    bucket = event["bucket"]
    manifest_key: str = event["manifest_key"]
    output_prefix: str = event.get("output_prefix", "plots/")

    s3 = boto3.client("s3")

    # Read manifest
    resp = s3.get_object(Bucket=bucket, Key=manifest_key)
    body = resp["Body"].read()

    records: list[ImageRecord] = []
    for line in body.split(b"\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        if data.get(MANIFEST_META_KEY):
            continue
        records.append(ImageRecord.from_dict(data))

    if not records:
        return {"plots": []}

    uploaded_keys: list[str] = []
    skipped: list[str] = []

    with tempfile.TemporaryDirectory(prefix="imgeda_plots_") as tmp_dir:
        config = PlotConfig(output_dir=tmp_dir)

        for plot_fn in ALL_PLOT_FUNCTIONS:
            try:
                local_path = plot_fn(records, config)
                filename = Path(local_path).name
                s3_key = f"{output_prefix.rstrip('/')}/{filename}"

                s3.upload_file(local_path, bucket, s3_key)
                uploaded_keys.append(s3_key)
            except Exception as exc:
                skipped.append(f"{plot_fn.__name__}: {exc!r}")
                continue

    return {"plots": uploaded_keys, "skipped": skipped}
