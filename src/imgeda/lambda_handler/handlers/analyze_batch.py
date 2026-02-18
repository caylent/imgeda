"""Download images from S3, analyze each, and upload JSONL results."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import orjson

from imgeda.core.analyzer import analyze_image
from imgeda.models.config import ScanConfig


def handle(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Analyze a batch of images from S3.

    Input event:
        source_bucket: S3 bucket containing images
        keys: List of S3 keys to analyze
        output_bucket: S3 bucket for results
        output_key: S3 key for the output JSONL file
        config: Optional ScanConfig overrides (dict)

    Returns:
        processed: Number of successfully analyzed images
        errors: Number of images that could not be analyzed
        output_key: S3 key where results were written
    """
    import boto3  # type: ignore[import-untyped]

    source_bucket = event["source_bucket"]
    keys: list[str] = event["keys"]
    output_bucket = event["output_bucket"]
    output_key: str = event["output_key"]
    config_overrides = event.get("config", {})

    # Build ScanConfig from overrides
    config_kwargs: dict[str, Any] = {}
    for field_name in ScanConfig.__dataclass_fields__:
        if field_name in config_overrides:
            config_kwargs[field_name] = config_overrides[field_name]
    config = ScanConfig(**config_kwargs)

    s3 = boto3.client("s3")
    processed = 0
    errors = 0
    lines: list[bytes] = []

    for key in keys:
        tmp_path = ""
        fd = -1
        try:
            # Download to /tmp
            suffix = os.path.splitext(key)[1] or ".jpg"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir="/tmp")
            os.close(fd)
            fd = -1  # Mark as closed

            s3.download_file(source_bucket, key, tmp_path)

            # Analyze
            record = analyze_image(tmp_path, config)
            # Preserve original S3 path instead of local /tmp path
            record.path = f"s3://{source_bucket}/{key}"
            record.filename = os.path.basename(key)

            lines.append(orjson.dumps(record.to_dict()))
            processed += 1
        except Exception:
            errors += 1
        finally:
            if fd >= 0:
                os.close(fd)
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Upload JSONL to output bucket
    body = b"\n".join(lines) + b"\n" if lines else b""
    s3.put_object(Bucket=output_bucket, Key=output_key, Body=body)

    return {"processed": processed, "errors": errors, "output_key": output_key}
