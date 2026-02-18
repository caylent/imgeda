"""Merge partial JSONL manifest files from S3 into a single manifest."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import orjson

from imgeda.models.manifest import MANIFEST_META_KEY, ImageRecord, ManifestMeta


def handle(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Merge partial JSONL files into a single manifest with metadata header.

    Input event:
        bucket: S3 bucket containing partial manifests
        partial_keys: List of S3 keys for partial JSONL files
        output_key: S3 key for the merged manifest
        input_dir: Original input directory/prefix for metadata

    Returns:
        total_records: Number of records in the merged manifest
        output_key: S3 key where the manifest was written
    """
    import boto3  # type: ignore[import-not-found]

    bucket = event["bucket"]
    partial_keys: list[str] = event["partial_keys"]
    output_key: str = event["output_key"]
    input_dir: str = event.get("input_dir", "")

    s3 = boto3.client("s3")
    all_records: list[ImageRecord] = []

    for key in partial_keys:
        resp = s3.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()
        for line in body.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            # Skip any accidental meta lines in partials
            if data.get(MANIFEST_META_KEY):
                continue
            all_records.append(ImageRecord.from_dict(data))

    # Build manifest with metadata header
    meta = ManifestMeta(
        input_dir=input_dir,
        total_files=len(all_records),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    lines: list[bytes] = [orjson.dumps(meta.to_dict())]
    for rec in all_records:
        lines.append(orjson.dumps(rec.to_dict()))

    body_bytes = b"\n".join(lines) + b"\n"
    s3.put_object(Bucket=bucket, Key=output_key, Body=body_bytes)

    return {"total_records": len(all_records), "output_key": output_key}
