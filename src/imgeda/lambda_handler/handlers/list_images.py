"""List images in an S3 bucket and split into batches for parallel processing."""

from __future__ import annotations

import posixpath
from typing import Any

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif")
DEFAULT_BATCH_SIZE = 20


def handle(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """List images in S3 and return batched key lists.

    Input event:
        bucket: S3 bucket name
        prefix: Key prefix to scan (default "")
        batch_size: Number of images per batch (default 20)
        extensions: List of allowed extensions (default common image exts)

    Returns:
        batches: List of key lists, each up to batch_size
        total_images: Total number of images found
    """
    import boto3  # type: ignore[import-not-found]

    bucket = event["bucket"]
    prefix = event.get("prefix", "")
    batch_size = event.get("batch_size", DEFAULT_BATCH_SIZE)
    extensions = tuple(event.get("extensions", DEFAULT_EXTENSIONS))

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            ext = posixpath.splitext(key)[1].lower()
            if ext in extensions:
                keys.append(key)

    batches = [keys[i : i + batch_size] for i in range(0, len(keys), batch_size)]

    return {"batches": batches, "total_images": len(keys)}
