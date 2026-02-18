"""Read a manifest from S3, compute aggregate statistics, and write JSON summary."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import orjson

from imgeda.core.aggregator import aggregate
from imgeda.models.manifest import MANIFEST_META_KEY, ImageRecord


def handle(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Aggregate image records from a manifest into a dataset summary.

    Input event:
        bucket: S3 bucket containing the manifest
        manifest_key: S3 key of the merged manifest JSONL
        output_key: S3 key for the output JSON summary

    Returns:
        output_key: S3 key where the summary was written
        summary: The DatasetSummary as a dict
    """
    import boto3  # type: ignore[import-untyped]

    bucket = event["bucket"]
    manifest_key: str = event["manifest_key"]
    output_key: str = event["output_key"]

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

    # Aggregate
    summary = aggregate(records)
    summary_dict = asdict(summary)

    # Upload summary JSON
    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=orjson.dumps(summary_dict, option=orjson.OPT_INDENT_2),
        ContentType="application/json",
    )

    return {"output_key": output_key, "summary": summary_dict}
