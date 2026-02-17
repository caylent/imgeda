"""Phase 2 Lambda handler stub for S3 Batch processing.

Receives events like:
    {
        "bucket": "my-images",
        "keys": ["path/to/image1.jpg", "path/to/image2.jpg"],
        "config": { ... ScanConfig overrides ... }
    }

Downloads images to /tmp, runs analyze_image(), writes JSONL to results bucket.
"""

from __future__ import annotations

from typing import Any


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entry point — Phase 2 implementation."""
    # TODO: Phase 2 implementation
    # 1. Parse event: bucket, keys, config, results_bucket
    # 2. For each key: download from S3 to /tmp
    # 3. Run analyze_image() on each
    # 4. Serialize results as JSONL
    # 5. Upload JSONL fragment to results bucket
    # 6. Return summary (processed count, errors)
    return {
        "statusCode": 501,
        "body": "Lambda handler not yet implemented. Phase 2.",
    }
