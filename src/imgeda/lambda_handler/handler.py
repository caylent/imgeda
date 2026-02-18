"""AWS Lambda entry point — routes events to action-specific handlers."""

from __future__ import annotations

import os
from typing import Any


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entry point — dispatches to action-specific handlers.

    Routing: checks event["action"] first, then falls back to the ACTION
    environment variable (set by CDK on each Lambda function).
    """
    action = event.get("action") or os.environ.get("ACTION", "").lower()

    if action == "list_images":
        from imgeda.lambda_handler.handlers.list_images import handle

        return handle(event, context)
    elif action == "analyze_batch":
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        return handle(event, context)
    elif action == "merge_manifests":
        from imgeda.lambda_handler.handlers.merge_manifests import handle

        return handle(event, context)
    elif action == "aggregate":
        from imgeda.lambda_handler.handlers.aggregate import handle

        return handle(event, context)
    elif action == "generate_plots":
        from imgeda.lambda_handler.handlers.generate_plots import handle

        return handle(event, context)
    else:
        return {
            "statusCode": 400,
            "body": f"Unknown or missing action: {action!r}. "
            "Supported: list_images, analyze_batch, merge_manifests, aggregate, generate_plots",
        }
