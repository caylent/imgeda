"""AWS Lambda entry point — routes events to action-specific handlers."""

from __future__ import annotations

from typing import Any


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entry point — dispatches to action-specific handlers.

    Events must include an "action" field:
        list_images, analyze_batch, merge_manifests, aggregate, generate_plots
    """
    action = event.get("action")

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
