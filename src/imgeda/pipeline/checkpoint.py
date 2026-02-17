"""Resume key logic for checkpoint/resume."""

from __future__ import annotations

import os

from imgeda.io.manifest_io import build_resume_set, read_manifest
from imgeda.models.manifest import ImageRecord


def load_processed_set(manifest_path: str) -> tuple[set[tuple[str, int, float]], list[ImageRecord]]:
    """Load already-processed image keys from an existing manifest."""
    _meta, records = read_manifest(manifest_path)
    resume_set = build_resume_set(records)
    return resume_set, records


def filter_pending(
    image_paths: list[str],
    processed: set[tuple[str, int, float]],
) -> list[str]:
    """Filter out already-processed images, returning only pending ones."""
    pending: list[str] = []
    for path in image_paths:
        try:
            stat = os.stat(path)
            key = (path, stat.st_size, stat.st_mtime)
            if key not in processed:
                pending.append(path)
        except OSError:
            pending.append(path)  # let analyzer handle the error
    return pending
