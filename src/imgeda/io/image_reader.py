"""Unified image reading abstraction (Pillow, with optional OpenCV)."""

from __future__ import annotations

import os
from pathlib import Path


def discover_images(
    root: str | Path,
    extensions: tuple[str, ...],
) -> list[str]:
    """Recursively discover image files under root, sorted by path."""
    root = Path(root)
    found: list[str] = []
    ext_set = {e.lower() for e in extensions}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if any(fn.lower().endswith(ext) for ext in ext_set):
                found.append(os.path.join(dirpath, fn))
    found.sort()
    return found
