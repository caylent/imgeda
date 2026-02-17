"""Core image analysis — pure function, Lambda-compatible."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from imgeda.core.detector import (
    compute_corner_stats,
    compute_pixel_stats,
    has_border_artifact,
    is_dark,
    is_overexposed,
)
from imgeda.core.hasher import compute_dhash, compute_phash
from imgeda.models.config import ScanConfig
from imgeda.models.manifest import ImageRecord


def analyze_image(path: str, config: ScanConfig) -> ImageRecord:
    """Analyze a single image. Never raises — errors set is_corrupt=True."""
    record = ImageRecord(
        path=path,
        filename=os.path.basename(path),
        analyzed_at=datetime.now(timezone.utc).isoformat(),
    )

    # Step 1: File metadata
    try:
        stat = os.stat(path)
        record.file_size_bytes = stat.st_size
        record.mtime = stat.st_mtime
    except OSError:
        record.is_corrupt = True
        return record

    # Step 2: Open and verify image
    try:
        img = Image.open(path)
        img.verify()
        # Re-open after verify (verify closes the file)
        img = Image.open(path)
    except Exception:
        record.is_corrupt = True
        return record

    try:
        record.width = img.width
        record.height = img.height
        record.format = img.format or ""
        record.color_mode = img.mode
        record.num_channels = len(img.getbands())
        record.aspect_ratio = round(img.width / img.height, 4) if img.height > 0 else 0.0

        # Convert to RGB for analysis
        rgb = img.convert("RGB")

        # Downsample if too large
        max_dim = config.max_image_dimension
        if rgb.width > max_dim or rgb.height > max_dim:
            ratio = max_dim / max(rgb.width, rgb.height)
            new_size = (int(rgb.width * ratio), int(rgb.height * ratio))
            rgb = rgb.resize(new_size, Image.LANCZOS)

        pixels = np.array(rgb, dtype=np.uint8)

        # Step 3: Pixel stats
        if not config.skip_pixel_stats:
            record.pixel_stats = compute_pixel_stats(pixels)
            record.is_dark = is_dark(record.pixel_stats, config.dark_threshold)
            record.is_overexposed = is_overexposed(record.pixel_stats, config.overexposed_threshold)

        # Step 4: Corner stats
        if not config.skip_pixel_stats:
            record.corner_stats = compute_corner_stats(pixels, config.corner_patch_fraction)
            record.has_border_artifact = has_border_artifact(
                record.corner_stats, config.artifact_threshold
            )

        # Step 5: Perceptual hashes
        if config.include_hashes:
            record.phash = compute_phash(rgb, config.hash_size)
            record.dhash = compute_dhash(rgb, config.hash_size)

    except Exception:
        record.is_corrupt = True

    return record
