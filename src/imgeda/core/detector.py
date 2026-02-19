"""Exposure and artifact detection from pixel data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from imgeda.models.manifest import CornerStats, PixelStats


def compute_pixel_stats(pixels: NDArray[np.uint8]) -> PixelStats:
    """Compute per-channel statistics from an RGB numpy array (H, W, 3)."""
    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    brightness = pixels.mean(axis=2)
    return PixelStats(
        mean_r=float(r.mean()),
        mean_g=float(g.mean()),
        mean_b=float(b.mean()),
        std_r=float(r.std()),
        std_g=float(g.std()),
        std_b=float(b.std()),
        mean_brightness=float(brightness.mean()),
        min_val=int(pixels.min()),
        max_val=int(pixels.max()),
    )


def compute_corner_stats(pixels: NDArray[np.uint8], patch_fraction: float = 0.1) -> CornerStats:
    """Compute corner vs center brightness delta for artifact detection."""
    h, w = pixels.shape[:2]
    ph = max(1, int(h * patch_fraction))
    pw = max(1, int(w * patch_fraction))

    corners = [
        pixels[:ph, :pw],  # top-left
        pixels[:ph, -pw:],  # top-right
        pixels[-ph:, :pw],  # bottom-left
        pixels[-ph:, -pw:],  # bottom-right
    ]
    corner_vals = np.concatenate([c.ravel() for c in corners])
    corner_mean = float(corner_vals.mean())

    ch, cw = h // 4, w // 4
    center = pixels[ch : h - ch, cw : w - cw]
    center_mean = float(center.mean())

    border_top = pixels[:ph, :]
    border_bottom = pixels[-ph:, :]
    border_left = pixels[:, :pw]
    border_right = pixels[:, -pw:]
    border_vals = np.concatenate(
        [
            border_top.ravel(),
            border_bottom.ravel(),
            border_left.ravel(),
            border_right.ravel(),
        ]
    )
    border_mean = float(border_vals.mean())

    delta = abs(corner_mean - center_mean)
    return CornerStats(
        corner_mean=corner_mean,
        center_mean=center_mean,
        border_mean=border_mean,
        delta=delta,
    )


def is_dark(pixel_stats: PixelStats, threshold: float) -> bool:
    return pixel_stats.mean_brightness < threshold


def is_overexposed(pixel_stats: PixelStats, threshold: float) -> bool:
    return pixel_stats.mean_brightness > threshold


def has_border_artifact(corner_stats: CornerStats, threshold: float) -> bool:
    return corner_stats.delta > threshold


def compute_blur_score(pixels: NDArray[np.uint8]) -> float:
    """Compute blur score via Laplacian variance. Lower = blurrier."""
    gray = pixels.mean(axis=2).astype(np.float64)
    # Laplacian kernel convolution via numpy
    laplacian = (
        np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
        - 4 * gray
    )
    return float(laplacian.var())
