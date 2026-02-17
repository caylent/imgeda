"""Programmatic test image fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    """Create a directory with various programmatic test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Normal images (varying sizes and colors)
    for i in range(10):
        w, h = 200 + i * 50, 150 + i * 30
        arr = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(img_dir / f"normal_{i:03d}.jpg")

    # Dark image
    arr = np.full((100, 100, 3), 15, dtype=np.uint8)
    Image.fromarray(arr).save(img_dir / "dark_001.png")

    # Overexposed image
    arr = np.full((100, 100, 3), 240, dtype=np.uint8)
    Image.fromarray(arr).save(img_dir / "bright_001.png")

    # Image with border artifact (dark corners, bright center)
    arr = np.full((200, 200, 3), 180, dtype=np.uint8)
    arr[:20, :20] = 20  # top-left corner dark
    arr[:20, -20:] = 20  # top-right corner dark
    arr[-20:, :20] = 20  # bottom-left corner dark
    arr[-20:, -20:] = 20  # bottom-right corner dark
    Image.fromarray(arr).save(img_dir / "artifact_001.png")

    # Duplicate pair (exact same content, different filenames)
    arr = np.random.randint(50, 200, (150, 150, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(img_dir / "dup_a.jpg")
    img.save(img_dir / "dup_b.jpg")

    # Corrupt file
    with open(img_dir / "corrupt.jpg", "wb") as f:
        f.write(b"not a real image file content")

    return img_dir


@pytest.fixture
def single_image(tmp_path: Path) -> str:
    """Create a single valid test image."""
    arr = np.random.randint(60, 200, (100, 100, 3), dtype=np.uint8)
    path = tmp_path / "test.jpg"
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture
def large_image(tmp_path: Path) -> str:
    """Create a large image that exceeds max_image_dimension."""
    arr = np.random.randint(60, 200, (3000, 4000, 3), dtype=np.uint8)
    path = tmp_path / "large.jpg"
    Image.fromarray(arr).save(path)
    return str(path)
