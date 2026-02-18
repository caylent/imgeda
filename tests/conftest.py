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


@pytest.fixture
def exif_image(tmp_path: Path) -> str:
    """Create a JPEG with EXIF metadata (camera, lens, focal length, GPS flag)."""
    arr = np.random.randint(60, 200, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    exif = img.getexif()

    # IFD0 tags
    exif[0x010F] = "Canon"  # Make
    exif[0x0110] = "Canon EOS 5D Mark IV"  # Model
    exif[0x0112] = 6  # Orientation (90° CW)

    # ExifIFD tags — write into the sub-IFD
    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x829A] = 0.001  # ExposureTime (1/1000s)
    exif_ifd[0x829D] = 2.8  # FNumber
    exif_ifd[0x8827] = 400  # ISOSpeedRatings
    exif_ifd[0x9003] = "2025:06:15 10:30:00"  # DateTimeOriginal
    exif_ifd[0x920A] = 14.0  # FocalLength (14mm)
    exif_ifd[0xA405] = 14  # FocalLengthIn35mmFilm
    exif_ifd[0xA434] = "Canon EF 14mm f/2.8L II USM"  # LensModel

    path = tmp_path / "exif_test.jpg"
    img.save(path, exif=exif.tobytes())
    return str(path)


@pytest.fixture
def exif_image_no_distortion(tmp_path: Path) -> str:
    """Create a JPEG with EXIF metadata for a normal focal length (no distortion risk)."""
    arr = np.random.randint(60, 200, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    exif = img.getexif()

    exif[0x010F] = "Nikon"  # Make
    exif[0x0110] = "Nikon D850"  # Model

    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x920A] = 50.0  # FocalLength (50mm)
    exif_ifd[0xA405] = 50  # FocalLengthIn35mmFilm
    exif_ifd[0xA434] = "Nikon AF-S NIKKOR 50mm f/1.4G"  # LensModel

    path = tmp_path / "exif_normal.jpg"
    img.save(path, exif=exif.tobytes())
    return str(path)
