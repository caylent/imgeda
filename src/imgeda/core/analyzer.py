"""Core image analysis — pure function, Lambda-compatible."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

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

# EXIF tag IDs (numeric to avoid dependency on PIL.ExifTags enum names)
_EXIF_MAKE = 0x010F  # 271 — Camera manufacturer
_EXIF_MODEL = 0x0110  # 272 — Camera model
_EXIF_ORIENTATION = 0x0112  # 274 — Image rotation tag
_EXIF_EXPOSURE_TIME = 0x829A  # 33434 — Shutter speed (seconds)
_EXIF_FNUMBER = 0x829D  # 33437 — Aperture f-number
_EXIF_ISO = 0x8827  # 34855 — ISO sensitivity
_EXIF_DATETIME_ORIGINAL = 0x9003  # 36867 — Original capture datetime
_EXIF_FOCAL_LENGTH = 0x920A  # 37386 — Focal length in mm
_EXIF_FOCAL_LENGTH_35MM = 0xA405  # 41989 — 35mm equivalent focal length
_EXIF_LENS_MODEL = 0xA434  # 42036 — Lens model string
_EXIF_GPS_INFO = 0x8825  # 34853 — GPS info sub-IFD pointer
_EXIF_EXIF_IFD = 0x8769  # 34665 — ExifIFD sub-IFD pointer


def _exif_str(exif: Any, tag: int) -> str | None:
    """Extract a stripped string from EXIF data."""
    val = exif.get(tag)
    if val is None:
        return None
    try:
        s = str(val).strip().rstrip("\x00")
        return s if s else None
    except Exception:
        return None


def _exif_float(exif: Any, tag: int) -> float | None:
    """Extract a float from EXIF, handling IFDRational values."""
    val = exif.get(tag)
    if val is None:
        return None
    try:
        result = float(val)
        return result if result > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _exif_int(exif: Any, tag: int) -> int | None:
    """Extract an int from EXIF data."""
    val = exif.get(tag)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _extract_exif(img: Image.Image, record: ImageRecord) -> None:
    """Extract EXIF metadata into an ImageRecord. Non-fatal on any error."""
    exif = img.getexif()
    if not exif:
        return

    # IFD0 tags (always available from getexif)
    record.camera_make = _exif_str(exif, _EXIF_MAKE)
    record.camera_model = _exif_str(exif, _EXIF_MODEL)

    orientation = _exif_int(exif, _EXIF_ORIENTATION)
    if orientation is not None:
        record.orientation_tag = orientation

    # ExifIFD tags — try both direct access and sub-IFD lookup
    exif_ifd = exif.get_ifd(_EXIF_EXIF_IFD)
    sources = [exif, exif_ifd] if exif_ifd else [exif]

    for src in sources:
        if record.focal_length_mm is None:
            record.focal_length_mm = _exif_float(src, _EXIF_FOCAL_LENGTH)
        if record.focal_length_35mm is None:
            record.focal_length_35mm = _exif_int(src, _EXIF_FOCAL_LENGTH_35MM)
        if record.iso_speed is None:
            iso_val = src.get(_EXIF_ISO)
            if iso_val is not None:
                try:
                    record.iso_speed = (
                        int(iso_val[0]) if isinstance(iso_val, (tuple, list)) else int(iso_val)
                    )
                except (TypeError, ValueError, IndexError):
                    pass
        if record.f_number is None:
            record.f_number = _exif_float(src, _EXIF_FNUMBER)
        if record.exposure_time_sec is None:
            record.exposure_time_sec = _exif_float(src, _EXIF_EXPOSURE_TIME)
        if record.datetime_original is None:
            record.datetime_original = _exif_str(src, _EXIF_DATETIME_ORIGINAL)
        if record.lens_model is None:
            record.lens_model = _exif_str(src, _EXIF_LENS_MODEL)

    # GPS presence check
    if _EXIF_GPS_INFO in exif or exif.get_ifd(_EXIF_GPS_INFO):
        record.has_gps_data = True

    # Compute distortion risk from 35mm-equivalent focal length
    fl35 = record.focal_length_35mm
    if fl35 is not None:
        if fl35 < 18:
            record.distortion_risk = "high"
        elif fl35 < 24:
            record.distortion_risk = "medium"
        else:
            record.distortion_risk = "low"


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

        # Step 3: EXIF metadata
        if not config.skip_exif:
            try:
                _extract_exif(img, record)
            except Exception:
                pass  # EXIF extraction failure is non-fatal

        # Convert to RGB for analysis
        rgb = img.convert("RGB")

        # Downsample if too large
        max_dim = config.max_image_dimension
        if rgb.width > max_dim or rgb.height > max_dim:
            ratio = max_dim / max(rgb.width, rgb.height)
            new_size = (int(rgb.width * ratio), int(rgb.height * ratio))
            rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)

        pixels = np.array(rgb, dtype=np.uint8)

        # Step 4: Pixel stats
        if not config.skip_pixel_stats:
            record.pixel_stats = compute_pixel_stats(pixels)
            record.is_dark = is_dark(record.pixel_stats, config.dark_threshold)
            record.is_overexposed = is_overexposed(record.pixel_stats, config.overexposed_threshold)

        # Step 5: Corner stats
        if not config.skip_pixel_stats:
            record.corner_stats = compute_corner_stats(pixels, config.corner_patch_fraction)
            record.has_border_artifact = has_border_artifact(
                record.corner_stats, config.artifact_threshold
            )

        # Step 6: Perceptual hashes
        if config.include_hashes:
            record.phash = compute_phash(rgb, config.hash_size)
            record.dhash = compute_dhash(rgb, config.hash_size)

    except Exception:
        record.is_corrupt = True

    return record
