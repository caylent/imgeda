"""Tests for core analyzer."""

from __future__ import annotations

from pathlib import Path

import pytest

from imgeda.core.analyzer import analyze_image
from imgeda.models.config import ScanConfig
from imgeda.models.manifest import ImageRecord


class TestAnalyzeImage:
    def test_normal_image(self, single_image: str) -> None:
        config = ScanConfig()
        record = analyze_image(single_image, config)

        assert not record.is_corrupt
        assert record.width == 100
        assert record.height == 100
        assert record.file_size_bytes > 0
        assert record.mtime > 0
        assert record.filename == "test.jpg"
        assert record.aspect_ratio == 1.0
        assert record.pixel_stats is not None
        assert 0 <= record.pixel_stats.mean_brightness <= 255
        assert record.phash is not None
        assert record.dhash is not None
        assert record.analyzed_at != ""

    def test_corrupt_file(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "bad.jpg"
        corrupt.write_bytes(b"not an image")
        config = ScanConfig()
        record = analyze_image(str(corrupt), config)

        assert record.is_corrupt

    def test_nonexistent_file(self) -> None:
        config = ScanConfig()
        record = analyze_image("/nonexistent/image.jpg", config)

        assert record.is_corrupt

    def test_skip_pixel_stats(self, single_image: str) -> None:
        config = ScanConfig(skip_pixel_stats=True)
        record = analyze_image(single_image, config)

        assert not record.is_corrupt
        assert record.pixel_stats is None
        assert record.corner_stats is None

    def test_no_hashes(self, single_image: str) -> None:
        config = ScanConfig(include_hashes=False)
        record = analyze_image(single_image, config)

        assert not record.is_corrupt
        assert record.phash is None
        assert record.dhash is None

    def test_large_image_downsampled(self, large_image: str) -> None:
        config = ScanConfig(max_image_dimension=1024)
        record = analyze_image(large_image, config)

        assert not record.is_corrupt
        assert record.width == 4000  # original dimensions preserved
        assert record.height == 3000
        assert record.pixel_stats is not None

    def test_dark_detection(self, tmp_image_dir: Path) -> None:
        config = ScanConfig()
        record = analyze_image(str(tmp_image_dir / "dark_001.png"), config)

        assert not record.is_corrupt
        assert record.is_dark
        assert not record.is_overexposed

    def test_overexposed_detection(self, tmp_image_dir: Path) -> None:
        config = ScanConfig()
        record = analyze_image(str(tmp_image_dir / "bright_001.png"), config)

        assert not record.is_corrupt
        assert not record.is_dark
        assert record.is_overexposed

    def test_border_artifact_detection(self, tmp_image_dir: Path) -> None:
        config = ScanConfig()
        record = analyze_image(str(tmp_image_dir / "artifact_001.png"), config)

        assert not record.is_corrupt
        assert record.has_border_artifact
        assert record.corner_stats is not None
        assert record.corner_stats.delta > 50


class TestExifExtraction:
    def test_exif_camera_metadata(self, exif_image: str) -> None:
        config = ScanConfig()
        record = analyze_image(exif_image, config)

        assert not record.is_corrupt
        assert record.camera_make == "Canon"
        assert record.camera_model == "Canon EOS 5D Mark IV"
        assert record.orientation_tag == 6

    def test_exif_lens_and_focal_length(self, exif_image: str) -> None:
        config = ScanConfig()
        record = analyze_image(exif_image, config)

        assert record.focal_length_mm == pytest.approx(14.0, abs=0.1)
        assert record.focal_length_35mm == 14
        assert record.lens_model == "Canon EF 14mm f/2.8L II USM"

    def test_exif_exposure_params(self, exif_image: str) -> None:
        config = ScanConfig()
        record = analyze_image(exif_image, config)

        assert record.iso_speed == 400
        assert record.f_number == pytest.approx(2.8, abs=0.1)
        assert record.exposure_time_sec == pytest.approx(0.001, abs=0.0001)
        assert record.datetime_original == "2025:06:15 10:30:00"

    def test_distortion_risk_high(self, exif_image: str) -> None:
        """14mm focal length should be flagged as high distortion risk."""
        config = ScanConfig()
        record = analyze_image(exif_image, config)

        assert record.distortion_risk == "high"

    def test_distortion_risk_low(self, exif_image_no_distortion: str) -> None:
        """50mm focal length should be low distortion risk."""
        config = ScanConfig()
        record = analyze_image(exif_image_no_distortion, config)

        assert record.distortion_risk == "low"
        assert record.camera_make == "Nikon"
        assert record.focal_length_35mm == 50

    def test_skip_exif(self, exif_image: str) -> None:
        config = ScanConfig(skip_exif=True)
        record = analyze_image(exif_image, config)

        assert not record.is_corrupt
        assert record.camera_make is None
        assert record.camera_model is None
        assert record.focal_length_mm is None

    def test_no_exif_data(self, single_image: str) -> None:
        """Images without EXIF should have None/False for all EXIF fields."""
        config = ScanConfig()
        record = analyze_image(single_image, config)

        assert not record.is_corrupt
        assert record.camera_make is None
        assert record.focal_length_mm is None
        assert record.distortion_risk is None
        assert record.has_gps_data is False

    def test_exif_roundtrip_via_dict(self, exif_image: str) -> None:
        """EXIF fields should survive to_dict/from_dict serialization."""
        config = ScanConfig()
        record = analyze_image(exif_image, config)

        data = record.to_dict()
        restored = ImageRecord.from_dict(data)

        assert restored.camera_make == record.camera_make
        assert restored.focal_length_35mm == record.focal_length_35mm
        assert restored.distortion_risk == record.distortion_risk
        assert restored.iso_speed == record.iso_speed
