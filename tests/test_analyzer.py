"""Tests for core analyzer."""

from __future__ import annotations

from pathlib import Path

from imgeda.core.analyzer import analyze_image
from imgeda.models.config import ScanConfig


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
