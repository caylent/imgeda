"""Tests for plotting modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import CornerStats, ImageRecord, PixelStats


@pytest.fixture
def sample_records() -> list[ImageRecord]:
    """Create a list of sample records for plotting tests."""
    records = []
    for i in range(50):
        records.append(
            ImageRecord(
                path=f"/data/img_{i:03d}.jpg",
                filename=f"img_{i:03d}.jpg",
                file_size_bytes=(i + 1) * 10000,
                width=200 + i * 20,
                height=150 + i * 15,
                format="JPEG",
                color_mode="RGB",
                num_channels=3,
                aspect_ratio=round((200 + i * 20) / (150 + i * 15), 4),
                pixel_stats=PixelStats(
                    mean_r=100 + i,
                    mean_g=110 + i,
                    mean_b=120 + i,
                    std_r=20.0,
                    std_g=22.0,
                    std_b=24.0,
                    mean_brightness=110 + i,
                ),
                corner_stats=CornerStats(
                    corner_mean=100 + i,
                    center_mean=130 + i,
                    border_mean=115 + i,
                    delta=30 + i,
                ),
                phash=f"{i:064x}",
            )
        )
    return records


class TestPlots:
    def test_dimensions_plot(self, sample_records: list[ImageRecord], tmp_path: Path) -> None:
        from imgeda.plotting.dimensions import plot_dimensions

        config = PlotConfig(output_dir=str(tmp_path))
        path = plot_dimensions(sample_records, config)
        assert Path(path).exists()

    def test_file_size_plot(self, sample_records: list[ImageRecord], tmp_path: Path) -> None:
        from imgeda.plotting.file_size import plot_file_size

        config = PlotConfig(output_dir=str(tmp_path))
        path = plot_file_size(sample_records, config)
        assert Path(path).exists()

    def test_aspect_ratio_plot(self, sample_records: list[ImageRecord], tmp_path: Path) -> None:
        from imgeda.plotting.aspect_ratio import plot_aspect_ratio

        config = PlotConfig(output_dir=str(tmp_path))
        path = plot_aspect_ratio(sample_records, config)
        assert Path(path).exists()

    def test_brightness_plot(self, sample_records: list[ImageRecord], tmp_path: Path) -> None:
        from imgeda.plotting.pixel_stats import plot_brightness

        config = PlotConfig(output_dir=str(tmp_path))
        path = plot_brightness(sample_records, config)
        assert Path(path).exists()

    def test_channels_plot(self, sample_records: list[ImageRecord], tmp_path: Path) -> None:
        from imgeda.plotting.pixel_stats import plot_channels

        config = PlotConfig(output_dir=str(tmp_path))
        path = plot_channels(sample_records, config)
        assert Path(path).exists()

    def test_artifacts_plot(self, sample_records: list[ImageRecord], tmp_path: Path) -> None:
        from imgeda.plotting.artifacts import plot_artifacts

        config = PlotConfig(output_dir=str(tmp_path))
        path = plot_artifacts(sample_records, config)
        assert Path(path).exists()

    def test_duplicates_plot(self, sample_records: list[ImageRecord], tmp_path: Path) -> None:
        from imgeda.plotting.duplicates import plot_duplicates

        config = PlotConfig(output_dir=str(tmp_path))
        path = plot_duplicates(sample_records, config)
        assert Path(path).exists()

    def test_empty_records(self, tmp_path: Path) -> None:
        from imgeda.plotting.dimensions import plot_dimensions

        config = PlotConfig(output_dir=str(tmp_path))
        # Should handle empty gracefully
        path = plot_dimensions([], config)
        assert Path(path).exists()
