"""Tests for plotting modules."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import CornerStats, ImageRecord, PixelStats
from imgeda.plotting.base import COLORS, apply_theme, direct_label, tufte_axes


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


class TestBaseHelpers:
    """Tests for base.py theme engine and helpers."""

    def test_apply_theme_sets_serif(self) -> None:
        apply_theme()
        assert "serif" in plt.rcParams["font.family"]
        assert plt.rcParams["axes.grid"] is False

    def test_apply_theme_sets_offwhite_bg(self) -> None:
        apply_theme()
        assert plt.rcParams["figure.facecolor"] == COLORS["bg"]
        assert plt.rcParams["axes.facecolor"] == COLORS["bg"]

    def test_apply_theme_tick_label_contrast(self) -> None:
        """Tick labels should be darker than tick marks for readability."""
        apply_theme()
        assert plt.rcParams["xtick.labelcolor"] == COLORS["text"]
        assert plt.rcParams["ytick.labelcolor"] == COLORS["text"]
        assert plt.rcParams["xtick.color"] == COLORS["light"]

    def test_tufte_axes_trims_spines(self) -> None:
        apply_theme()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [10, 20, 30])
        tufte_axes(ax)

        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        # Bottom spine should be bounded to data range
        lo, hi = ax.spines["bottom"].get_bounds()
        assert lo <= 1.0
        assert hi >= 3.0
        plt.close(fig)

    def test_tufte_axes_empty_plot(self) -> None:
        """tufte_axes should not crash on an empty axes."""
        apply_theme()
        fig, ax = plt.subplots()
        tufte_axes(ax)
        plt.close(fig)

    def test_direct_label_adds_text(self) -> None:
        apply_theme()
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        direct_label(ax, 1.5, 1.5, "test label")

        texts = ax.texts
        assert len(texts) == 1
        assert texts[0].get_text() == "test label"
        assert texts[0].get_fontstyle() == "italic"
        plt.close(fig)

    def test_colors_backward_compat(self) -> None:
        """All keys used by plot modules should exist."""
        required = [
            "primary",
            "secondary",
            "highlight",
            "neutral",
            "light",
            "channel_r",
            "channel_g",
            "channel_b",
            "bg",
            "text",
            "text_secondary",
            "danger",
            "success",
            "bg_accent",
        ]
        for key in required:
            assert key in COLORS, f"Missing COLORS key: {key}"


class TestFileSizeHelpers:
    """Tests for file_size.py auto-unit and adaptive bin helpers."""

    def test_auto_unit_kb(self) -> None:
        from imgeda.plotting.file_size import _auto_unit

        val, unit = _auto_unit(500.0)
        assert unit == "KB"
        assert val == 500.0

    def test_auto_unit_mb(self) -> None:
        from imgeda.plotting.file_size import _auto_unit

        val, unit = _auto_unit(2048.0)
        assert unit == "MB"
        assert abs(val - 2.0) < 0.01

    def test_auto_unit_gb(self) -> None:
        from imgeda.plotting.file_size import _auto_unit

        val, unit = _auto_unit(2_097_152.0)  # 2 GB in KB
        assert unit == "GB"
        assert abs(val - 2.0) < 0.01

    def test_format_size_large(self) -> None:
        from imgeda.plotting.file_size import _format_size

        result = _format_size(150_000.0)  # ~146 MB
        assert "MB" in result

    def test_format_size_small(self) -> None:
        from imgeda.plotting.file_size import _format_size

        result = _format_size(5.5)
        assert "KB" in result

    def test_adaptive_bins_small_dataset(self) -> None:
        from imgeda.plotting.file_size import _adaptive_bins

        bins = _adaptive_bins(25, 10.0)
        assert 15 <= bins <= 80

    def test_adaptive_bins_large_dataset(self) -> None:
        from imgeda.plotting.file_size import _adaptive_bins

        bins = _adaptive_bins(10_000, 10.0)
        assert bins >= 15

    def test_adaptive_bins_high_spread(self) -> None:
        from imgeda.plotting.file_size import _adaptive_bins

        bins_low = _adaptive_bins(1000, 10.0)
        bins_high = _adaptive_bins(1000, 5000.0)
        assert bins_high >= bins_low


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

    def test_empty_records_all_plots(self, tmp_path: Path) -> None:
        """All plot functions should handle empty input gracefully."""
        from imgeda.plotting.artifacts import plot_artifacts
        from imgeda.plotting.aspect_ratio import plot_aspect_ratio
        from imgeda.plotting.dimensions import plot_dimensions
        from imgeda.plotting.duplicates import plot_duplicates
        from imgeda.plotting.file_size import plot_file_size
        from imgeda.plotting.pixel_stats import plot_brightness, plot_channels

        config = PlotConfig(output_dir=str(tmp_path))
        for fn in [
            plot_dimensions,
            plot_file_size,
            plot_aspect_ratio,
            plot_brightness,
            plot_channels,
            plot_artifacts,
            plot_duplicates,
        ]:
            path = fn([], config)
            assert Path(path).exists(), f"{fn.__name__} failed on empty input"

    def test_dimensions_no_refs_for_small_images(self, tmp_path: Path) -> None:
        """Reference lines should not appear for tiny images far from any standard resolution."""
        from imgeda.plotting.dimensions import plot_dimensions

        tiny_records = [
            ImageRecord(
                path=f"/data/tiny_{i}.jpg",
                filename=f"tiny_{i}.jpg",
                file_size_bytes=1000,
                width=32,
                height=32,
                format="JPEG",
                color_mode="RGB",
                num_channels=3,
                aspect_ratio=1.0,
            )
            for i in range(10)
        ]
        config = PlotConfig(output_dir=str(tmp_path))
        path = plot_dimensions(tiny_records, config)
        assert Path(path).exists()
