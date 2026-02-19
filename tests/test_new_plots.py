"""Tests for new plot modules: blur, exif, annotations, embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imgeda.core.annotations import AnnotationStats
from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord


@pytest.fixture
def plot_config(tmp_path: Path) -> PlotConfig:
    return PlotConfig(output_dir=str(tmp_path))


@pytest.fixture
def records_with_blur() -> list[ImageRecord]:
    records = []
    for i in range(30):
        records.append(
            ImageRecord(
                path=f"/data/img_{i:03d}.jpg",
                filename=f"img_{i:03d}.jpg",
                blur_score=50.0 + i * 10,
                is_blurry=i < 5,
            )
        )
    return records


@pytest.fixture
def records_with_exif() -> list[ImageRecord]:
    records = []
    cameras = ["Canon EOS 5D", "Nikon D850", "Sony A7III"]
    for i in range(30):
        records.append(
            ImageRecord(
                path=f"/data/img_{i:03d}.jpg",
                filename=f"img_{i:03d}.jpg",
                camera_make=cameras[i % 3].split()[0],
                camera_model=cameras[i % 3],
                focal_length_35mm=24 + i * 5,
                iso_speed=100 + i * 200,
            )
        )
    return records


@pytest.fixture
def annotation_stats() -> AnnotationStats:
    stats = AnnotationStats(
        total_images=100,
        annotated_images=90,
        total_annotations=500,
        num_classes=5,
        class_counts={"cat": 200, "dog": 150, "bird": 80, "fish": 40, "snake": 30},
        objects_per_image=[5] * 90 + [0] * 10,
        mean_objects_per_image=4.5,
        max_objects_per_image=5,
        bbox_widths=[0.1 + i * 0.01 for i in range(500)],
        bbox_heights=[0.1 + i * 0.005 for i in range(500)],
        bbox_areas=[0.01 + i * 0.001 for i in range(500)],
        bbox_x_centers=[i / 500 for i in range(500)],
        bbox_y_centers=[i / 500 for i in range(500)],
        small_count=100,
        medium_count=300,
        large_count=100,
        co_occurrence={"cat": {"dog": 50, "bird": 10}, "dog": {"cat": 50}},
    )
    return stats


class TestBlurPlot:
    def test_generates_plot(
        self, records_with_blur: list[ImageRecord], plot_config: PlotConfig
    ) -> None:
        from imgeda.plotting.blur import plot_blur

        path = plot_blur(records_with_blur, plot_config)
        assert Path(path).exists()

    def test_empty_records(self, plot_config: PlotConfig) -> None:
        from imgeda.plotting.blur import plot_blur

        path = plot_blur([], plot_config)
        assert Path(path).exists()

    def test_no_blur_data(self, plot_config: PlotConfig) -> None:
        from imgeda.plotting.blur import plot_blur

        records = [ImageRecord(path="/a.jpg", filename="a.jpg")]
        path = plot_blur(records, plot_config)
        assert Path(path).exists()


class TestExifPlots:
    def test_camera_distribution(
        self, records_with_exif: list[ImageRecord], plot_config: PlotConfig
    ) -> None:
        from imgeda.plotting.exif import plot_camera_distribution

        path = plot_camera_distribution(records_with_exif, plot_config)
        assert Path(path).exists()

    def test_focal_length(
        self, records_with_exif: list[ImageRecord], plot_config: PlotConfig
    ) -> None:
        from imgeda.plotting.exif import plot_focal_length

        path = plot_focal_length(records_with_exif, plot_config)
        assert Path(path).exists()

    def test_iso_distribution(
        self, records_with_exif: list[ImageRecord], plot_config: PlotConfig
    ) -> None:
        from imgeda.plotting.exif import plot_iso_distribution

        path = plot_iso_distribution(records_with_exif, plot_config)
        assert Path(path).exists()

    def test_camera_empty(self, plot_config: PlotConfig) -> None:
        from imgeda.plotting.exif import plot_camera_distribution

        path = plot_camera_distribution([], plot_config)
        assert Path(path).exists()

    def test_focal_length_empty(self, plot_config: PlotConfig) -> None:
        from imgeda.plotting.exif import plot_focal_length

        path = plot_focal_length([], plot_config)
        assert Path(path).exists()

    def test_iso_empty(self, plot_config: PlotConfig) -> None:
        from imgeda.plotting.exif import plot_iso_distribution

        path = plot_iso_distribution([], plot_config)
        assert Path(path).exists()

    def test_no_exif_data(self, plot_config: PlotConfig) -> None:
        from imgeda.plotting.exif import (
            plot_camera_distribution,
            plot_focal_length,
            plot_iso_distribution,
        )

        records = [ImageRecord(path="/a.jpg", filename="a.jpg")]
        for fn in [plot_camera_distribution, plot_focal_length, plot_iso_distribution]:
            path = fn(records, plot_config)
            assert Path(path).exists()


class TestAnnotationPlots:
    def test_class_frequency(
        self, annotation_stats: AnnotationStats, plot_config: PlotConfig
    ) -> None:
        from imgeda.plotting.annotations import plot_class_frequency

        path = plot_class_frequency(annotation_stats, plot_config)
        assert Path(path).exists()

    def test_bbox_sizes(self, annotation_stats: AnnotationStats, plot_config: PlotConfig) -> None:
        from imgeda.plotting.annotations import plot_bbox_sizes

        path = plot_bbox_sizes(annotation_stats, plot_config)
        assert Path(path).exists()

    def test_objects_per_image(
        self, annotation_stats: AnnotationStats, plot_config: PlotConfig
    ) -> None:
        from imgeda.plotting.annotations import plot_objects_per_image

        path = plot_objects_per_image(annotation_stats, plot_config)
        assert Path(path).exists()

    def test_co_occurrence(
        self, annotation_stats: AnnotationStats, plot_config: PlotConfig
    ) -> None:
        from imgeda.plotting.annotations import plot_co_occurrence

        path = plot_co_occurrence(annotation_stats, plot_config)
        assert Path(path).exists()

    def test_annotation_coverage(
        self, annotation_stats: AnnotationStats, plot_config: PlotConfig
    ) -> None:
        from imgeda.plotting.annotations import plot_annotation_coverage

        path = plot_annotation_coverage(annotation_stats, plot_config)
        assert Path(path).exists()

    def test_empty_stats(self, plot_config: PlotConfig) -> None:
        from imgeda.plotting.annotations import (
            plot_annotation_coverage,
            plot_bbox_sizes,
            plot_class_frequency,
            plot_co_occurrence,
            plot_objects_per_image,
        )

        empty = AnnotationStats()
        for fn in [
            plot_class_frequency,
            plot_bbox_sizes,
            plot_objects_per_image,
            plot_co_occurrence,
            plot_annotation_coverage,
        ]:
            path = fn(empty, plot_config)
            assert Path(path).exists(), f"{fn.__name__} failed on empty stats"


class TestUmapPlot:
    def test_basic_plot(self, plot_config: PlotConfig) -> None:
        from imgeda.plotting.embeddings import plot_umap

        projection = np.random.randn(50, 2).astype(np.float32)
        path = plot_umap(projection, plot_config)
        assert Path(path).exists()

    def test_with_outlier_mask(self, plot_config: PlotConfig) -> None:
        from imgeda.plotting.embeddings import plot_umap

        projection = np.random.randn(50, 2).astype(np.float32)
        mask = np.zeros(50, dtype=np.bool_)
        mask[:5] = True
        path = plot_umap(projection, plot_config, outlier_mask=mask)
        assert Path(path).exists()
