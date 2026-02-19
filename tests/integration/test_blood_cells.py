"""Integration tests using the Blood Cells COCO dataset from HuggingFace.

Dataset: keremberke/blood-cell-object-detection (valid-mini split)
Format: COCO with 3 images, 43 annotations, 3 classes (platelets, rbc, wbc)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from imgeda.cli.app import app
from imgeda.core.annotations import analyze_annotations
from imgeda.core.format_detector import detect_format

runner = CliRunner()

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "blood_cells"


@pytest.fixture
def dataset_dir() -> str:
    if not FIXTURE_DIR.is_dir():
        pytest.skip("Blood cells fixture not found")
    return str(FIXTURE_DIR)


class TestFormatDetection:
    def test_detects_coco(self, dataset_dir: str) -> None:
        info = detect_format(dataset_dir)
        assert info.format == "coco"
        assert info.num_classes == 3
        assert info.annotations_path is not None

    def test_class_names(self, dataset_dir: str) -> None:
        info = detect_format(dataset_dir)
        assert info.class_names is not None
        assert "platelets" in info.class_names
        assert "rbc" in info.class_names
        assert "wbc" in info.class_names


class TestAnnotationAnalysis:
    def test_coco_analysis(self, dataset_dir: str) -> None:
        stats = analyze_annotations(dataset_dir, "coco")
        assert stats.total_annotations == 43
        assert stats.num_classes == 3
        assert stats.annotated_images == 3
        assert "rbc" in stats.class_counts
        assert "wbc" in stats.class_counts
        assert "platelets" in stats.class_counts

    def test_class_distribution(self, dataset_dir: str) -> None:
        stats = analyze_annotations(dataset_dir, "coco")
        # RBC should be the most common (red blood cells are abundant)
        assert stats.class_counts["rbc"] > stats.class_counts["wbc"]

    def test_bbox_stats_populated(self, dataset_dir: str) -> None:
        stats = analyze_annotations(dataset_dir, "coco")
        assert len(stats.bbox_widths) == 43
        assert len(stats.bbox_heights) == 43
        assert all(0 < w <= 1 for w in stats.bbox_widths)
        assert all(0 < h <= 1 for h in stats.bbox_heights)

    def test_size_classification(self, dataset_dir: str) -> None:
        stats = analyze_annotations(dataset_dir, "coco")
        total = stats.small_count + stats.medium_count + stats.large_count
        assert total == 43

    def test_co_occurrence(self, dataset_dir: str) -> None:
        stats = analyze_annotations(dataset_dir, "coco")
        # Multiple classes appear in the same image → co-occurrence should exist
        assert len(stats.co_occurrence) > 0


class TestAnnotationsCLI:
    def test_annotations_command(self, dataset_dir: str) -> None:
        result = runner.invoke(app, ["annotations", dataset_dir])
        assert result.exit_code == 0
        assert "Annotation Analysis" in result.output
        assert "rbc" in result.output

    def test_annotations_with_format(self, dataset_dir: str) -> None:
        result = runner.invoke(app, ["annotations", dataset_dir, "--format", "coco"])
        assert result.exit_code == 0

    def test_annotations_json_output(self, dataset_dir: str, tmp_path: Path) -> None:
        out = str(tmp_path / "ann.json")
        result = runner.invoke(app, ["annotations", dataset_dir, "-o", out])
        assert result.exit_code == 0
        assert Path(out).exists()

        import json

        with open(out) as f:
            data = json.load(f)
        assert data["total_annotations"] == 43
        assert data["num_classes"] == 3


class TestScanAndPipeline:
    @pytest.mark.timeout(120)
    def test_scan_blood_cells(self, dataset_dir: str, tmp_path: Path) -> None:
        """Full scan of blood cell images."""
        manifest = str(tmp_path / "manifest.jsonl")
        result = runner.invoke(
            app,
            [
                "scan",
                str(FIXTURE_DIR / "images"),
                "-o",
                manifest,
                "--workers",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert Path(manifest).exists()

        # Verify manifest contents
        from imgeda.io.manifest_io import read_manifest

        meta, records = read_manifest(manifest)
        assert len(records) == 3
        for r in records:
            assert r.width == 416
            assert r.height == 416
            assert not r.is_corrupt

    @pytest.mark.timeout(120)
    def test_scan_then_report(self, dataset_dir: str, tmp_path: Path) -> None:
        """Scan + report generation on real images."""
        manifest = str(tmp_path / "manifest.jsonl")
        report = str(tmp_path / "report.html")

        # Scan
        result = runner.invoke(
            app,
            ["scan", str(FIXTURE_DIR / "images"), "-o", manifest, "--workers", "2"],
        )
        assert result.exit_code == 0

        # Report
        result = runner.invoke(app, ["report", "-m", manifest, "-o", report])
        assert result.exit_code == 0
        assert Path(report).exists()

        content = Path(report).read_text()
        assert "imgeda" in content.lower()
        assert "<img" in content  # Embedded plots

    @pytest.mark.timeout(120)
    def test_scan_then_all_plots(self, dataset_dir: str, tmp_path: Path) -> None:
        """Scan + all plots on real images."""
        manifest = str(tmp_path / "manifest.jsonl")
        plots_dir = str(tmp_path / "plots")

        # Scan
        result = runner.invoke(
            app,
            ["scan", str(FIXTURE_DIR / "images"), "-o", manifest, "--workers", "2"],
        )
        assert result.exit_code == 0

        # All plots
        result = runner.invoke(app, ["plot", "all", "-m", manifest, "-o", plots_dir])
        assert result.exit_code == 0
        plots = list(Path(plots_dir).glob("*.png"))
        assert len(plots) >= 11

    @pytest.mark.timeout(120)
    def test_scan_then_csv_export(self, dataset_dir: str, tmp_path: Path) -> None:
        """Scan + CSV export."""
        manifest = str(tmp_path / "manifest.jsonl")
        csv_out = str(tmp_path / "export.csv")

        result = runner.invoke(
            app,
            ["scan", str(FIXTURE_DIR / "images"), "-o", manifest, "--workers", "2"],
        )
        assert result.exit_code == 0

        result = runner.invoke(app, ["export", "csv", "-m", manifest, "-o", csv_out])
        assert result.exit_code == 0
        assert Path(csv_out).exists()

        import csv

        with open(csv_out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["format"] == "JPEG"


class TestAnnotationPlots:
    def test_all_annotation_plots(self, dataset_dir: str, tmp_path: Path) -> None:
        """Generate all annotation plots from real data."""
        from imgeda.models.config import PlotConfig
        from imgeda.plotting.annotations import (
            plot_annotation_coverage,
            plot_bbox_sizes,
            plot_class_frequency,
            plot_co_occurrence,
            plot_objects_per_image,
        )

        stats = analyze_annotations(dataset_dir, "coco")
        config = PlotConfig(output_dir=str(tmp_path))

        for fn in [
            plot_class_frequency,
            plot_bbox_sizes,
            plot_objects_per_image,
            plot_co_occurrence,
            plot_annotation_coverage,
        ]:
            path = fn(stats, config)
            assert Path(path).exists(), f"{fn.__name__} failed"
