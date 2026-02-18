"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from imgeda.cli.app import app
from imgeda.io.manifest_io import append_records, create_manifest
from imgeda.models.manifest import CornerStats, ImageRecord, ManifestMeta, PixelStats

runner = CliRunner()


class TestCLI:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "imgeda" in result.output.lower() or "image" in result.output.lower()

    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "imgeda" in result.output

    def test_scan_help(self) -> None:
        result = runner.invoke(app, ["scan", "--help"])
        assert result.exit_code == 0
        assert "directory" in result.output.lower() or "scan" in result.output.lower()

    @pytest.mark.timeout(60)
    def test_scan_command(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        output = str(tmp_path / "manifest.jsonl")
        result = runner.invoke(
            app,
            [
                "scan",
                str(tmp_image_dir),
                "-o",
                output,
                "--workers",
                "2",
                "--checkpoint-every",
                "5",
            ],
        )
        assert result.exit_code == 0
        assert Path(output).exists()

    def test_scan_invalid_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "scan",
                str(tmp_path / "nonexistent"),
                "-o",
                str(tmp_path / "out.jsonl"),
            ],
        )
        assert result.exit_code != 0

    def test_info_command(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)
        records = [
            ImageRecord(
                path="/data/a.jpg",
                filename="a.jpg",
                width=100,
                height=100,
                format="JPEG",
                color_mode="RGB",
                file_size_bytes=5000,
            ),
        ]
        append_records(str(manifest), records)

        result = runner.invoke(app, ["info", "-m", str(manifest)])
        assert result.exit_code == 0

    def test_plot_help(self) -> None:
        result = runner.invoke(app, ["plot", "--help"])
        assert result.exit_code == 0

    def test_check_help(self) -> None:
        result = runner.invoke(app, ["check", "--help"])
        assert result.exit_code == 0

    @pytest.mark.timeout(60)
    def test_report_command(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)
        records = [
            ImageRecord(
                path="/data/a.jpg",
                filename="a.jpg",
                width=100,
                height=100,
                format="JPEG",
                color_mode="RGB",
                file_size_bytes=5000,
                pixel_stats=PixelStats(
                    mean_r=120.0, mean_g=130.0, mean_b=140.0, mean_brightness=130.0
                ),
                corner_stats=CornerStats(corner_mean=100.0, center_mean=130.0, delta=30.0),
                phash="abcd1234",
            ),
        ]
        append_records(str(manifest), records)

        report_path = str(tmp_path / "report.html")
        result = runner.invoke(app, ["report", "-m", str(manifest), "-o", report_path])
        assert result.exit_code == 0
        assert Path(report_path).exists()
        content = Path(report_path).read_text()
        assert "imgeda" in content.lower()
        assert "<img" in content  # embedded plots


class TestPlotCLI:
    @pytest.fixture
    def manifest_for_plots(self, tmp_path: Path) -> tuple[str, str]:
        """Create a manifest with fully-populated records for plot testing.

        Returns (manifest_path, plots_output_dir).
        """
        manifest = tmp_path / "manifest.jsonl"
        plots_dir = str(tmp_path / "plots")
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)
        records = [
            ImageRecord(
                path=f"/data/img_{i}.jpg",
                filename=f"img_{i}.jpg",
                width=100 + i * 50,
                height=80 + i * 30,
                format="JPEG",
                color_mode="RGB",
                file_size_bytes=5000 + i * 1000,
                aspect_ratio=(100 + i * 50) / (80 + i * 30),
                pixel_stats=PixelStats(
                    mean_r=100.0 + i * 5,
                    mean_g=110.0 + i * 3,
                    mean_b=120.0 + i * 2,
                    mean_brightness=110.0 + i * 3,
                ),
                corner_stats=CornerStats(
                    corner_mean=80.0 + i,
                    center_mean=130.0 + i,
                    delta=50.0 + i,
                ),
                phash="abcd1234" if i < 3 else f"hash_{i:04x}",
            )
            for i in range(10)
        ]
        append_records(str(manifest), records)
        return str(manifest), plots_dir

    @pytest.fixture
    def empty_manifest(self, tmp_path: Path) -> str:
        """Create a manifest with metadata but no records."""
        manifest = tmp_path / "empty_manifest.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)
        return str(manifest)

    @pytest.mark.parametrize(
        "subcommand",
        [
            "dimensions",
            "file-size",
            "aspect-ratio",
            "brightness",
            "channels",
            "artifacts",
            "duplicates",
            "all",
        ],
    )
    def test_plot_subcommand(self, manifest_for_plots: tuple[str, str], subcommand: str) -> None:
        manifest, plots_dir = manifest_for_plots
        result = runner.invoke(app, ["plot", subcommand, "-m", manifest, "-o", plots_dir])
        assert result.exit_code == 0
        assert Path(plots_dir).exists()

    def test_plot_missing_manifest(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "plot",
                "dimensions",
                "-m",
                str(tmp_path / "nonexistent.jsonl"),
                "-o",
                str(tmp_path / "plots"),
            ],
        )
        assert result.exit_code != 0

    def test_plot_empty_manifest(self, empty_manifest: str, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "plot",
                "dimensions",
                "-m",
                empty_manifest,
                "-o",
                str(tmp_path / "plots"),
            ],
        )
        assert result.exit_code != 0


class TestCheckCommands:
    @pytest.fixture
    def manifest_with_issues(self, tmp_path: Path) -> str:
        manifest = tmp_path / "manifest.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)
        records = [
            ImageRecord(path="/data/good.jpg", filename="good.jpg", width=100, height=100),
            ImageRecord(path="/data/corrupt.jpg", filename="corrupt.jpg", is_corrupt=True),
            ImageRecord(
                path="/data/dark.jpg",
                filename="dark.jpg",
                is_dark=True,
                pixel_stats=PixelStats(mean_brightness=20.0),
            ),
            ImageRecord(
                path="/data/bright.jpg",
                filename="bright.jpg",
                is_overexposed=True,
                pixel_stats=PixelStats(mean_brightness=240.0),
            ),
            ImageRecord(
                path="/data/artifact.jpg",
                filename="artifact.jpg",
                has_border_artifact=True,
                corner_stats=CornerStats(delta=70.0),
            ),
        ]
        append_records(str(manifest), records)
        return str(manifest)

    def test_check_corrupt(self, manifest_with_issues: str) -> None:
        result = runner.invoke(app, ["check", "corrupt", "-m", manifest_with_issues])
        assert result.exit_code == 0
        assert "1" in result.output

    def test_check_exposure(self, manifest_with_issues: str) -> None:
        result = runner.invoke(app, ["check", "exposure", "-m", manifest_with_issues])
        assert result.exit_code == 0

    def test_check_artifacts(self, manifest_with_issues: str) -> None:
        result = runner.invoke(app, ["check", "artifacts", "-m", manifest_with_issues])
        assert result.exit_code == 0

    def test_check_all(self, manifest_with_issues: str) -> None:
        result = runner.invoke(app, ["check", "all", "-m", manifest_with_issues])
        assert result.exit_code == 0
        assert "Near-duplicate" in result.output
