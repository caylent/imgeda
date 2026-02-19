"""Tests for new CLI commands: blur check, leakage check, annotations, csv export, new plot commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from imgeda.cli.app import app
from imgeda.io.manifest_io import append_records, create_manifest
from imgeda.models.manifest import CornerStats, ImageRecord, ManifestMeta, PixelStats

runner = CliRunner()


class TestCheckBlurCLI:
    @pytest.fixture
    def manifest_with_blur(self, tmp_path: Path) -> str:
        manifest = tmp_path / "manifest.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)
        records = [
            ImageRecord(path="/data/sharp.jpg", filename="sharp.jpg", blur_score=500.0),
            ImageRecord(
                path="/data/blurry.jpg", filename="blurry.jpg", blur_score=30.0, is_blurry=True
            ),
        ]
        append_records(str(manifest), records)
        return str(manifest)

    def test_blur_check(self, manifest_with_blur: str) -> None:
        result = runner.invoke(app, ["check", "blur", "-m", manifest_with_blur])
        assert result.exit_code == 0
        assert "1" in result.output  # 1 blurry image

    def test_blur_check_with_output(self, manifest_with_blur: str, tmp_path: Path) -> None:
        out = str(tmp_path / "blurry.json")
        result = runner.invoke(app, ["check", "blur", "-m", manifest_with_blur, "-o", out])
        assert result.exit_code == 0
        assert Path(out).exists()


class TestCheckLeakageCLI:
    @pytest.fixture
    def two_manifests_with_leak(self, tmp_path: Path) -> tuple[str, str]:
        m1 = tmp_path / "train.jsonl"
        m2 = tmp_path / "val.jsonl"
        meta1 = ManifestMeta(input_dir="/train", created_at="now")
        meta2 = ManifestMeta(input_dir="/val", created_at="now")
        create_manifest(str(m1), meta1)
        create_manifest(str(m2), meta2)

        # Same phash in both splits = leakage
        train_records = [
            ImageRecord(path="/train/a.jpg", filename="a.jpg", phash="abcd1234abcd1234"),
            ImageRecord(path="/train/b.jpg", filename="b.jpg", phash="11112222"),
        ]
        val_records = [
            ImageRecord(path="/val/a_copy.jpg", filename="a_copy.jpg", phash="abcd1234abcd1234"),
            ImageRecord(path="/val/c.jpg", filename="c.jpg", phash="33334444"),
        ]
        append_records(str(m1), train_records)
        append_records(str(m2), val_records)
        return str(m1), str(m2)

    def test_leakage_detected(self, two_manifests_with_leak: tuple[str, str]) -> None:
        m1, m2 = two_manifests_with_leak
        result = runner.invoke(app, ["check", "leakage", "-m", m1, "-m", m2])
        assert result.exit_code == 0
        assert "leakage" in result.output.lower() or "leaked" in result.output.lower()

    def test_leakage_requires_two_manifests(self, tmp_path: Path) -> None:
        m = tmp_path / "m.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(m), meta)
        append_records(str(m), [ImageRecord(path="/a.jpg", filename="a.jpg")])

        result = runner.invoke(app, ["check", "leakage", "-m", str(m)])
        assert result.exit_code == 1


class TestAnnotationsCLI:
    @pytest.fixture
    def yolo_dataset(self, tmp_path: Path) -> str:
        ds = tmp_path / "dataset"
        ds.mkdir()
        label_dir = ds / "labels"
        label_dir.mkdir()
        (label_dir / "img_001.txt").write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.8 0.1 0.1\n")
        (label_dir / "img_002.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        # Create data.yaml for format detection
        (ds / "data.yaml").write_text("names: [cat, dog]\nnc: 2\n")
        return str(ds)

    def test_annotations_command(self, yolo_dataset: str) -> None:
        result = runner.invoke(app, ["annotations", yolo_dataset])
        assert result.exit_code == 0
        assert "Annotation Analysis" in result.output

    def test_annotations_with_format(self, yolo_dataset: str) -> None:
        result = runner.invoke(app, ["annotations", yolo_dataset, "--format", "yolo"])
        assert result.exit_code == 0

    def test_annotations_with_output(self, yolo_dataset: str, tmp_path: Path) -> None:
        out = str(tmp_path / "ann.json")
        result = runner.invoke(app, ["annotations", yolo_dataset, "-o", out])
        assert result.exit_code == 0
        assert Path(out).exists()

    def test_annotations_invalid_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["annotations", str(tmp_path / "nonexistent")])
        assert result.exit_code == 1

    def test_annotations_no_annotations_format(self, tmp_path: Path) -> None:
        """A flat directory has no annotations to analyze."""
        ds = tmp_path / "flat_dataset"
        ds.mkdir()
        # Create a dummy image so it's not empty
        (ds / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0")
        result = runner.invoke(app, ["annotations", str(ds), "--format", "flat"])
        assert result.exit_code == 0


class TestExportCsvCLI:
    @pytest.fixture
    def export_manifest(self, tmp_path: Path) -> str:
        manifest = tmp_path / "manifest.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)
        records = [
            ImageRecord(
                path=f"/data/img_{i}.jpg",
                filename=f"img_{i}.jpg",
                width=640,
                height=480,
                format="JPEG",
            )
            for i in range(5)
        ]
        append_records(str(manifest), records)
        return str(manifest)

    def test_csv_export(self, export_manifest: str, tmp_path: Path) -> None:
        out = str(tmp_path / "out.csv")
        result = runner.invoke(app, ["export", "csv", "-m", export_manifest, "-o", out])
        assert result.exit_code == 0
        assert Path(out).exists()
        content = Path(out).read_text()
        assert "path" in content
        assert "img_0" in content

    def test_csv_export_missing_manifest(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["export", "csv", "-m", str(tmp_path / "nope.jsonl"), "-o", str(tmp_path / "out.csv")],
        )
        assert result.exit_code == 1


class TestNewPlotCLI:
    @pytest.fixture
    def manifest_for_new_plots(self, tmp_path: Path) -> tuple[str, str]:
        manifest = tmp_path / "manifest.jsonl"
        plots_dir = str(tmp_path / "plots")
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)
        records = [
            ImageRecord(
                path=f"/data/img_{i}.jpg",
                filename=f"img_{i}.jpg",
                width=640,
                height=480,
                format="JPEG",
                file_size_bytes=5000 + i * 1000,
                aspect_ratio=640 / 480,
                blur_score=50.0 + i * 20,
                is_blurry=i < 2,
                camera_model="Canon EOS 5D",
                camera_make="Canon",
                focal_length_35mm=24 + i * 10,
                iso_speed=100 + i * 200,
                pixel_stats=PixelStats(mean_brightness=128.0),
                corner_stats=CornerStats(corner_mean=80.0, center_mean=130.0, delta=50.0),
                phash=f"hash_{i:08x}",
            )
            for i in range(10)
        ]
        append_records(str(manifest), records)
        return str(manifest), plots_dir

    @pytest.mark.parametrize("subcommand", ["blur", "exif-camera", "exif-focal", "exif-iso"])
    def test_new_plot_subcommands(
        self, manifest_for_new_plots: tuple[str, str], subcommand: str
    ) -> None:
        manifest, plots_dir = manifest_for_new_plots
        result = runner.invoke(app, ["plot", subcommand, "-m", manifest, "-o", plots_dir])
        assert result.exit_code == 0
        assert Path(plots_dir).exists()

    def test_all_plots_includes_new(self, manifest_for_new_plots: tuple[str, str]) -> None:
        """The 'all' subcommand should generate all 11 plots."""
        manifest, plots_dir = manifest_for_new_plots
        result = runner.invoke(app, ["plot", "all", "-m", manifest, "-o", plots_dir])
        assert result.exit_code == 0
        plots = list(Path(plots_dir).glob("*.png"))
        assert len(plots) >= 11


class TestCheckAllIncludesBlur:
    def test_all_checks_shows_blurry(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)
        records = [
            ImageRecord(path="/data/a.jpg", filename="a.jpg", is_blurry=True, blur_score=30.0),
            ImageRecord(path="/data/b.jpg", filename="b.jpg"),
        ]
        append_records(str(manifest), records)

        result = runner.invoke(app, ["check", "all", "-m", str(manifest)])
        assert result.exit_code == 0
        assert "Blurry" in result.output
