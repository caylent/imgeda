"""Integration tests for scan command flag combinations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from imgeda.cli.app import app
from imgeda.io.manifest_io import read_manifest

runner = CliRunner()


@pytest.mark.integration
class TestScanOptions:
    """Test various scan flag combinations against real image directories."""

    @pytest.mark.timeout(120)
    def test_skip_pixel_stats(self, e2e_workspace: SimpleNamespace) -> None:
        """--skip-pixel-stats should produce records with no pixel_stats."""
        manifest = str(Path(e2e_workspace.manifest).parent / "skip_pixel.jsonl")
        result = runner.invoke(
            app,
            [
                "scan",
                str(e2e_workspace.image_dir),
                "-o",
                manifest,
                "--workers",
                "2",
                "--skip-pixel-stats",
            ],
        )
        assert result.exit_code == 0, f"scan --skip-pixel-stats failed:\n{result.output}"

        meta, records = read_manifest(manifest)
        assert meta is not None
        assert len(records) > 0

        non_corrupt = [r for r in records if not r.is_corrupt]
        assert len(non_corrupt) > 0, "should have at least one non-corrupt record"
        for rec in non_corrupt:
            assert rec.pixel_stats is None, (
                f"record {rec.filename} should have no pixel_stats with --skip-pixel-stats"
            )

    @pytest.mark.timeout(120)
    def test_no_hashes(self, e2e_workspace: SimpleNamespace) -> None:
        """--no-hashes should produce records with no phash or dhash."""
        manifest = str(Path(e2e_workspace.manifest).parent / "no_hashes.jsonl")
        result = runner.invoke(
            app,
            [
                "scan",
                str(e2e_workspace.image_dir),
                "-o",
                manifest,
                "--workers",
                "2",
                "--no-hashes",
            ],
        )
        assert result.exit_code == 0, f"scan --no-hashes failed:\n{result.output}"

        meta, records = read_manifest(manifest)
        assert meta is not None
        assert len(records) > 0

        for rec in records:
            assert rec.phash is None or rec.phash == "", (
                f"record {rec.filename} should have no phash with --no-hashes, got '{rec.phash}'"
            )
            assert rec.dhash is None or rec.dhash == "", (
                f"record {rec.filename} should have no dhash with --no-hashes, got '{rec.dhash}'"
            )

    @pytest.mark.timeout(120)
    def test_extensions_filter(self, e2e_workspace: SimpleNamespace) -> None:
        """--extensions .png should scan only PNG files."""
        manifest = str(Path(e2e_workspace.manifest).parent / "png_only.jsonl")
        result = runner.invoke(
            app,
            [
                "scan",
                str(e2e_workspace.image_dir),
                "-o",
                manifest,
                "--workers",
                "2",
                "--extensions",
                ".png",
            ],
        )
        assert result.exit_code == 0, f"scan --extensions .png failed:\n{result.output}"

        meta, records = read_manifest(manifest)
        assert meta is not None
        # The fixture creates 3 PNG files: dark_001.png, bright_001.png, artifact_001.png
        assert len(records) == 3, (
            f"expected 3 PNG-only records, got {len(records)}: {[r.filename for r in records]}"
        )
        for rec in records:
            assert rec.filename.endswith(".png"), f"expected only .png files, got {rec.filename}"

    @pytest.mark.timeout(120)
    def test_force_rescan(self, e2e_workspace: SimpleNamespace) -> None:
        """--force should overwrite an existing manifest from scratch."""
        manifest = str(Path(e2e_workspace.manifest).parent / "force_test.jsonl")

        # First scan
        result1 = runner.invoke(
            app,
            [
                "scan",
                str(e2e_workspace.image_dir),
                "-o",
                manifest,
                "--workers",
                "2",
            ],
        )
        assert result1.exit_code == 0

        _, records1 = read_manifest(manifest)
        first_count = len(records1)
        assert first_count > 0

        # Force rescan: should produce same count (no duplicates from appending)
        result2 = runner.invoke(
            app,
            [
                "scan",
                str(e2e_workspace.image_dir),
                "-o",
                manifest,
                "--workers",
                "2",
                "--force",
            ],
        )
        assert result2.exit_code == 0

        _, records2 = read_manifest(manifest)
        assert len(records2) == first_count, (
            f"--force rescan should produce {first_count} records, got {len(records2)} "
            "(possible duplicate entries from append without truncation)"
        )
