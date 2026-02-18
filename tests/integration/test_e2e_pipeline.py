"""End-to-end integration tests exercising the full CLI pipeline in sequence.

Each step verifies its output feeds correctly into the next command,
mirroring a real user workflow: scan -> check -> plot -> report -> info
-> rescan -> diff -> gate -> export.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from imgeda.cli.app import app
from imgeda.io.manifest_io import read_manifest

runner = CliRunner()

# Total expected images: 10 normal + 1 dark + 1 bright + 1 artifact + 2 dup + 1 corrupt = 16
EXPECTED_IMAGE_COUNT = 16

# PNG magic bytes (first 8 bytes)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.mark.integration
class TestEndToEndPipeline:
    """Full pipeline exercised in strict order.

    Uses a single workspace fixture; each test method is named with a numeric
    prefix so pytest's default sort order matches the required execution sequence.
    """

    # ------------------------------------------------------------------
    # Step 1: scan
    # ------------------------------------------------------------------
    @pytest.mark.timeout(120)
    def test_step01_scan(self, e2e_workspace: SimpleNamespace) -> None:
        """Scan the image directory and produce a JSONL manifest."""
        result = runner.invoke(
            app,
            [
                "scan",
                str(e2e_workspace.image_dir),
                "-o",
                e2e_workspace.manifest,
                "--workers",
                "2",
                "--checkpoint-every",
                "5",
            ],
        )
        assert result.exit_code == 0, f"scan failed:\n{result.output}"
        assert Path(e2e_workspace.manifest).exists(), "manifest file not created"

        meta, records = read_manifest(e2e_workspace.manifest)
        assert meta is not None, "manifest missing metadata header"
        assert len(records) == EXPECTED_IMAGE_COUNT, (
            f"expected {EXPECTED_IMAGE_COUNT} records, got {len(records)}"
        )

    # ------------------------------------------------------------------
    # Step 2: check corrupt
    # ------------------------------------------------------------------
    @pytest.mark.timeout(30)
    def test_step02_check_corrupt(self, e2e_workspace: SimpleNamespace) -> None:
        """check corrupt should find exactly the corrupt.jpg file."""
        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(app, ["check", "corrupt", "-m", e2e_workspace.manifest])
        assert result.exit_code == 0, f"check corrupt failed:\n{result.output}"
        assert "Corrupt images:" in result.output
        # We created exactly 1 corrupt file
        assert "1 found" in result.output
        assert "corrupt.jpg" in result.output

    # ------------------------------------------------------------------
    # Step 3: check exposure
    # ------------------------------------------------------------------
    @pytest.mark.timeout(30)
    def test_step03_check_exposure(self, e2e_workspace: SimpleNamespace) -> None:
        """check exposure should find the dark and bright images."""
        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(app, ["check", "exposure", "-m", e2e_workspace.manifest])
        assert result.exit_code == 0, f"check exposure failed:\n{result.output}"
        assert "Exposure issues:" in result.output
        # At least 2 (1 dark + 1 bright); the output line says "N found"
        found_line = [ln for ln in result.output.splitlines() if "found" in ln]
        assert found_line, "no 'found' line in exposure output"
        count = int(found_line[0].split(":")[1].strip().split()[0])
        assert count >= 2, f"expected >=2 exposure issues, got {count}"

    # ------------------------------------------------------------------
    # Step 4: check artifacts
    # ------------------------------------------------------------------
    @pytest.mark.timeout(30)
    def test_step04_check_artifacts(self, e2e_workspace: SimpleNamespace) -> None:
        """check artifacts should find the artifact image."""
        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(app, ["check", "artifacts", "-m", e2e_workspace.manifest])
        assert result.exit_code == 0, f"check artifacts failed:\n{result.output}"
        assert "Border artifacts:" in result.output
        found_line = [ln for ln in result.output.splitlines() if "found" in ln]
        assert found_line, "no 'found' line in artifacts output"
        count = int(found_line[0].split(":")[1].strip().split()[0])
        assert count >= 1, f"expected >=1 artifact, got {count}"

    # ------------------------------------------------------------------
    # Step 5: check duplicates
    # ------------------------------------------------------------------
    @pytest.mark.timeout(30)
    def test_step05_check_duplicates(self, e2e_workspace: SimpleNamespace) -> None:
        """check duplicates should find the dup_a / dup_b pair."""
        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(app, ["check", "duplicates", "-m", e2e_workspace.manifest])
        assert result.exit_code == 0, f"check duplicates failed:\n{result.output}"
        assert "Duplicate groups:" in result.output
        found_line = [ln for ln in result.output.splitlines() if "found" in ln]
        assert found_line, "no 'found' line in duplicates output"
        count = int(found_line[0].split(":")[1].strip().split()[0])
        assert count >= 1, f"expected >=1 duplicate group, got {count}"

    # ------------------------------------------------------------------
    # Step 6: check all
    # ------------------------------------------------------------------
    @pytest.mark.timeout(30)
    def test_step06_check_all(self, e2e_workspace: SimpleNamespace) -> None:
        """check all should report all issue categories."""
        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(app, ["check", "all", "-m", e2e_workspace.manifest])
        assert result.exit_code == 0, f"check all failed:\n{result.output}"
        output = result.output
        assert "Corrupt:" in output
        assert "Dark:" in output
        assert "Overexposed:" in output
        assert "Border artifacts:" in output
        assert "Exact duplicates:" in output
        assert "Near-duplicate groups:" in output

    # ------------------------------------------------------------------
    # Step 7: plot all
    # ------------------------------------------------------------------
    @pytest.mark.timeout(60)
    def test_step07_plot_all(self, e2e_workspace: SimpleNamespace) -> None:
        """plot all should produce 7 PNG files with valid PNG headers."""
        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(
            app,
            [
                "plot",
                "all",
                "-m",
                e2e_workspace.manifest,
                "-o",
                e2e_workspace.plots_dir,
            ],
        )
        assert result.exit_code == 0, f"plot all failed:\n{result.output}"

        plots_dir = Path(e2e_workspace.plots_dir)
        assert plots_dir.is_dir(), "plots directory not created"

        png_files = sorted(plots_dir.glob("*.png"))
        assert len(png_files) == 7, (
            f"expected 7 PNG plots, found {len(png_files)}: {[p.name for p in png_files]}"
        )

        # Verify each file has valid PNG magic bytes
        for png in png_files:
            header = png.read_bytes()[:8]
            assert header == PNG_MAGIC, f"{png.name} does not start with PNG magic bytes"

    # ------------------------------------------------------------------
    # Step 8: report
    # ------------------------------------------------------------------
    @pytest.mark.timeout(60)
    def test_step08_report(self, e2e_workspace: SimpleNamespace) -> None:
        """report should produce an HTML file with embedded <img> tags."""
        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(
            app,
            [
                "report",
                "-m",
                e2e_workspace.manifest,
                "-o",
                e2e_workspace.report_html,
            ],
        )
        assert result.exit_code == 0, f"report failed:\n{result.output}"

        report_path = Path(e2e_workspace.report_html)
        assert report_path.exists(), "report HTML not created"

        html = report_path.read_text()
        assert "<img" in html, "report missing embedded <img> tags"
        assert "data:image/png;base64," in html, "report missing base64-encoded images"
        assert "imgeda" in html.lower(), "report missing 'imgeda' branding"
        # Verify it contains the image count
        assert str(EXPECTED_IMAGE_COUNT) in html, (
            f"report should contain image count {EXPECTED_IMAGE_COUNT}"
        )

    # ------------------------------------------------------------------
    # Step 9: info
    # ------------------------------------------------------------------
    @pytest.mark.timeout(30)
    def test_step09_info(self, e2e_workspace: SimpleNamespace) -> None:
        """info should display the correct total image count."""
        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(app, ["info", "-m", e2e_workspace.manifest])
        assert result.exit_code == 0, f"info failed:\n{result.output}"
        assert str(EXPECTED_IMAGE_COUNT) in result.output, (
            f"info output should contain '{EXPECTED_IMAGE_COUNT}'"
        )

    # ------------------------------------------------------------------
    # Step 10: modify images + rescan
    # ------------------------------------------------------------------
    @pytest.mark.timeout(120)
    def test_step10_rescan_after_modification(self, e2e_workspace: SimpleNamespace) -> None:
        """After adding new images and removing one, rescan into manifest_v2."""
        self._ensure_manifest(e2e_workspace)
        img_dir = e2e_workspace.image_dir

        # Add 2 new images
        rng = np.random.RandomState(99)
        for i in range(2):
            arr = rng.randint(60, 200, (120, 160, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_dir / f"added_{i:03d}.jpg")

        # Remove one existing image
        removed = img_dir / "normal_000.jpg"
        assert removed.exists(), "normal_000.jpg should exist before removal"
        removed.unlink()

        # Rescan into manifest_v2
        result = runner.invoke(
            app,
            [
                "scan",
                str(img_dir),
                "-o",
                e2e_workspace.manifest_v2,
                "--workers",
                "2",
                "--checkpoint-every",
                "5",
            ],
        )
        assert result.exit_code == 0, f"rescan failed:\n{result.output}"

        meta_v2, records_v2 = read_manifest(e2e_workspace.manifest_v2)
        assert meta_v2 is not None
        # Original 16, minus 1 removed, plus 2 added = 17
        expected_v2 = EXPECTED_IMAGE_COUNT - 1 + 2
        assert len(records_v2) == expected_v2, (
            f"expected {expected_v2} records in v2, got {len(records_v2)}"
        )

    # ------------------------------------------------------------------
    # Step 11: diff
    # ------------------------------------------------------------------
    @pytest.mark.timeout(30)
    def test_step11_diff(self, e2e_workspace: SimpleNamespace) -> None:
        """diff should detect added and removed images between manifests."""
        self._ensure_manifest(e2e_workspace)
        self._ensure_manifest_v2(e2e_workspace)

        result = runner.invoke(
            app,
            [
                "diff",
                "--old",
                e2e_workspace.manifest,
                "--new",
                e2e_workspace.manifest_v2,
                "-o",
                e2e_workspace.diff_json,
            ],
        )
        assert result.exit_code == 0, f"diff failed:\n{result.output}"
        assert "Added:" in result.output
        assert "Removed:" in result.output

        # Parse the JSON output for strong assertions
        diff_path = Path(e2e_workspace.diff_json)
        assert diff_path.exists(), "diff JSON output not created"
        diff_data = json.loads(diff_path.read_text())

        assert len(diff_data["added"]) == 2, f"expected 2 added, got {len(diff_data['added'])}"
        assert len(diff_data["removed"]) == 1, (
            f"expected 1 removed, got {len(diff_data['removed'])}"
        )
        assert diff_data["summary"]["total_old"] == EXPECTED_IMAGE_COUNT
        assert diff_data["summary"]["total_new"] == EXPECTED_IMAGE_COUNT - 1 + 2

    # ------------------------------------------------------------------
    # Step 12: gate (pass)
    # ------------------------------------------------------------------
    @pytest.mark.timeout(30)
    def test_step12_gate_pass(self, e2e_workspace: SimpleNamespace) -> None:
        """gate with lenient policy should exit 0 (PASSED)."""
        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(
            app,
            [
                "gate",
                "-m",
                e2e_workspace.manifest,
                "-p",
                e2e_workspace.gate_pass_policy,
                "-o",
                e2e_workspace.gate_json,
            ],
        )
        assert result.exit_code == 0, f"gate pass failed:\n{result.output}"
        assert "PASSED" in result.output

        gate_path = Path(e2e_workspace.gate_json)
        assert gate_path.exists(), "gate JSON output not created"
        gate_data = json.loads(gate_path.read_text())
        assert gate_data["passed"] is True
        assert gate_data["total_images"] == EXPECTED_IMAGE_COUNT

    # ------------------------------------------------------------------
    # Step 13: gate (fail)
    # ------------------------------------------------------------------
    @pytest.mark.timeout(30)
    def test_step13_gate_fail(self, e2e_workspace: SimpleNamespace) -> None:
        """gate with impossible policy should exit 2 (FAILED)."""
        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(
            app,
            [
                "gate",
                "-m",
                e2e_workspace.manifest,
                "-p",
                e2e_workspace.gate_fail_policy,
            ],
        )
        assert result.exit_code == 2, (
            f"gate should exit 2, got {result.exit_code}:\n{result.output}"
        )
        assert "FAILED" in result.output

    # ------------------------------------------------------------------
    # Step 14: export parquet
    # ------------------------------------------------------------------
    @pytest.mark.timeout(30)
    def test_step14_export_parquet(self, e2e_workspace: SimpleNamespace) -> None:
        """export parquet should produce a file whose row count matches the manifest."""
        pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")

        self._ensure_manifest(e2e_workspace)
        result = runner.invoke(
            app,
            [
                "export",
                "parquet",
                "-m",
                e2e_workspace.manifest,
                "-o",
                e2e_workspace.parquet_out,
            ],
        )
        assert result.exit_code == 0, f"export parquet failed:\n{result.output}"

        parquet_path = Path(e2e_workspace.parquet_out)
        assert parquet_path.exists(), "Parquet file not created"

        table = pq.read_table(str(parquet_path))
        assert table.num_rows == EXPECTED_IMAGE_COUNT, (
            f"Parquet row count {table.num_rows} != manifest count {EXPECTED_IMAGE_COUNT}"
        )
        # Verify essential columns exist
        col_names = set(table.column_names)
        for col in ("path", "filename", "width", "height", "is_corrupt"):
            assert col in col_names, f"missing column '{col}' in Parquet output"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_manifest(self, ws: SimpleNamespace) -> None:
        """If the primary manifest does not exist yet, run scan to create it."""
        if Path(ws.manifest).exists():
            return
        result = runner.invoke(
            app,
            [
                "scan",
                str(ws.image_dir),
                "-o",
                ws.manifest,
                "--workers",
                "2",
                "--checkpoint-every",
                "5",
            ],
        )
        assert result.exit_code == 0, f"helper scan failed:\n{result.output}"

    def _ensure_manifest_v2(self, ws: SimpleNamespace) -> None:
        """If manifest_v2 does not exist, run the modification + rescan."""
        if Path(ws.manifest_v2).exists():
            return
        img_dir = ws.image_dir
        rng = np.random.RandomState(99)
        for i in range(2):
            path = img_dir / f"added_{i:03d}.jpg"
            if not path.exists():
                arr = rng.randint(60, 200, (120, 160, 3), dtype=np.uint8)
                Image.fromarray(arr).save(path)
        removed = img_dir / "normal_000.jpg"
        if removed.exists():
            removed.unlink()
        result = runner.invoke(
            app,
            [
                "scan",
                str(img_dir),
                "-o",
                ws.manifest_v2,
                "--workers",
                "2",
                "--checkpoint-every",
                "5",
            ],
        )
        assert result.exit_code == 0, f"helper rescan failed:\n{result.output}"
