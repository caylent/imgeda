"""Tests for pipeline runner and checkpoint."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from imgeda.io.manifest_io import append_records, create_manifest, read_manifest
from imgeda.models.config import ScanConfig
from imgeda.models.manifest import ManifestMeta
from imgeda.pipeline.checkpoint import filter_pending
from imgeda.pipeline.runner import run_scan
from imgeda.pipeline.signals import ShutdownHandler, worker_init


class TestPipeline:
    @pytest.mark.timeout(60)
    def test_scan_basic(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """Scan test images and verify manifest."""
        output = str(tmp_path / "manifest.jsonl")
        config = ScanConfig(workers=2, checkpoint_every=5)

        total, corrupt = run_scan(str(tmp_image_dir), output, config)

        assert total > 0
        meta, records = read_manifest(output)
        assert meta is not None
        assert len(records) == total
        assert corrupt >= 1  # we have one corrupt file

    @pytest.mark.timeout(60)
    def test_scan_resume_no_duplicates(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """Resume on a fully-scanned manifest must not duplicate records."""
        output = str(tmp_path / "manifest.jsonl")
        config = ScanConfig(workers=2, checkpoint_every=5)

        # First scan
        total1, _ = run_scan(str(tmp_image_dir), output, config)

        # Second scan (resume) — should find everything processed
        total2, _ = run_scan(str(tmp_image_dir), output, config)

        # Should not add duplicates
        _, records = read_manifest(output)
        assert len(records) == total1
        assert total2 == total1

    @pytest.mark.timeout(60)
    def test_scan_resume_preserves_records(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """Resume must preserve all previously-written records (no truncation)."""
        output = str(tmp_path / "manifest.jsonl")
        config = ScanConfig(workers=2, checkpoint_every=5)

        # Full scan
        run_scan(str(tmp_image_dir), output, config)
        _, records_before = read_manifest(output)
        paths_before = {r.path for r in records_before}

        # Resume run
        run_scan(str(tmp_image_dir), output, config)
        _, records_after = read_manifest(output)
        paths_after = {r.path for r in records_after}

        # Every record from the first run must still be present
        assert paths_before == paths_after

    @pytest.mark.timeout(60)
    def test_scan_resume_partial(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """Simulate a partial run: pre-populate manifest with some records, then resume."""
        from imgeda.io.image_reader import discover_images

        output = str(tmp_path / "manifest.jsonl")
        config = ScanConfig(workers=2, checkpoint_every=5)
        all_images = discover_images(str(tmp_image_dir), config.extensions)
        total_images = len(all_images)
        assert total_images > 3, "Need enough test images for a meaningful partial test"

        # Create a manifest with only the first 3 images processed
        meta = ManifestMeta(
            input_dir=os.path.abspath(str(tmp_image_dir)),
            total_files=total_images,
            created_at="2025-01-01T00:00:00+00:00",
        )
        create_manifest(output, meta)

        # Analyze first 3 images and write their records
        from imgeda.core.analyzer import analyze_image

        partial_records = [analyze_image(p, config) for p in all_images[:3]]
        append_records(output, partial_records)
        partial_paths = {r.path for r in partial_records}

        # Resume should process the remaining images
        total, _ = run_scan(str(tmp_image_dir), output, config)
        _, records = read_manifest(output)

        assert total == total_images
        assert len(records) == total_images

        # Verify previously-processed records are still present
        final_paths = {r.path for r in records}
        assert partial_paths.issubset(final_paths)

    @pytest.mark.timeout(60)
    def test_scan_force_truncates(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """--force must truncate and rescan from scratch."""
        output = str(tmp_path / "manifest.jsonl")

        # Normal scan first
        config = ScanConfig(workers=2, checkpoint_every=5)
        run_scan(str(tmp_image_dir), output, config)
        meta1, records1 = read_manifest(output)

        # Force rescan
        config_force = ScanConfig(workers=2, checkpoint_every=5, force=True, resume=False)
        run_scan(str(tmp_image_dir), output, config_force)
        meta2, records2 = read_manifest(output)

        # Force should have rescanned; same count but fresh metadata timestamp
        assert len(records1) > 0
        assert len(records2) == len(records1)
        assert meta1 is not None
        assert meta2 is not None
        assert meta2.created_at != meta1.created_at

    @pytest.mark.timeout(60)
    def test_scan_resume_nonexistent_manifest(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """Resume with no existing manifest should create a new one and scan all images."""
        output = str(tmp_path / "manifest.jsonl")
        assert not Path(output).exists()

        config = ScanConfig(workers=2, checkpoint_every=5, resume=True)
        total, _ = run_scan(str(tmp_image_dir), output, config)

        meta, records = read_manifest(output)
        assert meta is not None
        assert len(records) == total
        assert total > 0

    @pytest.mark.timeout(60)
    def test_scan_resume_updates_metadata(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """Resume must update the metadata header (e.g., total_files) without truncating."""
        output = str(tmp_path / "manifest.jsonl")
        config = ScanConfig(workers=2, checkpoint_every=5)

        # First scan
        run_scan(str(tmp_image_dir), output, config)
        meta1, records1 = read_manifest(output)
        assert meta1 is not None

        # Resume scan
        run_scan(str(tmp_image_dir), output, config)
        meta2, records2 = read_manifest(output)
        assert meta2 is not None

        # Records preserved, metadata refreshed
        assert len(records2) == len(records1)
        assert meta2.total_files == meta1.total_files

    @pytest.mark.timeout(60)
    def test_scan_metadata_only(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """Test skip pixel stats mode."""
        output = str(tmp_path / "manifest.jsonl")
        config = ScanConfig(workers=2, skip_pixel_stats=True, include_hashes=False)

        run_scan(str(tmp_image_dir), output, config)
        _, records = read_manifest(output)

        for r in records:
            if not r.is_corrupt:
                assert r.pixel_stats is None
                assert r.phash is None


class TestCheckpoint:
    def test_filter_pending(self, tmp_image_dir: Path) -> None:
        from imgeda.io.image_reader import discover_images

        images = discover_images(str(tmp_image_dir), ScanConfig().extensions)
        assert len(images) > 0

        # Mark first 3 as processed
        import os

        processed = set()
        for p in images[:3]:
            s = os.stat(p)
            processed.add((p, s.st_size, s.st_mtime))

        pending = filter_pending(images, processed)
        assert len(pending) == len(images) - 3


class TestSignalHandler:
    def test_initial_state(self) -> None:
        handler = ShutdownHandler()
        assert not handler.is_shutting_down

    def test_request_shutdown(self) -> None:
        handler = ShutdownHandler()
        handler.request_shutdown()
        assert handler.is_shutting_down

    def test_install_uninstall(self) -> None:
        handler = ShutdownHandler()
        handler.install()
        handler.uninstall()
        assert not handler.is_shutting_down

    def test_signal_sets_shutdown(self) -> None:
        handler = ShutdownHandler()
        handler.install()
        try:
            signal.raise_signal(signal.SIGINT)
        except KeyboardInterrupt:
            pass  # Should not happen on first signal
        assert handler.is_shutting_down
        handler.uninstall()

    def test_worker_init(self) -> None:
        original = signal.getsignal(signal.SIGINT)
        worker_init()
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        signal.signal(signal.SIGINT, original)  # restore
