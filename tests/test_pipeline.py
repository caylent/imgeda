"""Tests for pipeline runner and checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from imgeda.io.manifest_io import read_manifest
from imgeda.models.config import ScanConfig
from imgeda.pipeline.checkpoint import filter_pending
from imgeda.pipeline.runner import run_scan


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
    def test_scan_resume(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """Test that resume skips already-processed images."""
        output = str(tmp_path / "manifest.jsonl")
        config = ScanConfig(workers=2, checkpoint_every=5)

        # First scan
        total1, _ = run_scan(str(tmp_image_dir), output, config)

        # Second scan (resume) — should find everything processed
        total2, _ = run_scan(str(tmp_image_dir), output, config)

        # Should not add duplicates
        _, records = read_manifest(output)
        assert len(records) == total1

    @pytest.mark.timeout(60)
    def test_scan_force(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """Test force rescan."""
        output = str(tmp_path / "manifest.jsonl")
        config = ScanConfig(workers=2, checkpoint_every=5, force=True, resume=False)

        run_scan(str(tmp_image_dir), output, config)
        _, records1 = read_manifest(output)

        run_scan(str(tmp_image_dir), output, config)
        _, records2 = read_manifest(output)

        # Force should have rescanned but result in same count
        assert len(records1) > 0
        assert len(records2) > 0

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
