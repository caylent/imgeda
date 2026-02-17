"""Tests for manifest I/O."""

from __future__ import annotations

from pathlib import Path

from imgeda.io.manifest_io import (
    append_records,
    build_resume_set,
    read_manifest,
    write_meta,
)
from imgeda.models.manifest import ImageRecord, ManifestMeta, PixelStats


class TestManifestIO:
    def test_write_and_read_meta(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        meta = ManifestMeta(input_dir="/data/images", total_files=100, created_at="2025-01-01")
        write_meta(str(path), meta)

        loaded_meta, records = read_manifest(str(path))
        assert loaded_meta is not None
        assert loaded_meta.input_dir == "/data/images"
        assert loaded_meta.total_files == 100
        assert records == []

    def test_append_and_read_records(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        write_meta(str(path), meta)

        records = [
            ImageRecord(
                path="/data/a.jpg",
                filename="a.jpg",
                width=100,
                height=100,
                file_size_bytes=1000,
                mtime=1.0,
            ),
            ImageRecord(
                path="/data/b.jpg",
                filename="b.jpg",
                width=200,
                height=200,
                file_size_bytes=2000,
                mtime=2.0,
            ),
        ]
        append_records(str(path), records)

        loaded_meta, loaded = read_manifest(str(path))
        assert loaded_meta is not None
        assert len(loaded) == 2
        assert loaded[0].path == "/data/a.jpg"
        assert loaded[1].width == 200

    def test_crash_tolerance(self, tmp_path: Path) -> None:
        """Truncated last line should be skipped."""
        path = tmp_path / "manifest.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        write_meta(str(path), meta)

        records = [
            ImageRecord(path="/data/a.jpg", filename="a.jpg", file_size_bytes=1000, mtime=1.0),
        ]
        append_records(str(path), records)

        # Append truncated JSON line
        with open(path, "ab") as f:
            f.write(b'{"path": "/data/b.jpg", "trunc')

        loaded_meta, loaded = read_manifest(str(path))
        assert len(loaded) == 1  # truncated line skipped

    def test_resume_set(self) -> None:
        records = [
            ImageRecord(path="/a.jpg", file_size_bytes=100, mtime=1.0),
            ImageRecord(path="/b.jpg", file_size_bytes=200, mtime=2.0),
        ]
        resume = build_resume_set(records)
        assert ("/a.jpg", 100, 1.0) in resume
        assert ("/b.jpg", 200, 2.0) in resume
        assert ("/c.jpg", 300, 3.0) not in resume

    def test_records_with_pixel_stats(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        write_meta(str(path), meta)

        records = [
            ImageRecord(
                path="/data/a.jpg",
                filename="a.jpg",
                pixel_stats=PixelStats(
                    mean_r=120.0, mean_g=130.0, mean_b=140.0, mean_brightness=130.0
                ),
            ),
        ]
        append_records(str(path), records)

        _, loaded = read_manifest(str(path))
        assert len(loaded) == 1
        assert loaded[0].pixel_stats is not None
        assert loaded[0].pixel_stats.mean_r == 120.0

    def test_empty_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.jsonl"
        meta, records = read_manifest(str(path))
        assert meta is None
        assert records == []
