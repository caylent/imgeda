"""Tests for CSV export."""

from __future__ import annotations

import csv
from pathlib import Path

from imgeda.io.csv_io import _flatten_record, records_to_csv
from imgeda.models.manifest import CornerStats, ImageRecord, PixelStats


class TestFlattenRecord:
    def test_basic_fields(self) -> None:
        rec = ImageRecord(
            path="/data/img.jpg",
            filename="img.jpg",
            width=640,
            height=480,
            format="JPEG",
        )
        flat = _flatten_record(rec)
        assert flat["path"] == "/data/img.jpg"
        assert flat["width"] == 640
        assert flat["format"] == "JPEG"

    def test_optional_fields_empty(self) -> None:
        rec = ImageRecord(path="/a.jpg", filename="a.jpg")
        flat = _flatten_record(rec)
        assert flat["camera_make"] == ""
        assert flat["blur_score"] == ""
        assert flat["focal_length_mm"] == ""

    def test_pixel_stats_included(self) -> None:
        rec = ImageRecord(
            path="/a.jpg",
            filename="a.jpg",
            pixel_stats=PixelStats(mean_r=120.0, mean_g=130.0, mean_b=140.0),
        )
        flat = _flatten_record(rec)
        assert flat["mean_r"] == 120.0
        assert flat["mean_g"] == 130.0

    def test_corner_stats_included(self) -> None:
        rec = ImageRecord(
            path="/a.jpg",
            filename="a.jpg",
            corner_stats=CornerStats(corner_mean=80.0, center_mean=130.0, delta=50.0),
        )
        flat = _flatten_record(rec)
        assert flat["corner_mean"] == 80.0
        assert flat["corner_delta"] == 50.0

    def test_blur_fields(self) -> None:
        rec = ImageRecord(path="/a.jpg", filename="a.jpg", blur_score=55.3, is_blurry=True)
        flat = _flatten_record(rec)
        assert flat["blur_score"] == 55.3
        assert flat["is_blurry"] is True


class TestRecordsToCsv:
    def test_writes_csv(self, tmp_path: Path) -> None:
        records = [
            ImageRecord(path="/a.jpg", filename="a.jpg", width=100, height=100, format="JPEG"),
            ImageRecord(path="/b.png", filename="b.png", width=200, height=200, format="PNG"),
        ]
        out = str(tmp_path / "out.csv")
        count = records_to_csv(records, out)
        assert count == 2
        assert Path(out).exists()

        # Verify CSV structure
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["path"] == "/a.jpg"
        assert rows[1]["format"] == "PNG"

    def test_empty_records(self, tmp_path: Path) -> None:
        out = str(tmp_path / "empty.csv")
        count = records_to_csv([], out)
        assert count == 0
        assert not Path(out).exists()

    def test_all_columns_present(self, tmp_path: Path) -> None:
        records = [
            ImageRecord(
                path="/a.jpg",
                filename="a.jpg",
                pixel_stats=PixelStats(mean_r=100.0),
                corner_stats=CornerStats(delta=30.0),
                blur_score=123.4,
            ),
        ]
        out = str(tmp_path / "full.csv")
        records_to_csv(records, out)

        with open(out) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        # Verify key columns exist
        assert "blur_score" in row
        assert "mean_r" in row
        assert "corner_delta" in row
        assert "camera_make" in row
