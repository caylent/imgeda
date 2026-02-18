"""Tests for Parquet export."""

from __future__ import annotations

from pathlib import Path

import pytest

from imgeda.models.manifest import CornerStats, ImageRecord, PixelStats


class TestParquetExport:
    @pytest.fixture()
    def sample_records(self) -> list[ImageRecord]:
        return [
            ImageRecord(
                path=f"/img_{i}.jpg",
                filename=f"img_{i}.jpg",
                file_size_bytes=1000 * (i + 1),
                width=640,
                height=480,
                format="JPEG",
                color_mode="RGB",
                num_channels=3,
                aspect_ratio=1.3333,
                pixel_stats=PixelStats(
                    mean_r=120.0 + i,
                    mean_g=130.0 + i,
                    mean_b=140.0 + i,
                    mean_brightness=130.0 + i,
                ),
                corner_stats=CornerStats(
                    corner_mean=100.0,
                    center_mean=130.0,
                    delta=30.0,
                ),
                phash=f"{i:016x}",
            )
            for i in range(20)
        ]

    def test_export_and_read_back(self, sample_records: list[ImageRecord], tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        from imgeda.io.parquet_io import records_to_parquet

        output = str(tmp_path / "test.parquet")
        row_count = records_to_parquet(sample_records, output)
        assert row_count == 20
        assert Path(output).exists()

        # Read back and validate
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        table = pq.read_table(output)
        assert table.num_rows == 20

        # Check key columns exist
        col_names = table.column_names
        assert "path" in col_names
        assert "width" in col_names
        assert "pixel_stats.mean_r" in col_names
        assert "corner_stats.delta" in col_names
        assert "is_corrupt" in col_names

        # Verify data
        paths = table.column("path").to_pylist()
        assert paths[0] == "/img_0.jpg"
        assert paths[19] == "/img_19.jpg"

    def test_export_empty_records(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        from imgeda.io.parquet_io import records_to_parquet

        output = str(tmp_path / "empty.parquet")
        row_count = records_to_parquet([], output)
        assert row_count == 0
        assert Path(output).exists()

    def test_export_without_optional_fields(self, tmp_path: Path) -> None:
        """Records without pixel_stats or corner_stats should export with zeros."""
        pytest.importorskip("pyarrow")
        from imgeda.io.parquet_io import records_to_parquet

        records = [
            ImageRecord(path="/a.jpg", filename="a.jpg", width=100, height=100),
        ]
        output = str(tmp_path / "no_stats.parquet")
        row_count = records_to_parquet(records, output)
        assert row_count == 1

        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        table = pq.read_table(output)
        assert table.column("pixel_stats.mean_r").to_pylist() == [0.0]
        assert table.column("corner_stats.delta").to_pylist() == [0.0]

    def test_flatten_record(self) -> None:
        from imgeda.io.parquet_io import _flatten_record

        rec = ImageRecord(
            path="/test.jpg",
            filename="test.jpg",
            pixel_stats=PixelStats(mean_r=100.0, mean_brightness=120.0),
            corner_stats=CornerStats(delta=45.0),
        )
        flat = _flatten_record(rec)
        assert flat["path"] == "/test.jpg"
        assert flat["pixel_stats.mean_r"] == 100.0
        assert flat["pixel_stats.mean_brightness"] == 120.0
        assert flat["corner_stats.delta"] == 45.0
