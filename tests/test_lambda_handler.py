"""Tests for Lambda handler router and individual action handlers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import orjson
import pytest
from PIL import Image

from imgeda.lambda_handler.handler import handler
from imgeda.models.manifest import MANIFEST_META_KEY, ImageRecord, ManifestMeta

# Inject a mock boto3 into sys.modules so lazy imports inside handlers resolve
_mock_boto3 = MagicMock()
sys.modules.setdefault("boto3", _mock_boto3)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_boto3_mock() -> None:
    """Reset the shared boto3 mock before each test."""
    _mock_boto3.reset_mock()


@pytest.fixture()
def mock_s3() -> MagicMock:
    """Return a fresh mock S3 client wired into the boto3 mock."""
    client = MagicMock()
    _mock_boto3.client.return_value = client
    return client


@pytest.fixture()
def test_image_path(tmp_path: Path) -> str:
    """Create a single valid test image and return its path."""
    arr = np.random.randint(60, 200, (100, 100, 3), dtype=np.uint8)
    path = tmp_path / "test.jpg"
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture()
def sample_records() -> list[dict[str, Any]]:
    """Return a list of sample ImageRecord dicts."""
    records = []
    for i in range(5):
        rec = ImageRecord(
            path=f"s3://bucket/img_{i}.jpg",
            filename=f"img_{i}.jpg",
            file_size_bytes=1000 * (i + 1),
            width=640,
            height=480,
            format="JPEG",
            color_mode="RGB",
            num_channels=3,
            aspect_ratio=1.3333,
        )
        records.append(rec.to_dict())
    return records


@pytest.fixture()
def manifest_body(sample_records: list[dict[str, Any]]) -> bytes:
    """Build a JSONL manifest (meta + records) as bytes."""
    meta = ManifestMeta(input_dir="s3://bucket/images/", total_files=len(sample_records))
    lines = [orjson.dumps(meta.to_dict())]
    for rec_dict in sample_records:
        lines.append(orjson.dumps(rec_dict))
    return b"\n".join(lines) + b"\n"


def _s3_body(data: bytes) -> dict[str, Any]:
    """Wrap bytes into the shape returned by s3.get_object."""
    body_mock = MagicMock()
    body_mock.read.return_value = data
    return {"Body": body_mock}


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


class TestHandlerRouter:
    def test_unknown_action_returns_400(self) -> None:
        result = handler({"action": "nonexistent"}, None)
        assert result["statusCode"] == 400
        assert "nonexistent" in result["body"]

    def test_missing_action_returns_400(self) -> None:
        result = handler({}, None)
        assert result["statusCode"] == 400

    @patch("imgeda.lambda_handler.handlers.list_images.handle")
    def test_routes_to_list_images(self, mock_handle: MagicMock) -> None:
        mock_handle.return_value = {"batches": [], "total_images": 0}
        result = handler({"action": "list_images", "bucket": "b"}, None)
        mock_handle.assert_called_once()
        assert result["total_images"] == 0

    @patch("imgeda.lambda_handler.handlers.analyze_batch.handle")
    def test_routes_to_analyze_batch(self, mock_handle: MagicMock) -> None:
        mock_handle.return_value = {"processed": 0, "errors": 0, "output_key": "k"}
        result = handler({"action": "analyze_batch"}, None)
        mock_handle.assert_called_once()
        assert "processed" in result

    @patch("imgeda.lambda_handler.handlers.merge_manifests.handle")
    def test_routes_to_merge_manifests(self, mock_handle: MagicMock) -> None:
        mock_handle.return_value = {"total_records": 0, "output_key": "k"}
        handler({"action": "merge_manifests"}, None)
        mock_handle.assert_called_once()

    @patch("imgeda.lambda_handler.handlers.aggregate.handle")
    def test_routes_to_aggregate(self, mock_handle: MagicMock) -> None:
        mock_handle.return_value = {"output_key": "k", "summary": {}}
        handler({"action": "aggregate"}, None)
        mock_handle.assert_called_once()

    @patch("imgeda.lambda_handler.handlers.generate_plots.handle")
    def test_routes_to_generate_plots(self, mock_handle: MagicMock) -> None:
        mock_handle.return_value = {"plots": []}
        handler({"action": "generate_plots"}, None)
        mock_handle.assert_called_once()


# ---------------------------------------------------------------------------
# list_images tests
# ---------------------------------------------------------------------------


class TestListImages:
    def test_list_images_basic(self, mock_s3: MagicMock) -> None:
        from imgeda.lambda_handler.handlers.list_images import handle

        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "images/photo1.jpg"},
                    {"Key": "images/photo2.png"},
                    {"Key": "images/readme.txt"},
                    {"Key": "images/photo3.jpeg"},
                ]
            }
        ]

        event = {"bucket": "my-bucket", "prefix": "images/", "batch_size": 2}
        result = handle(event, None)

        assert result["total_images"] == 3  # .txt excluded
        assert len(result["batches"]) == 2
        assert len(result["batches"][0]) == 2
        assert len(result["batches"][1]) == 1

    def test_list_images_empty_bucket(self, mock_s3: MagicMock) -> None:
        from imgeda.lambda_handler.handlers.list_images import handle

        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        result = handle({"bucket": "empty"}, None)
        assert result["total_images"] == 0
        assert result["batches"] == []

    def test_list_images_custom_extensions(self, mock_s3: MagicMock) -> None:
        from imgeda.lambda_handler.handlers.list_images import handle

        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "a.tiff"},
                    {"Key": "b.jpg"},
                    {"Key": "c.png"},
                ]
            }
        ]

        result = handle({"bucket": "b", "extensions": [".tiff"]}, None)
        assert result["total_images"] == 1


# ---------------------------------------------------------------------------
# analyze_batch tests
# ---------------------------------------------------------------------------


class TestAnalyzeBatch:
    def test_analyze_batch_basic(self, mock_s3: MagicMock, test_image_path: str) -> None:
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        def fake_download(bucket: str, key: str, path: str) -> None:
            import shutil

            shutil.copy2(test_image_path, path)

        mock_s3.download_file.side_effect = fake_download

        event = {
            "source_bucket": "input",
            "keys": ["img1.jpg", "img2.jpg"],
            "output_bucket": "output",
            "output_key": "partials/batch_0.jsonl",
        }
        result = handle(event, None)

        assert result["processed"] == 2
        assert result["errors"] == 0
        assert result["output_key"] == "partials/batch_0.jsonl"
        mock_s3.put_object.assert_called_once()

        # Verify the JSONL content
        call_kwargs = mock_s3.put_object.call_args[1]
        body = call_kwargs["Body"]
        lines = [line for line in body.split(b"\n") if line.strip()]
        assert len(lines) == 2

    def test_analyze_batch_with_errors(self, mock_s3: MagicMock) -> None:
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        mock_s3.download_file.side_effect = Exception("download failed")

        event = {
            "source_bucket": "input",
            "keys": ["bad.jpg"],
            "output_bucket": "output",
            "output_key": "partials/batch_err.jsonl",
        }
        result = handle(event, None)

        assert result["processed"] == 0
        assert result["errors"] == 1

    def test_analyze_batch_with_config(self, mock_s3: MagicMock, test_image_path: str) -> None:
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        def fake_download(bucket: str, key: str, path: str) -> None:
            import shutil

            shutil.copy2(test_image_path, path)

        mock_s3.download_file.side_effect = fake_download

        event = {
            "source_bucket": "input",
            "keys": ["img1.jpg"],
            "output_bucket": "output",
            "output_key": "partials/batch_cfg.jsonl",
            "config": {"skip_pixel_stats": True, "include_hashes": False},
        }
        result = handle(event, None)
        assert result["processed"] == 1


# ---------------------------------------------------------------------------
# merge_manifests tests
# ---------------------------------------------------------------------------


class TestMergeManifests:
    def test_merge_basic(self, mock_s3: MagicMock, sample_records: list[dict[str, Any]]) -> None:
        from imgeda.lambda_handler.handlers.merge_manifests import handle

        # Split records into two partial files
        part1 = b"\n".join(orjson.dumps(r) for r in sample_records[:3]) + b"\n"
        part2 = b"\n".join(orjson.dumps(r) for r in sample_records[3:]) + b"\n"

        mock_s3.get_object.side_effect = [_s3_body(part1), _s3_body(part2)]

        event = {
            "bucket": "output",
            "partial_keys": ["partials/batch_0.jsonl", "partials/batch_1.jsonl"],
            "output_key": "manifest.jsonl",
            "input_dir": "s3://input/images/",
        }
        result = handle(event, None)

        assert result["total_records"] == 5
        assert result["output_key"] == "manifest.jsonl"

        # Verify uploaded manifest has meta header
        call_kwargs = mock_s3.put_object.call_args[1]
        body = call_kwargs["Body"]
        first_line = body.split(b"\n")[0]
        meta = orjson.loads(first_line)
        assert meta[MANIFEST_META_KEY] is True
        assert meta["total_files"] == 5

    def test_merge_empty_partials(self, mock_s3: MagicMock) -> None:
        from imgeda.lambda_handler.handlers.merge_manifests import handle

        mock_s3.get_object.return_value = _s3_body(b"")

        result = handle(
            {
                "bucket": "output",
                "partial_keys": ["empty.jsonl"],
                "output_key": "manifest.jsonl",
            },
            None,
        )
        assert result["total_records"] == 0


# ---------------------------------------------------------------------------
# aggregate tests
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_aggregate_basic(self, mock_s3: MagicMock, manifest_body: bytes) -> None:
        from imgeda.lambda_handler.handlers.aggregate import handle

        mock_s3.get_object.return_value = _s3_body(manifest_body)

        event = {
            "bucket": "output",
            "manifest_key": "manifest.jsonl",
            "output_key": "summary.json",
        }
        result = handle(event, None)

        assert result["output_key"] == "summary.json"
        summary = result["summary"]
        assert summary["total_images"] == 5
        assert summary["total_size_bytes"] == 15000

        # Verify JSON was uploaded
        mock_s3.put_object.assert_called_once()

    def test_aggregate_empty_manifest(self, mock_s3: MagicMock) -> None:
        from imgeda.lambda_handler.handlers.aggregate import handle

        meta = ManifestMeta(input_dir="", total_files=0)
        body = orjson.dumps(meta.to_dict()) + b"\n"
        mock_s3.get_object.return_value = _s3_body(body)

        result = handle(
            {
                "bucket": "output",
                "manifest_key": "empty.jsonl",
                "output_key": "summary.json",
            },
            None,
        )
        assert result["summary"]["total_images"] == 0


# ---------------------------------------------------------------------------
# generate_plots tests
# ---------------------------------------------------------------------------


class TestGeneratePlots:
    def test_generate_plots_basic(self, mock_s3: MagicMock, manifest_body: bytes) -> None:
        from imgeda.lambda_handler.handlers.generate_plots import handle

        mock_s3.get_object.return_value = _s3_body(manifest_body)

        event = {
            "bucket": "output",
            "manifest_key": "manifest.jsonl",
            "output_prefix": "plots/",
        }
        result = handle(event, None)

        assert isinstance(result["plots"], list)
        # At least some plots should be generated
        assert len(result["plots"]) > 0
        # Verify upload_file was called
        assert mock_s3.upload_file.call_count == len(result["plots"])

    def test_generate_plots_empty_records(self, mock_s3: MagicMock) -> None:
        from imgeda.lambda_handler.handlers.generate_plots import handle

        meta = ManifestMeta(input_dir="", total_files=0)
        body = orjson.dumps(meta.to_dict()) + b"\n"
        mock_s3.get_object.return_value = _s3_body(body)

        result = handle(
            {
                "bucket": "output",
                "manifest_key": "empty.jsonl",
                "output_prefix": "plots/",
            },
            None,
        )
        assert result["plots"] == []


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_required_fields_list_images(self, mock_s3: MagicMock) -> None:
        """list_images should raise KeyError when bucket is missing."""
        from imgeda.lambda_handler.handlers.list_images import handle

        with pytest.raises(KeyError):
            handle({}, None)

    def test_missing_required_fields_analyze_batch(self, mock_s3: MagicMock) -> None:
        """analyze_batch should raise KeyError when required fields are missing."""
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        with pytest.raises(KeyError):
            handle({"action": "analyze_batch"}, None)
