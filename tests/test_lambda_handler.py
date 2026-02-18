"""Tests for Lambda handler router and individual action handlers.

Uses moto to provide a realistic mock S3 backend instead of blind MagicMock patching.
"""

from __future__ import annotations

import io
from typing import Any

import boto3
import numpy as np
import orjson
import pytest
from moto import mock_aws
from PIL import Image

from imgeda.lambda_handler.handler import handler
from imgeda.models.manifest import MANIFEST_META_KEY, ImageRecord, ManifestMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUCKET = "test-bucket"
OUTPUT_BUCKET = "output-bucket"


def _create_test_image_bytes(width: int = 100, height: int = 100, fmt: str = "JPEG") -> bytes:
    """Create a small test image and return its bytes."""
    arr = np.random.randint(60, 200, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_sample_records(count: int = 5) -> list[dict[str, Any]]:
    """Build a list of sample ImageRecord dicts."""
    records = []
    for i in range(count):
        rec = ImageRecord(
            path=f"s3://{BUCKET}/img_{i}.jpg",
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


def _build_manifest_body(records: list[dict[str, Any]], input_dir: str = "") -> bytes:
    """Build a JSONL manifest (meta header + records) as bytes."""
    meta = ManifestMeta(input_dir=input_dir or f"s3://{BUCKET}/images/", total_files=len(records))
    lines = [orjson.dumps(meta.to_dict())]
    for rec_dict in records:
        lines.append(orjson.dumps(rec_dict))
    return b"\n".join(lines) + b"\n"


def _upload_manifest(s3_client: Any, bucket: str, key: str, records: list[dict[str, Any]]) -> None:
    """Upload a JSONL manifest to mock S3."""
    body = _build_manifest_body(records)
    s3_client.put_object(Bucket=bucket, Key=key, Body=body)


def _s3_get_body(s3_client: Any, bucket: str, key: str) -> bytes:
    """Read and return the full body of an S3 object."""
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def s3_client() -> Any:
    """Return a boto3 S3 client connected to the moto mock."""
    return boto3.client("s3", region_name="us-east-1")


@pytest.fixture()
def setup_buckets(s3_client: Any) -> None:
    """Create the input and output buckets in mock S3."""
    s3_client.create_bucket(Bucket=BUCKET)
    s3_client.create_bucket(Bucket=OUTPUT_BUCKET)


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

    @mock_aws
    def test_routes_via_action_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When event has no 'action', fall back to ACTION env var (CDK path)."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.put_object(Bucket=BUCKET, Key="test.jpg", Body=b"data")

        monkeypatch.setenv("ACTION", "list_images")
        result = handler({"bucket": BUCKET}, None)
        assert result["total_images"] == 1

    def test_env_var_fallback_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ACTION env var should be lowercased for matching."""
        monkeypatch.setenv("ACTION", "NONEXISTENT")
        result = handler({}, None)
        assert result["statusCode"] == 400
        assert "nonexistent" in result["body"]

    @mock_aws
    def test_routes_to_list_images(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.put_object(Bucket=BUCKET, Key="img.jpg", Body=b"data")

        result = handler({"action": "list_images", "bucket": BUCKET}, None)
        assert result["total_images"] == 1
        assert len(result["batches"]) == 1

    @mock_aws
    def test_routes_to_analyze_batch(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.create_bucket(Bucket=OUTPUT_BUCKET)
        img_bytes = _create_test_image_bytes()
        s3.put_object(Bucket=BUCKET, Key="photo.jpg", Body=img_bytes)

        result = handler(
            {
                "action": "analyze_batch",
                "source_bucket": BUCKET,
                "keys": ["photo.jpg"],
                "output_bucket": OUTPUT_BUCKET,
                "output_key": "partials/batch_0.jsonl",
            },
            None,
        )
        assert "processed" in result
        assert result["processed"] == 1

    @mock_aws
    def test_routes_to_merge_manifests(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)
        records = _make_sample_records(2)
        part = b"\n".join(orjson.dumps(r) for r in records) + b"\n"
        s3.put_object(Bucket=OUTPUT_BUCKET, Key="partials/p.jsonl", Body=part)

        result = handler(
            {
                "action": "merge_manifests",
                "bucket": OUTPUT_BUCKET,
                "partial_keys": ["partials/p.jsonl"],
                "output_key": "manifest.jsonl",
            },
            None,
        )
        assert result["total_records"] == 2

    @mock_aws
    def test_routes_to_aggregate(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)
        records = _make_sample_records(3)
        _upload_manifest(s3, OUTPUT_BUCKET, "manifest.jsonl", records)

        result = handler(
            {
                "action": "aggregate",
                "bucket": OUTPUT_BUCKET,
                "manifest_key": "manifest.jsonl",
                "output_key": "summary.json",
            },
            None,
        )
        assert result["summary"]["total_images"] == 3

    @mock_aws
    def test_routes_to_generate_plots(self) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)
        records = _make_sample_records(3)
        _upload_manifest(s3, OUTPUT_BUCKET, "manifest.jsonl", records)

        result = handler(
            {
                "action": "generate_plots",
                "bucket": OUTPUT_BUCKET,
                "manifest_key": "manifest.jsonl",
                "output_prefix": "plots/",
            },
            None,
        )
        assert isinstance(result["plots"], list)


# ---------------------------------------------------------------------------
# list_images tests
# ---------------------------------------------------------------------------


class TestListImages:
    @mock_aws
    def test_list_images_basic(self) -> None:
        from imgeda.lambda_handler.handlers.list_images import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        # Upload a mix of image and non-image files
        s3.put_object(Bucket=BUCKET, Key="images/photo1.jpg", Body=b"jpg")
        s3.put_object(Bucket=BUCKET, Key="images/photo2.png", Body=b"png")
        s3.put_object(Bucket=BUCKET, Key="images/readme.txt", Body=b"txt")
        s3.put_object(Bucket=BUCKET, Key="images/photo3.jpeg", Body=b"jpeg")

        event = {"bucket": BUCKET, "prefix": "images/", "batch_size": 2}
        result = handle(event, None)

        assert result["total_images"] == 3  # .txt excluded
        assert len(result["batches"]) == 2
        assert len(result["batches"][0]) == 2
        assert len(result["batches"][1]) == 1

    @mock_aws
    def test_list_images_empty_bucket(self) -> None:
        from imgeda.lambda_handler.handlers.list_images import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        result = handle({"bucket": BUCKET}, None)
        assert result["total_images"] == 0
        assert result["batches"] == []

    @mock_aws
    def test_list_images_custom_extensions(self) -> None:
        from imgeda.lambda_handler.handlers.list_images import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        s3.put_object(Bucket=BUCKET, Key="a.tiff", Body=b"tiff")
        s3.put_object(Bucket=BUCKET, Key="b.jpg", Body=b"jpg")
        s3.put_object(Bucket=BUCKET, Key="c.png", Body=b"png")

        result = handle({"bucket": BUCKET, "extensions": [".tiff"]}, None)
        assert result["total_images"] == 1
        assert result["batches"][0] == ["a.tiff"]

    @mock_aws
    def test_list_images_pagination(self) -> None:
        """Verify list_images handles many objects (tests paginator path)."""
        from imgeda.lambda_handler.handlers.list_images import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)

        # Upload 25 images to force chunking with default batch_size=20
        for i in range(25):
            s3.put_object(Bucket=BUCKET, Key=f"img_{i:03d}.jpg", Body=b"data")

        result = handle({"bucket": BUCKET}, None)
        assert result["total_images"] == 25
        # Default batch_size is 20 -> 2 batches: [20, 5]
        assert len(result["batches"]) == 2
        assert len(result["batches"][0]) == 20
        assert len(result["batches"][1]) == 5


# ---------------------------------------------------------------------------
# analyze_batch tests
# ---------------------------------------------------------------------------


class TestAnalyzeBatch:
    @mock_aws
    def test_analyze_batch_basic(self) -> None:
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        # Upload real images
        for name in ["img1.jpg", "img2.jpg"]:
            s3.put_object(Bucket=BUCKET, Key=name, Body=_create_test_image_bytes())

        event = {
            "source_bucket": BUCKET,
            "keys": ["img1.jpg", "img2.jpg"],
            "output_bucket": OUTPUT_BUCKET,
            "output_key": "partials/batch_0.jsonl",
        }
        result = handle(event, None)

        assert result["processed"] == 2
        assert result["errors"] == 0
        assert result["output_key"] == "partials/batch_0.jsonl"

        # Verify JSONL was actually written to S3
        body = _s3_get_body(s3, OUTPUT_BUCKET, "partials/batch_0.jsonl")
        lines = [line for line in body.split(b"\n") if line.strip()]
        assert len(lines) == 2

        # Verify each line is valid JSON with expected fields
        for line in lines:
            rec = orjson.loads(line)
            assert "width" in rec
            assert "height" in rec
            assert rec["width"] == 100
            assert rec["height"] == 100
            assert rec["path"].startswith("s3://")

    @mock_aws
    def test_analyze_batch_corrupt_image(self) -> None:
        """Corrupt image data is still 'processed' (analyze_image never raises)."""
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        # Upload a non-image file — downloads OK but analyze_image flags is_corrupt
        s3.put_object(Bucket=BUCKET, Key="bad.jpg", Body=b"this is not an image")

        event = {
            "source_bucket": BUCKET,
            "keys": ["bad.jpg"],
            "output_bucket": OUTPUT_BUCKET,
            "output_key": "partials/batch_err.jsonl",
        }
        result = handle(event, None)

        # analyze_image never raises, so the record is "processed" with is_corrupt=True
        assert result["processed"] == 1
        assert result["errors"] == 0

        body = _s3_get_body(s3, OUTPUT_BUCKET, "partials/batch_err.jsonl")
        rec = orjson.loads(body.split(b"\n")[0])
        assert rec["is_corrupt"] is True

    @mock_aws
    def test_analyze_batch_with_config(self) -> None:
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        s3.put_object(Bucket=BUCKET, Key="img1.jpg", Body=_create_test_image_bytes())

        event = {
            "source_bucket": BUCKET,
            "keys": ["img1.jpg"],
            "output_bucket": OUTPUT_BUCKET,
            "output_key": "partials/batch_cfg.jsonl",
            "config": {"skip_pixel_stats": True, "include_hashes": False},
        }
        result = handle(event, None)
        assert result["processed"] == 1

        # Verify config was applied: no pixel_stats or hashes
        body = _s3_get_body(s3, OUTPUT_BUCKET, "partials/batch_cfg.jsonl")
        rec = orjson.loads(body.split(b"\n")[0])
        assert rec["pixel_stats"] is None
        assert rec["phash"] is None
        assert rec["dhash"] is None

    @mock_aws
    def test_analyze_batch_missing_key_in_s3(self) -> None:
        """Keys that don't exist in S3 should be counted as errors."""
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        event = {
            "source_bucket": BUCKET,
            "keys": ["nonexistent.jpg"],
            "output_bucket": OUTPUT_BUCKET,
            "output_key": "partials/batch_miss.jsonl",
        }
        result = handle(event, None)

        assert result["processed"] == 0
        assert result["errors"] == 1

    @mock_aws
    def test_analyze_batch_png_image(self) -> None:
        """Verify PNG images are handled correctly."""
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        s3.put_object(Bucket=BUCKET, Key="photo.png", Body=_create_test_image_bytes(fmt="PNG"))

        event = {
            "source_bucket": BUCKET,
            "keys": ["photo.png"],
            "output_bucket": OUTPUT_BUCKET,
            "output_key": "partials/batch_png.jsonl",
        }
        result = handle(event, None)
        assert result["processed"] == 1

        body = _s3_get_body(s3, OUTPUT_BUCKET, "partials/batch_png.jsonl")
        rec = orjson.loads(body.split(b"\n")[0])
        assert rec["format"] == "PNG"


# ---------------------------------------------------------------------------
# merge_manifests tests
# ---------------------------------------------------------------------------


class TestMergeManifests:
    @mock_aws
    def test_merge_basic(self) -> None:
        from imgeda.lambda_handler.handlers.merge_manifests import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        records = _make_sample_records(5)
        part1 = b"\n".join(orjson.dumps(r) for r in records[:3]) + b"\n"
        part2 = b"\n".join(orjson.dumps(r) for r in records[3:]) + b"\n"

        s3.put_object(Bucket=OUTPUT_BUCKET, Key="partials/batch_0.jsonl", Body=part1)
        s3.put_object(Bucket=OUTPUT_BUCKET, Key="partials/batch_1.jsonl", Body=part2)

        event = {
            "bucket": OUTPUT_BUCKET,
            "partial_keys": ["partials/batch_0.jsonl", "partials/batch_1.jsonl"],
            "output_key": "manifest.jsonl",
            "input_dir": "s3://input/images/",
        }
        result = handle(event, None)

        assert result["total_records"] == 5
        assert result["output_key"] == "manifest.jsonl"

        # Verify the merged manifest was written with a proper meta header
        body = _s3_get_body(s3, OUTPUT_BUCKET, "manifest.jsonl")
        lines = [line for line in body.split(b"\n") if line.strip()]
        assert len(lines) == 6  # 1 meta + 5 records

        meta = orjson.loads(lines[0])
        assert meta[MANIFEST_META_KEY] is True
        assert meta["total_files"] == 5
        assert meta["input_dir"] == "s3://input/images/"

    @mock_aws
    def test_merge_empty_partials(self) -> None:
        from imgeda.lambda_handler.handlers.merge_manifests import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        s3.put_object(Bucket=OUTPUT_BUCKET, Key="empty.jsonl", Body=b"")

        result = handle(
            {
                "bucket": OUTPUT_BUCKET,
                "partial_keys": ["empty.jsonl"],
                "output_key": "manifest.jsonl",
            },
            None,
        )
        assert result["total_records"] == 0

    @mock_aws
    def test_merge_skips_malformed_lines(self) -> None:
        """Malformed JSON lines should be skipped without crashing."""
        from imgeda.lambda_handler.handlers.merge_manifests import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        records = _make_sample_records(2)
        good_line = orjson.dumps(records[0])
        bad_line = b"this is not valid json {{{{"
        good_line2 = orjson.dumps(records[1])
        body = good_line + b"\n" + bad_line + b"\n" + good_line2 + b"\n"

        s3.put_object(Bucket=OUTPUT_BUCKET, Key="mixed.jsonl", Body=body)

        result = handle(
            {
                "bucket": OUTPUT_BUCKET,
                "partial_keys": ["mixed.jsonl"],
                "output_key": "manifest.jsonl",
            },
            None,
        )
        assert result["total_records"] == 2
        assert result["skipped_lines"] == 1

    @mock_aws
    def test_merge_strips_accidental_meta_from_partials(self) -> None:
        """Meta lines in partial files should be skipped during merge."""
        from imgeda.lambda_handler.handlers.merge_manifests import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        # Build a partial that accidentally has a meta line
        records = _make_sample_records(2)
        meta = ManifestMeta(input_dir="s3://old/", total_files=99)
        body = orjson.dumps(meta.to_dict()) + b"\n"
        body += b"\n".join(orjson.dumps(r) for r in records) + b"\n"

        s3.put_object(Bucket=OUTPUT_BUCKET, Key="partial_with_meta.jsonl", Body=body)

        result = handle(
            {
                "bucket": OUTPUT_BUCKET,
                "partial_keys": ["partial_with_meta.jsonl"],
                "output_key": "manifest.jsonl",
            },
            None,
        )
        # Only the 2 real records, not the stale meta
        assert result["total_records"] == 2

    @mock_aws
    def test_merge_accepts_analyze_results(self) -> None:
        """Step Functions Map output is an array of results; merge should extract output_keys."""
        from imgeda.lambda_handler.handlers.merge_manifests import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        records = _make_sample_records(4)
        part1 = b"\n".join(orjson.dumps(r) for r in records[:2]) + b"\n"
        part2 = b"\n".join(orjson.dumps(r) for r in records[2:]) + b"\n"

        s3.put_object(Bucket=OUTPUT_BUCKET, Key="partials/batch-0.jsonl", Body=part1)
        s3.put_object(Bucket=OUTPUT_BUCKET, Key="partials/batch-1.jsonl", Body=part2)

        # Simulate Step Functions Map output (no partial_keys, just analyze_results)
        event = {
            "bucket": OUTPUT_BUCKET,
            "analyze_results": [
                {"processed": 2, "errors": 0, "output_key": "partials/batch-0.jsonl"},
                {"processed": 2, "errors": 0, "output_key": "partials/batch-1.jsonl"},
            ],
            "output_key": "manifest.jsonl",
            "input_dir": "s3://input/images/",
        }
        result = handle(event, None)

        assert result["total_records"] == 4
        assert result["output_key"] == "manifest.jsonl"


# ---------------------------------------------------------------------------
# aggregate tests
# ---------------------------------------------------------------------------


class TestAggregate:
    @mock_aws
    def test_aggregate_basic(self) -> None:
        from imgeda.lambda_handler.handlers.aggregate import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        records = _make_sample_records(5)
        _upload_manifest(s3, OUTPUT_BUCKET, "manifest.jsonl", records)

        event = {
            "bucket": OUTPUT_BUCKET,
            "manifest_key": "manifest.jsonl",
            "output_key": "summary.json",
        }
        result = handle(event, None)

        assert result["output_key"] == "summary.json"
        summary = result["summary"]
        assert summary["total_images"] == 5
        assert summary["total_size_bytes"] == 15000  # 1000+2000+3000+4000+5000

        # Verify JSON was actually uploaded to S3
        body = _s3_get_body(s3, OUTPUT_BUCKET, "summary.json")
        uploaded_summary = orjson.loads(body)
        assert uploaded_summary["total_images"] == 5

    @mock_aws
    def test_aggregate_empty_manifest(self) -> None:
        from imgeda.lambda_handler.handlers.aggregate import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        meta = ManifestMeta(input_dir="", total_files=0)
        body = orjson.dumps(meta.to_dict()) + b"\n"
        s3.put_object(Bucket=OUTPUT_BUCKET, Key="empty.jsonl", Body=body)

        result = handle(
            {
                "bucket": OUTPUT_BUCKET,
                "manifest_key": "empty.jsonl",
                "output_key": "summary.json",
            },
            None,
        )
        assert result["summary"]["total_images"] == 0

    @mock_aws
    def test_aggregate_verifies_uploaded_json_format(self) -> None:
        """Verify the uploaded summary is valid indented JSON with correct content type."""
        from imgeda.lambda_handler.handlers.aggregate import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        records = _make_sample_records(3)
        _upload_manifest(s3, OUTPUT_BUCKET, "manifest.jsonl", records)

        handle(
            {
                "bucket": OUTPUT_BUCKET,
                "manifest_key": "manifest.jsonl",
                "output_key": "summary.json",
            },
            None,
        )

        body = _s3_get_body(s3, OUTPUT_BUCKET, "summary.json")
        # Should be valid JSON
        parsed = orjson.loads(body)
        assert parsed["total_images"] == 3
        # Should be indented (contains newlines)
        assert b"\n" in body


# ---------------------------------------------------------------------------
# generate_plots tests
# ---------------------------------------------------------------------------


class TestGeneratePlots:
    @mock_aws
    def test_generate_plots_basic(self) -> None:
        from imgeda.lambda_handler.handlers.generate_plots import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        records = _make_sample_records(5)
        _upload_manifest(s3, OUTPUT_BUCKET, "manifest.jsonl", records)

        event = {
            "bucket": OUTPUT_BUCKET,
            "manifest_key": "manifest.jsonl",
            "output_prefix": "plots/",
        }
        result = handle(event, None)

        assert isinstance(result["plots"], list)
        assert len(result["plots"]) > 0

        # Verify each plot was actually uploaded to S3
        for key in result["plots"]:
            assert key.startswith("plots/")
            body = _s3_get_body(s3, OUTPUT_BUCKET, key)
            # PNG files start with the PNG magic bytes
            assert body[:4] == b"\x89PNG", f"Expected PNG file at {key}"

    @mock_aws
    def test_generate_plots_empty_records(self) -> None:
        from imgeda.lambda_handler.handlers.generate_plots import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        meta = ManifestMeta(input_dir="", total_files=0)
        body = orjson.dumps(meta.to_dict()) + b"\n"
        s3.put_object(Bucket=OUTPUT_BUCKET, Key="empty.jsonl", Body=body)

        result = handle(
            {
                "bucket": OUTPUT_BUCKET,
                "manifest_key": "empty.jsonl",
                "output_prefix": "plots/",
            },
            None,
        )
        assert result["plots"] == []

    @mock_aws
    def test_generate_plots_custom_prefix(self) -> None:
        from imgeda.lambda_handler.handlers.generate_plots import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        records = _make_sample_records(3)
        _upload_manifest(s3, OUTPUT_BUCKET, "manifest.jsonl", records)

        result = handle(
            {
                "bucket": OUTPUT_BUCKET,
                "manifest_key": "manifest.jsonl",
                "output_prefix": "results/charts/",
            },
            None,
        )

        for key in result["plots"]:
            assert key.startswith("results/charts/")


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @mock_aws
    def test_missing_required_fields_list_images(self) -> None:
        """list_images should raise KeyError when bucket is missing."""
        from imgeda.lambda_handler.handlers.list_images import handle

        with pytest.raises(KeyError):
            handle({}, None)

    @mock_aws
    def test_missing_required_fields_analyze_batch(self) -> None:
        """analyze_batch should raise KeyError when required fields are missing."""
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        with pytest.raises(KeyError):
            handle({"action": "analyze_batch"}, None)

    @mock_aws
    def test_analyze_batch_nonexistent_source_bucket(self) -> None:
        """Download from a non-existent bucket counts as an error (caught internally)."""
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        # Source bucket doesn't exist: download_file raises ClientError,
        # caught by analyze_batch's per-key try/except -> counted as error
        result = handle(
            {
                "source_bucket": "no-such-bucket",
                "keys": ["img.jpg"],
                "output_bucket": OUTPUT_BUCKET,
                "output_key": "partials/batch.jsonl",
            },
            None,
        )
        assert result["processed"] == 0
        assert result["errors"] == 1

    @mock_aws
    def test_aggregate_nonexistent_manifest(self) -> None:
        """Aggregating from a non-existent manifest key should raise ClientError."""
        from botocore.exceptions import ClientError

        from imgeda.lambda_handler.handlers.aggregate import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        with pytest.raises(ClientError):
            handle(
                {
                    "bucket": OUTPUT_BUCKET,
                    "manifest_key": "nonexistent.jsonl",
                    "output_key": "summary.json",
                },
                None,
            )

    @mock_aws
    def test_merge_nonexistent_partial(self) -> None:
        """Merging a non-existent partial key should skip it gracefully."""
        from imgeda.lambda_handler.handlers.merge_manifests import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        # One real partial + one missing
        records = _make_sample_records(2)
        part = b"\n".join(orjson.dumps(r) for r in records) + b"\n"
        s3.put_object(Bucket=OUTPUT_BUCKET, Key="partials/good.jsonl", Body=part)

        result = handle(
            {
                "bucket": OUTPUT_BUCKET,
                "partial_keys": ["partials/good.jsonl", "partials/missing.jsonl"],
                "output_key": "manifest.jsonl",
            },
            None,
        )
        # Should process the good partial and skip the missing one
        assert result["total_records"] == 2
        assert result["skipped_keys"] == 1

    @mock_aws
    def test_analyze_batch_mix_of_good_and_corrupt(self) -> None:
        """Batch with valid + corrupt images: both are processed (core never raises)."""
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        good_bytes = _create_test_image_bytes()
        s3.put_object(Bucket=BUCKET, Key="good.jpg", Body=good_bytes)
        s3.put_object(Bucket=BUCKET, Key="corrupt.jpg", Body=b"not an image")

        result = handle(
            {
                "source_bucket": BUCKET,
                "keys": ["good.jpg", "corrupt.jpg"],
                "output_bucket": OUTPUT_BUCKET,
                "output_key": "partials/mixed.jsonl",
            },
            None,
        )
        # analyze_image never raises — corrupt files get is_corrupt=True in the record
        assert result["processed"] == 2
        assert result["errors"] == 0

        # Verify both records written, one flagged corrupt
        body = _s3_get_body(s3, OUTPUT_BUCKET, "partials/mixed.jsonl")
        lines = [line for line in body.split(b"\n") if line.strip()]
        assert len(lines) == 2
        records = [orjson.loads(line) for line in lines]
        corrupt_flags = [r.get("is_corrupt", False) for r in records]
        assert True in corrupt_flags  # at least one is corrupt

    @mock_aws
    def test_analyze_batch_missing_s3_key(self) -> None:
        """Keys that don't exist in S3 count as errors (download_file raises)."""
        from imgeda.lambda_handler.handlers.analyze_batch import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        # good.jpg exists, ghost.jpg does not
        good_bytes = _create_test_image_bytes()
        s3.put_object(Bucket=BUCKET, Key="good.jpg", Body=good_bytes)

        result = handle(
            {
                "source_bucket": BUCKET,
                "keys": ["good.jpg", "ghost.jpg"],
                "output_bucket": OUTPUT_BUCKET,
                "output_key": "partials/partial.jsonl",
            },
            None,
        )
        assert result["processed"] == 1
        assert result["errors"] == 1

    @mock_aws
    def test_merge_with_failed_batch_in_analyze_results(self) -> None:
        """Map output may include batches with errors=N but no output_key should be skipped."""
        from imgeda.lambda_handler.handlers.merge_manifests import handle

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=OUTPUT_BUCKET)

        records = _make_sample_records(3)
        part = b"\n".join(orjson.dumps(r) for r in records) + b"\n"
        s3.put_object(Bucket=OUTPUT_BUCKET, Key="partials/batch-0.jsonl", Body=part)

        result = handle(
            {
                "bucket": OUTPUT_BUCKET,
                "analyze_results": [
                    {"processed": 3, "errors": 0, "output_key": "partials/batch-0.jsonl"},
                    # batch-1 wrote a partial but had some errors
                    {"processed": 0, "errors": 5, "output_key": "partials/batch-1.jsonl"},
                ],
                "output_key": "manifest.jsonl",
            },
            None,
        )
        # Should get the 3 records from batch-0; batch-1's partial doesn't exist
        assert result["total_records"] == 3
        assert result["skipped_keys"] == 1
