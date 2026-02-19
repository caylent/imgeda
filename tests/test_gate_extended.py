"""Tests for new gate policy rules (blur, artifact, width, height, format, aspect ratio)."""

from __future__ import annotations

from imgeda.core.gate import evaluate_policy
from imgeda.models.manifest import ImageRecord
from imgeda.models.policy import Policy


def _records(n: int = 200, **kwargs: object) -> list[ImageRecord]:
    """Create N simple records with optional overrides."""
    defaults: dict[str, object] = {
        "width": 640,
        "height": 480,
        "format": "JPEG",
        "aspect_ratio": 640 / 480,
    }
    defaults.update(kwargs)
    return [
        ImageRecord(path=f"/img_{i}.jpg", filename=f"img_{i}.jpg", **defaults) for i in range(n)
    ]


class TestMaxBlurryPct:
    def test_disabled_by_default(self) -> None:
        """max_blurry_pct=100 means the check is not run."""
        records = _records(10, is_blurry=True)
        policy = Policy(min_images_total=1, max_blurry_pct=100.0)
        result = evaluate_policy(records, policy)
        # No blurry check should be added
        check_names = [c.name for c in result.checks]
        assert "max_blurry_pct" not in check_names

    def test_enabled_and_passes(self) -> None:
        records = _records(100)  # is_blurry defaults to False
        policy = Policy(min_images_total=1, max_blurry_pct=5.0)
        result = evaluate_policy(records, policy)
        check = next(c for c in result.checks if c.name == "max_blurry_pct")
        assert check.passed
        assert check.observed == 0.0

    def test_enabled_and_fails(self) -> None:
        records = _records(10)
        for r in records[:3]:
            r.is_blurry = True
        policy = Policy(min_images_total=1, max_blurry_pct=10.0)
        result = evaluate_policy(records, policy)
        check = next(c for c in result.checks if c.name == "max_blurry_pct")
        assert not check.passed
        assert check.observed == 30.0


class TestMaxArtifactPct:
    def test_disabled_by_default(self) -> None:
        records = _records(10, has_border_artifact=True)
        policy = Policy(min_images_total=1, max_artifact_pct=100.0)
        result = evaluate_policy(records, policy)
        check_names = [c.name for c in result.checks]
        assert "max_artifact_pct" not in check_names

    def test_enabled_and_fails(self) -> None:
        records = _records(10)
        for r in records[:2]:
            r.has_border_artifact = True
        policy = Policy(min_images_total=1, max_artifact_pct=5.0)
        result = evaluate_policy(records, policy)
        check = next(c for c in result.checks if c.name == "max_artifact_pct")
        assert not check.passed


class TestMinWidth:
    def test_disabled_when_zero(self) -> None:
        policy = Policy(min_images_total=1, min_width=0)
        result = evaluate_policy(_records(5), policy)
        check_names = [c.name for c in result.checks]
        assert "min_width" not in check_names

    def test_passes(self) -> None:
        policy = Policy(min_images_total=1, min_width=100)
        result = evaluate_policy(_records(5, width=640), policy)
        check = next(c for c in result.checks if c.name == "min_width")
        assert check.passed

    def test_fails(self) -> None:
        records = _records(5, width=50)
        policy = Policy(min_images_total=1, min_width=100)
        result = evaluate_policy(records, policy)
        check = next(c for c in result.checks if c.name == "min_width")
        assert not check.passed


class TestMinHeight:
    def test_disabled_when_zero(self) -> None:
        policy = Policy(min_images_total=1, min_height=0)
        result = evaluate_policy(_records(5), policy)
        check_names = [c.name for c in result.checks]
        assert "min_height" not in check_names

    def test_fails(self) -> None:
        records = _records(5, height=50)
        policy = Policy(min_images_total=1, min_height=100)
        result = evaluate_policy(records, policy)
        check = next(c for c in result.checks if c.name == "min_height")
        assert not check.passed


class TestAllowedFormats:
    def test_disabled_when_empty(self) -> None:
        policy = Policy(min_images_total=1, allowed_formats=[])
        result = evaluate_policy(_records(5), policy)
        check_names = [c.name for c in result.checks]
        assert "allowed_formats" not in check_names

    def test_passes(self) -> None:
        policy = Policy(min_images_total=1, allowed_formats=["jpeg", "png"])
        result = evaluate_policy(_records(5, format="JPEG"), policy)
        check = next(c for c in result.checks if c.name == "allowed_formats")
        assert check.passed

    def test_fails(self) -> None:
        records = _records(5, format="TIFF")
        policy = Policy(min_images_total=1, allowed_formats=["jpeg", "png"])
        result = evaluate_policy(records, policy)
        check = next(c for c in result.checks if c.name == "allowed_formats")
        assert not check.passed


class TestMaxAspectRatio:
    def test_disabled_when_zero(self) -> None:
        policy = Policy(min_images_total=1, max_aspect_ratio=0.0)
        result = evaluate_policy(_records(5), policy)
        check_names = [c.name for c in result.checks]
        assert "max_aspect_ratio" not in check_names

    def test_passes(self) -> None:
        policy = Policy(min_images_total=1, max_aspect_ratio=2.0)
        result = evaluate_policy(_records(5, aspect_ratio=1.33), policy)
        check = next(c for c in result.checks if c.name == "max_aspect_ratio")
        assert check.passed

    def test_fails(self) -> None:
        records = _records(5, aspect_ratio=5.0)
        policy = Policy(min_images_total=1, max_aspect_ratio=2.0)
        result = evaluate_policy(records, policy)
        check = next(c for c in result.checks if c.name == "max_aspect_ratio")
        assert not check.passed
