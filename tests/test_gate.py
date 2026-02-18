"""Tests for quality gate evaluation."""

from __future__ import annotations

from pathlib import Path

from imgeda.core.gate import evaluate_policy, load_policy
from imgeda.models.manifest import ImageRecord
from imgeda.models.policy import Policy


class TestEvaluatePolicy:
    def test_passing_gate(self) -> None:
        records = [ImageRecord(path=f"/img_{i}.jpg", filename=f"img_{i}.jpg") for i in range(200)]
        policy = Policy(min_images_total=100, max_corrupt_pct=1.0)
        result = evaluate_policy(records, policy)
        assert result.passed
        assert result.total_images == 200
        assert all(c.passed for c in result.checks)

    def test_failing_min_images(self) -> None:
        records = [ImageRecord(path="/a.jpg", filename="a.jpg")]
        policy = Policy(min_images_total=100)
        result = evaluate_policy(records, policy)
        assert not result.passed
        min_check = next(c for c in result.checks if c.name == "min_images_total")
        assert not min_check.passed
        assert min_check.observed == 1.0

    def test_failing_corrupt_pct(self) -> None:
        records = [
            ImageRecord(path="/good.jpg", filename="good.jpg"),
            ImageRecord(path="/bad.jpg", filename="bad.jpg", is_corrupt=True),
        ]
        policy = Policy(min_images_total=1, max_corrupt_pct=1.0)
        result = evaluate_policy(records, policy)
        assert not result.passed
        corrupt_check = next(c for c in result.checks if c.name == "max_corrupt_pct")
        assert not corrupt_check.passed
        assert corrupt_check.observed == 50.0
        assert "/bad.jpg" in corrupt_check.sample_paths

    def test_failing_overexposed_pct(self) -> None:
        records = [ImageRecord(path=f"/img_{i}.jpg", filename=f"img_{i}.jpg") for i in range(10)]
        records[0].is_overexposed = True
        policy = Policy(min_images_total=1, max_overexposed_pct=5.0)
        result = evaluate_policy(records, policy)
        assert not result.passed
        check = next(c for c in result.checks if c.name == "max_overexposed_pct")
        assert not check.passed

    def test_failing_underexposed_pct(self) -> None:
        records = [ImageRecord(path=f"/img_{i}.jpg", filename=f"img_{i}.jpg") for i in range(10)]
        records[0].is_dark = True
        policy = Policy(min_images_total=1, max_underexposed_pct=5.0)
        result = evaluate_policy(records, policy)
        assert not result.passed
        check = next(c for c in result.checks if c.name == "max_underexposed_pct")
        assert not check.passed

    def test_failing_duplicate_pct(self) -> None:
        records = [
            ImageRecord(path="/a.jpg", filename="a.jpg", phash="aabb"),
            ImageRecord(path="/b.jpg", filename="b.jpg", phash="aabb"),
        ]
        policy = Policy(min_images_total=1, max_duplicate_pct=1.0)
        result = evaluate_policy(records, policy)
        assert not result.passed
        check = next(c for c in result.checks if c.name == "max_duplicate_pct")
        assert not check.passed

    def test_empty_records(self) -> None:
        result = evaluate_policy([], Policy())
        assert not result.passed
        assert result.total_images == 0

    def test_to_dict(self) -> None:
        records = [ImageRecord(path="/a.jpg", filename="a.jpg")]
        result = evaluate_policy(records, Policy(min_images_total=1))
        d = result.to_dict()
        assert "passed" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)


class TestLoadPolicy:
    def test_load_policy(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yml"
        policy_file.write_text(
            "max_corrupt_pct: 2.5\nmin_images_total: 50\nmax_duplicate_pct: 15.0\n"
        )
        policy = load_policy(str(policy_file))
        assert policy.max_corrupt_pct == 2.5
        assert policy.min_images_total == 50
        assert policy.max_duplicate_pct == 15.0
        # Defaults preserved for unset fields
        assert policy.max_overexposed_pct == 5.0
        assert policy.max_underexposed_pct == 5.0

    def test_load_empty_policy(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "empty.yml"
        policy_file.write_text("")
        policy = load_policy(str(policy_file))
        # All defaults
        assert policy.max_corrupt_pct == 1.0
        assert policy.min_images_total == 100
