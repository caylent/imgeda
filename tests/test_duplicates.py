"""Tests for duplicate detection."""

from __future__ import annotations

from imgeda.core.duplicates import find_exact_duplicates, find_near_duplicates
from imgeda.models.manifest import ImageRecord


class TestExactDuplicates:
    def test_finds_exact_matches(self) -> None:
        records = [
            ImageRecord(path="/a.jpg", phash="abcd1234"),
            ImageRecord(path="/b.jpg", phash="abcd1234"),
            ImageRecord(path="/c.jpg", phash="different"),
        ]
        groups = find_exact_duplicates(records)
        assert len(groups) == 1
        assert "abcd1234" in groups
        assert len(groups["abcd1234"]) == 2

    def test_excludes_corrupt(self) -> None:
        records = [
            ImageRecord(path="/a.jpg", phash="abcd1234"),
            ImageRecord(path="/b.jpg", phash="abcd1234", is_corrupt=True),
        ]
        groups = find_exact_duplicates(records)
        assert len(groups) == 0

    def test_no_duplicates(self) -> None:
        records = [
            ImageRecord(path="/a.jpg", phash="aaaa"),
            ImageRecord(path="/b.jpg", phash="bbbb"),
        ]
        groups = find_exact_duplicates(records)
        assert len(groups) == 0

    def test_empty_input(self) -> None:
        assert find_exact_duplicates([]) == {}


class TestNearDuplicates:
    def test_finds_near_matches(self) -> None:
        # Two hashes that differ by just a few bits
        records = [
            ImageRecord(path="/a.jpg", phash="0000000000000000"),
            ImageRecord(path="/b.jpg", phash="0000000000000001"),  # 1 bit diff
            ImageRecord(path="/c.jpg", phash="ffffffffffffffff"),  # very different
        ]
        groups = find_near_duplicates(records, hamming_threshold=8)
        assert len(groups) >= 1
        paths = {r.path for group in groups for r in group}
        assert "/a.jpg" in paths
        assert "/b.jpg" in paths

    def test_empty_input(self) -> None:
        assert find_near_duplicates([]) == []

    def test_no_hashes(self) -> None:
        records = [
            ImageRecord(path="/a.jpg"),
            ImageRecord(path="/b.jpg"),
        ]
        assert find_near_duplicates(records) == []

    def test_excludes_corrupt(self) -> None:
        records = [
            ImageRecord(path="/a.jpg", phash="0000000000000000", is_corrupt=True),
            ImageRecord(path="/b.jpg", phash="0000000000000001", is_corrupt=True),
        ]
        assert find_near_duplicates(records) == []
