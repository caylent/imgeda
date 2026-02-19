"""Tests for cross-split data leakage detection."""

from __future__ import annotations

from imgeda.core.leakage import _hamming_distance, detect_leakage
from imgeda.models.manifest import ImageRecord


def _rec(path: str, phash: str) -> ImageRecord:
    return ImageRecord(path=path, filename=path.split("/")[-1], phash=phash)


class TestHammingDistance:
    def test_identical(self) -> None:
        assert _hamming_distance("abcd", "abcd") == 0

    def test_one_bit_diff(self) -> None:
        # 0xa = 1010, 0xb = 1011 → 1 bit difference
        assert _hamming_distance("a", "b") == 1

    def test_all_bits_diff(self) -> None:
        # 0x0 vs 0xf = 0000 vs 1111 → 4 bits
        assert _hamming_distance("0", "f") == 4

    def test_invalid_hex(self) -> None:
        assert _hamming_distance("xyz", "abc") == 999

    def test_longer_hashes(self) -> None:
        h1 = "0" * 16
        h2 = "0" * 15 + "1"
        assert _hamming_distance(h1, h2) == 1


class TestDetectLeakage:
    def test_no_leakage(self) -> None:
        splits = {
            "train": [_rec("/train/a.jpg", "aaaa"), _rec("/train/b.jpg", "bbbb")],
            "val": [_rec("/val/c.jpg", "cccc"), _rec("/val/d.jpg", "dddd")],
        }
        result = detect_leakage(splits, hamming_threshold=0)
        assert len(result) == 0

    def test_exact_leakage(self) -> None:
        splits = {
            "train": [_rec("/train/a.jpg", "aaaa"), _rec("/train/b.jpg", "bbbb")],
            "val": [_rec("/val/a_copy.jpg", "aaaa")],
        }
        result = detect_leakage(splits, hamming_threshold=0)
        assert len(result) >= 2  # Both images with same hash flagged
        found_in = set()
        for item in result:
            for s in item["found_in"]:
                found_in.add(s)
        assert "train" in found_in
        assert "val" in found_in

    def test_near_leakage(self) -> None:
        """Near-duplicate hashes across splits should be flagged."""
        splits = {
            "train": [_rec("/train/a.jpg", "aaaa0000")],
            "val": [_rec("/val/b.jpg", "aaaa0001")],  # 1 bit difference
        }
        result = detect_leakage(splits, hamming_threshold=8)
        assert len(result) >= 1

    def test_single_split_no_leakage(self) -> None:
        """A single split can't have cross-split leakage."""
        splits = {
            "train": [_rec("/train/a.jpg", "aaaa"), _rec("/train/b.jpg", "aaaa")],
        }
        result = detect_leakage(splits, hamming_threshold=0)
        assert len(result) == 0

    def test_empty_splits(self) -> None:
        result = detect_leakage({}, hamming_threshold=0)
        assert result == []

    def test_records_without_phash(self) -> None:
        splits = {
            "train": [ImageRecord(path="/train/a.jpg", filename="a.jpg")],
            "val": [ImageRecord(path="/val/b.jpg", filename="b.jpg")],
        }
        result = detect_leakage(splits, hamming_threshold=0)
        assert result == []

    def test_three_splits(self) -> None:
        """Leakage across 3 splits."""
        common_hash = "deadbeef"
        splits = {
            "train": [_rec("/train/a.jpg", common_hash)],
            "val": [_rec("/val/a.jpg", common_hash)],
            "test": [_rec("/test/a.jpg", common_hash)],
        }
        result = detect_leakage(splits, hamming_threshold=0)
        assert len(result) >= 3
        # All three splits should be in found_in for at least one record
        all_splits_found = set()
        for item in result:
            for s in item["found_in"]:
                all_splits_found.add(s)
        assert all_splits_found == {"train", "val", "test"}

    def test_result_sorted_by_path(self) -> None:
        splits = {
            "train": [_rec("/z.jpg", "aaaa"), _rec("/a.jpg", "bbbb")],
            "val": [_rec("/y.jpg", "aaaa"), _rec("/b.jpg", "bbbb")],
        }
        result = detect_leakage(splits, hamming_threshold=0)
        paths = [r["path"] for r in result]
        assert paths == sorted(paths)
