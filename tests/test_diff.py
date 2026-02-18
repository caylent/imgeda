"""Tests for manifest diff logic."""

from __future__ import annotations

from imgeda.core.diff import diff_manifests
from imgeda.models.manifest import ImageRecord


class TestDiffManifests:
    def test_identical_manifests(self) -> None:
        records = [
            ImageRecord(path="/a.jpg", filename="a.jpg", width=100, height=100),
            ImageRecord(path="/b.jpg", filename="b.jpg", width=200, height=200),
        ]
        result = diff_manifests(records, records)
        assert result.summary.added_count == 0
        assert result.summary.removed_count == 0
        assert result.summary.changed_count == 0
        assert result.unchanged_count == 2

    def test_added_images(self) -> None:
        old = [ImageRecord(path="/a.jpg", filename="a.jpg")]
        new = [
            ImageRecord(path="/a.jpg", filename="a.jpg"),
            ImageRecord(path="/b.jpg", filename="b.jpg"),
        ]
        result = diff_manifests(old, new)
        assert result.added == ["/b.jpg"]
        assert result.summary.added_count == 1
        assert result.summary.removed_count == 0

    def test_removed_images(self) -> None:
        old = [
            ImageRecord(path="/a.jpg", filename="a.jpg"),
            ImageRecord(path="/b.jpg", filename="b.jpg"),
        ]
        new = [ImageRecord(path="/a.jpg", filename="a.jpg")]
        result = diff_manifests(old, new)
        assert result.removed == ["/b.jpg"]
        assert result.summary.removed_count == 1
        assert result.summary.added_count == 0

    def test_changed_images(self) -> None:
        old = [ImageRecord(path="/a.jpg", filename="a.jpg", width=100, height=100)]
        new = [ImageRecord(path="/a.jpg", filename="a.jpg", width=200, height=150)]
        result = diff_manifests(old, new)
        assert len(result.changed) == 1
        changed = result.changed[0]
        assert changed.path == "/a.jpg"
        assert "width" in changed.fields
        assert changed.fields["width"] == (100, 200)
        assert "height" in changed.fields
        assert changed.fields["height"] == (100, 150)

    def test_mixed_diff(self) -> None:
        old = [
            ImageRecord(path="/a.jpg", filename="a.jpg", width=100),
            ImageRecord(path="/b.jpg", filename="b.jpg", width=200),
            ImageRecord(path="/c.jpg", filename="c.jpg", width=300),
        ]
        new = [
            ImageRecord(path="/a.jpg", filename="a.jpg", width=100),  # unchanged
            ImageRecord(path="/b.jpg", filename="b.jpg", width=250),  # changed
            ImageRecord(path="/d.jpg", filename="d.jpg", width=400),  # added
        ]
        result = diff_manifests(old, new)
        assert result.added == ["/d.jpg"]
        assert result.removed == ["/c.jpg"]
        assert len(result.changed) == 1
        assert result.unchanged_count == 1

    def test_empty_manifests(self) -> None:
        result = diff_manifests([], [])
        assert result.summary.total_old == 0
        assert result.summary.total_new == 0
        assert result.added == []
        assert result.removed == []

    def test_corrupt_counts(self) -> None:
        old = [
            ImageRecord(path="/a.jpg", filename="a.jpg", is_corrupt=True),
            ImageRecord(path="/b.jpg", filename="b.jpg"),
        ]
        new = [
            ImageRecord(path="/a.jpg", filename="a.jpg"),
            ImageRecord(path="/b.jpg", filename="b.jpg"),
        ]
        result = diff_manifests(old, new)
        assert result.summary.corrupt_old == 1
        assert result.summary.corrupt_new == 0

    def test_to_dict(self) -> None:
        old = [ImageRecord(path="/a.jpg", filename="a.jpg")]
        new = [ImageRecord(path="/b.jpg", filename="b.jpg")]
        result = diff_manifests(old, new)
        d = result.to_dict()
        assert "added" in d
        assert "removed" in d
        assert "summary" in d
        assert d["added"] == ["/b.jpg"]
        assert d["removed"] == ["/a.jpg"]
