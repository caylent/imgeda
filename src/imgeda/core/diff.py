"""Manifest comparison — pure diff logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from imgeda.models.manifest import ImageRecord


@dataclass(slots=True)
class DiffResult:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[ChangedRecord] = field(default_factory=list)
    unchanged_count: int = 0

    # Distribution deltas (new - old)
    summary: DiffSummary = field(default_factory=lambda: DiffSummary())

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "added": sorted(self.added),
            "removed": sorted(self.removed),
            "changed": [c.to_dict() for c in sorted(self.changed, key=lambda c: c.path)],
            "unchanged_count": self.unchanged_count,
            "summary": asdict(self.summary),
        }
        return d


@dataclass(slots=True)
class ChangedRecord:
    path: str = ""
    fields: dict[str, tuple[object, object]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "fields": {k: {"old": v[0], "new": v[1]} for k, v in self.fields.items()},
        }


@dataclass(slots=True)
class DiffSummary:
    total_old: int = 0
    total_new: int = 0
    added_count: int = 0
    removed_count: int = 0
    changed_count: int = 0
    corrupt_old: int = 0
    corrupt_new: int = 0
    duplicate_groups_old: int = 0
    duplicate_groups_new: int = 0


_COMPARE_FIELDS = (
    "file_size_bytes",
    "width",
    "height",
    "format",
    "color_mode",
    "is_corrupt",
    "is_dark",
    "is_overexposed",
    "has_border_artifact",
    "phash",
)


def diff_manifests(
    old_records: list[ImageRecord],
    new_records: list[ImageRecord],
) -> DiffResult:
    """Compare two manifest record lists by path. Return structured diff."""
    old_by_path = {r.path: r for r in old_records}
    new_by_path = {r.path: r for r in new_records}

    old_paths = set(old_by_path.keys())
    new_paths = set(new_by_path.keys())

    added = sorted(new_paths - old_paths)
    removed = sorted(old_paths - new_paths)
    common = old_paths & new_paths

    changed: list[ChangedRecord] = []
    unchanged_count = 0

    for path in sorted(common):
        old_rec = old_by_path[path]
        new_rec = new_by_path[path]
        diffs: dict[str, tuple[object, object]] = {}

        for fld in _COMPARE_FIELDS:
            old_val = getattr(old_rec, fld)
            new_val = getattr(new_rec, fld)
            if old_val != new_val:
                diffs[fld] = (old_val, new_val)

        if diffs:
            changed.append(ChangedRecord(path=path, fields=diffs))
        else:
            unchanged_count += 1

    # Compute duplicate group counts
    from imgeda.core.duplicates import find_exact_duplicates

    dup_old = len(find_exact_duplicates(old_records))
    dup_new = len(find_exact_duplicates(new_records))

    summary = DiffSummary(
        total_old=len(old_records),
        total_new=len(new_records),
        added_count=len(added),
        removed_count=len(removed),
        changed_count=len(changed),
        corrupt_old=sum(1 for r in old_records if r.is_corrupt),
        corrupt_new=sum(1 for r in new_records if r.is_corrupt),
        duplicate_groups_old=dup_old,
        duplicate_groups_new=dup_new,
    )

    return DiffResult(
        added=added,
        removed=removed,
        changed=changed,
        unchanged_count=unchanged_count,
        summary=summary,
    )
