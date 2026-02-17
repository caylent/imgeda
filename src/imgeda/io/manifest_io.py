"""JSONL manifest read/write/append with crash-tolerant parsing."""

from __future__ import annotations

import os
from pathlib import Path

import orjson

from imgeda.models.manifest import ImageRecord, ManifestMeta


def write_meta(path: str | Path, meta: ManifestMeta) -> None:
    """Write (or overwrite) the metadata header as the first line of the manifest."""
    path = Path(path)
    data = orjson.dumps(meta.to_dict(), option=orjson.OPT_APPEND_NEWLINE)
    if path.exists():
        # Read existing content, replace first line
        existing = path.read_bytes()
        lines = existing.split(b"\n", 1)
        rest = lines[1] if len(lines) > 1 else b""
        path.write_bytes(data + rest.lstrip(b"\n") if rest.strip() else data)
    else:
        path.write_bytes(data)


def append_records(path: str | Path, records: list[ImageRecord]) -> None:
    """Append records to JSONL manifest file."""
    path = Path(path)
    with open(path, "ab") as f:
        for rec in records:
            f.write(orjson.dumps(rec.to_dict(), option=orjson.OPT_APPEND_NEWLINE))
        f.flush()
        os.fsync(f.fileno())


def read_manifest(path: str | Path) -> tuple[ManifestMeta | None, list[ImageRecord]]:
    """Read a JSONL manifest, skipping corrupt trailing lines (crash tolerance)."""
    path = Path(path)
    if not path.exists():
        return None, []

    meta: ManifestMeta | None = None
    records: list[ImageRecord] = []

    with open(path, "rb") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = orjson.loads(line)
            except orjson.JSONDecodeError:
                # Skip corrupt lines (likely truncated from crash)
                continue

            if i == 0 and data.get("__manifest_meta__"):
                meta = ManifestMeta.from_dict(data)
            else:
                records.append(ImageRecord.from_dict(data))

    return meta, records


def build_resume_set(records: list[ImageRecord]) -> set[tuple[str, int, float]]:
    """Build set of (path, file_size_bytes, mtime) for resume detection."""
    return {(r.path, r.file_size_bytes, r.mtime) for r in records}


def make_resume_key(path: str, size: int, mtime: float) -> tuple[str, int, float]:
    """Create a resume key for an image file."""
    return (path, size, mtime)
