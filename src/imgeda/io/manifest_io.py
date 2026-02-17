"""JSONL manifest read/write/append with crash-tolerant parsing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import orjson

from imgeda.models.manifest import MANIFEST_META_KEY, ImageRecord, ManifestMeta


def write_meta(path: str | Path, meta: ManifestMeta) -> None:
    """Write (or overwrite) the metadata header as the first line of the manifest.

    Uses atomic write via temp file + rename to avoid corruption on crash.
    """
    path = Path(path)
    meta_line = orjson.dumps(meta.to_dict(), option=orjson.OPT_APPEND_NEWLINE)

    if not path.exists():
        path.write_bytes(meta_line)
        return

    # Read existing records (skip old meta), write new meta + records atomically
    rest_lines: list[bytes] = []
    with open(path, "rb") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = orjson.loads(stripped)
            except orjson.JSONDecodeError:
                continue
            if data.get(MANIFEST_META_KEY):
                continue  # skip old meta line
            rest_lines.append(line if line.endswith(b"\n") else line + b"\n")

    # Write to temp file then rename for atomicity
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(meta_line)
            for line in rest_lines:
                f.write(line)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def create_manifest(path: str | Path, meta: ManifestMeta) -> None:
    """Create a fresh manifest file with only the metadata header (truncates existing)."""
    path = Path(path)
    meta_line = orjson.dumps(meta.to_dict(), option=orjson.OPT_APPEND_NEWLINE)
    path.write_bytes(meta_line)


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
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = orjson.loads(line)
            except orjson.JSONDecodeError:
                # Skip corrupt lines (likely truncated from crash)
                continue

            if data.get(MANIFEST_META_KEY) and meta is None:
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
