"""Integration-specific fixtures for end-to-end CLI pipeline tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def e2e_image_dir(tmp_path: Path) -> Path:
    """Create a directory with ~16 programmatic images covering all issue types.

    Layout:
        10 normal JPEGs  (varying sizes, random pixel data)
         1 dark PNG       (mean brightness ~15)
         1 bright PNG     (mean brightness ~240)
         1 artifact PNG   (dark corners, bright center)
         2 duplicate JPEGs (exact same pixel content)
         1 corrupt JPEG   (garbage bytes)
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    rng = np.random.RandomState(42)

    # 10 normal images with varying dimensions
    for i in range(10):
        w, h = 200 + i * 50, 150 + i * 30
        arr = rng.randint(60, 200, (h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"normal_{i:03d}.jpg")

    # Dark image (underexposed)
    arr = np.full((100, 100, 3), 15, dtype=np.uint8)
    Image.fromarray(arr).save(img_dir / "dark_001.png")

    # Bright image (overexposed)
    arr = np.full((100, 100, 3), 240, dtype=np.uint8)
    Image.fromarray(arr).save(img_dir / "bright_001.png")

    # Artifact image (dark corners, bright center)
    arr = np.full((200, 200, 3), 180, dtype=np.uint8)
    arr[:20, :20] = 20  # top-left
    arr[:20, -20:] = 20  # top-right
    arr[-20:, :20] = 20  # bottom-left
    arr[-20:, -20:] = 20  # bottom-right
    Image.fromarray(arr).save(img_dir / "artifact_001.png")

    # Duplicate pair (identical pixel content, different filenames)
    dup_arr = rng.randint(50, 200, (150, 150, 3), dtype=np.uint8)
    dup_img = Image.fromarray(dup_arr)
    dup_img.save(img_dir / "dup_a.jpg")
    dup_img.save(img_dir / "dup_b.jpg")

    # Corrupt file (not a valid image)
    (img_dir / "corrupt.jpg").write_bytes(b"not a real image file content")

    return img_dir


@pytest.fixture
def e2e_workspace(tmp_path: Path, e2e_image_dir: Path) -> SimpleNamespace:
    """Return a SimpleNamespace containing all paths needed for integration tests.

    Also creates the gate policy YAML files on disk.
    """
    ws = SimpleNamespace(
        image_dir=e2e_image_dir,
        manifest=str(tmp_path / "manifest.jsonl"),
        manifest_v2=str(tmp_path / "manifest_v2.jsonl"),
        plots_dir=str(tmp_path / "plots"),
        report_html=str(tmp_path / "report.html"),
        diff_json=str(tmp_path / "diff.json"),
        gate_pass_policy=str(tmp_path / "gate_pass.yml"),
        gate_fail_policy=str(tmp_path / "gate_fail.yml"),
        gate_json=str(tmp_path / "gate.json"),
        parquet_out=str(tmp_path / "export.parquet"),
    )

    # Passing policy: generous thresholds that our 16-image dataset will satisfy
    Path(ws.gate_pass_policy).write_text(
        "min_images_total: 5\n"
        "max_corrupt_pct: 10.0\n"
        "max_overexposed_pct: 20.0\n"
        "max_underexposed_pct: 20.0\n"
        "max_duplicate_pct: 20.0\n"
    )

    # Failing policy: impossible thresholds
    Path(ws.gate_fail_policy).write_text(
        "min_images_total: 1000\n"
        "max_corrupt_pct: 0.0\n"
        "max_overexposed_pct: 0.0\n"
        "max_underexposed_pct: 0.0\n"
        "max_duplicate_pct: 0.0\n"
    )

    return ws
