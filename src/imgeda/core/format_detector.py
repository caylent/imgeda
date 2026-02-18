"""Dataset format detection — probes directory structure to identify ML dataset formats."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


@dataclass
class DatasetInfo:
    """Detected dataset format and metadata."""

    format: str  # "yolo" | "coco" | "voc" | "classification" | "flat"
    image_dirs: list[str]  # paths to image directories (may be per-split)
    num_images: int
    estimated_size_bytes: int
    splits: dict[str, int]  # {"train": 2940, "val": 740} or {} for flat
    num_classes: int | None = None
    class_names: list[str] | None = None
    annotations_path: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


def detect_format(root: str) -> DatasetInfo:
    """Detect ML dataset format by inspecting directory structure.

    Checks in order: YOLO, COCO, Pascal VOC, Classification, Flat (fallback).
    """
    root_path = Path(root)

    # 1. YOLO — data.yaml at root
    info = _try_yolo(root_path)
    if info is not None:
        return info

    # 2. COCO — annotations/*.json with COCO keys
    info = _try_coco(root_path)
    if info is not None:
        return info

    # 3. Pascal VOC — Annotations/ + JPEGImages/
    info = _try_voc(root_path)
    if info is not None:
        return info

    # 4. Classification — >3 subdirs each containing images
    info = _try_classification(root_path)
    if info is not None:
        return info

    # 5. Flat fallback
    return _build_flat(root_path)


def _count_images_in(directory: Path) -> int:
    """Count image files recursively under a directory."""
    count = 0
    if not directory.is_dir():
        return 0
    for dirpath, _dirnames, filenames in os.walk(directory):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTENSIONS:
                count += 1
    return count


def _estimate_size(directory: Path, sample_limit: int = 100) -> int:
    """Estimate total image size by sampling up to sample_limit files."""
    sizes: list[int] = []
    total_images = 0
    for dirpath, _dirnames, filenames in os.walk(directory):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTENSIONS:
                total_images += 1
                if len(sizes) < sample_limit:
                    try:
                        sizes.append(os.path.getsize(os.path.join(dirpath, fn)))
                    except OSError:
                        pass
    if not sizes:
        return 0
    avg = sum(sizes) / len(sizes)
    return int(avg * total_images)


def _parse_simple_yaml(path: Path) -> dict[str, str | list[str]]:
    """Parse a simple YAML file without pyyaml dependency.

    Handles basic key: value pairs and simple lists (names: [...] or
    names:\n  - item lines). Enough for YOLO data.yaml files.
    """
    result: dict[str, str | list[str]] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return result

    lines = text.splitlines()
    current_key: str | None = None
    current_list: list[str] | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check for list continuation (indented "- item")
        if current_key is not None and current_list is not None:
            if stripped.startswith("- "):
                item = stripped[2:].strip().strip("'\"")
                current_list.append(item)
                continue
            else:
                # End of list
                result[current_key] = current_list
                current_key = None
                current_list = None

        if ":" not in stripped:
            continue

        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()

        if not value:
            # Could be start of a list
            current_key = key
            current_list = []
        elif value.startswith("[") and value.endswith("]"):
            # Inline list: names: [cat, dog, bird]
            items = value[1:-1].split(",")
            result[key] = [item.strip().strip("'\"") for item in items if item.strip()]
        else:
            result[key] = value.strip("'\"")

    # Flush any pending list
    if current_key is not None and current_list is not None:
        result[current_key] = current_list

    return result


def _try_yolo(root: Path) -> DatasetInfo | None:
    """Detect YOLO format via data.yaml."""
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        return None

    parsed = _parse_simple_yaml(yaml_path)
    if not parsed:
        return None

    # Extract class names
    names_val = parsed.get("names")
    class_names: list[str] | None = None
    num_classes: int | None = None
    if isinstance(names_val, list):
        class_names = names_val[:10]
        num_classes = len(names_val)
    elif "nc" in parsed:
        nc_val = parsed["nc"]
        if isinstance(nc_val, str) and nc_val.isdigit():
            num_classes = int(nc_val)

    # Detect splits from data.yaml or directory structure
    splits: dict[str, int] = {}
    image_dirs: list[str] = []

    for split_name in ("train", "val", "test"):
        split_val = parsed.get(split_name)
        if isinstance(split_val, str):
            split_path = root / split_val
            # YOLO convention: images dir mirrors the path
            # data.yaml may point to images/train or just train
            if split_path.is_dir():
                count = _count_images_in(split_path)
                if count > 0:
                    splits[split_name] = count
                    image_dirs.append(str(split_path))
                continue
            # Try under images/
            img_split = root / "images" / split_name
            if img_split.is_dir():
                count = _count_images_in(img_split)
                if count > 0:
                    splits[split_name] = count
                    image_dirs.append(str(img_split))

    # Fallback: check images/ dir directly
    if not image_dirs:
        images_dir = root / "images"
        if images_dir.is_dir():
            image_dirs.append(str(images_dir))

    # Detect annotations path
    labels_dir = root / "labels"
    annotations_path = str(labels_dir) if labels_dir.is_dir() else None

    num_images = sum(splits.values()) if splits else sum(
        _count_images_in(Path(d)) for d in image_dirs
    )

    return DatasetInfo(
        format="yolo",
        image_dirs=image_dirs or [str(root)],
        num_images=num_images,
        estimated_size_bytes=_estimate_size(root),
        splits=splits,
        num_classes=num_classes,
        class_names=class_names,
        annotations_path=annotations_path,
        extra={},
    )


def _try_coco(root: Path) -> DatasetInfo | None:
    """Detect COCO format via annotations/*.json with COCO keys."""
    ann_dir = root / "annotations"
    if not ann_dir.is_dir():
        return None

    json_files = list(ann_dir.glob("*.json"))
    if not json_files:
        return None

    # Check first JSON for COCO structure
    coco_file: Path | None = None
    categories: list[dict[str, str]] = []
    for jf in json_files:
        try:
            # Read first portion to check keys
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "images" in data and "annotations" in data:
                coco_file = jf
                categories = data.get("categories", [])
                break
        except (json.JSONDecodeError, OSError):
            continue

    if coco_file is None:
        return None

    # Extract class info
    class_names: list[str] | None = None
    num_classes: int | None = None
    if categories:
        all_names = [c.get("name", "") for c in categories if isinstance(c, dict)]
        num_classes = len(all_names)
        class_names = all_names[:10] if all_names else None

    # Detect splits from annotation filenames
    splits: dict[str, int] = {}
    image_dirs: list[str] = []

    images_dir = root / "images"
    if images_dir.is_dir():
        image_dirs.append(str(images_dir))

    # Try to detect splits from annotation file names (e.g. instances_train2017.json)
    for jf in json_files:
        name = jf.stem.lower()
        for split_name in ("train", "val", "test"):
            if split_name in name:
                try:
                    with open(jf, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "images" in data:
                        splits[split_name] = len(data["images"])
                except (json.JSONDecodeError, OSError):
                    pass

    num_images = sum(splits.values()) if splits else (
        _count_images_in(images_dir) if images_dir.is_dir() else 0
    )

    return DatasetInfo(
        format="coco",
        image_dirs=image_dirs or [str(root)],
        num_images=num_images,
        estimated_size_bytes=_estimate_size(root),
        splits=splits,
        num_classes=num_classes,
        class_names=class_names,
        annotations_path=str(ann_dir),
        extra={},
    )


def _try_voc(root: Path) -> DatasetInfo | None:
    """Detect Pascal VOC format via Annotations/ + JPEGImages/."""
    ann_dir = root / "Annotations"
    img_dir = root / "JPEGImages"

    if not ann_dir.is_dir() or not img_dir.is_dir():
        return None

    # Check for XML files in Annotations
    xml_files = list(ann_dir.glob("*.xml"))
    if not xml_files:
        return None

    num_images = _count_images_in(img_dir)

    # Check for ImageSets/Main/ split files
    splits: dict[str, int] = {}
    imagesets_dir = root / "ImageSets" / "Main"
    if imagesets_dir.is_dir():
        for txt_file in imagesets_dir.glob("*.txt"):
            split_name = txt_file.stem.lower()
            if split_name in ("train", "val", "test", "trainval"):
                try:
                    lines = txt_file.read_text().strip().splitlines()
                    count = len([ln for ln in lines if ln.strip()])
                    if count > 0:
                        splits[split_name] = count
                except OSError:
                    pass

    return DatasetInfo(
        format="voc",
        image_dirs=[str(img_dir)],
        num_images=num_images,
        estimated_size_bytes=_estimate_size(img_dir),
        splits=splits,
        num_classes=None,
        class_names=None,
        annotations_path=str(ann_dir),
        extra={},
    )


def _try_classification(root: Path) -> DatasetInfo | None:
    """Detect classification format: >3 subdirs each containing images."""
    # Skip if annotation-style dirs exist
    for ann_dir_name in ("labels", "annotations", "Annotations"):
        if (root / ann_dir_name).is_dir():
            return None

    subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if len(subdirs) < 3:
        return None

    # Check that most subdirs contain images
    dirs_with_images = 0
    total_images = 0
    for sd in subdirs:
        count = _count_images_in(sd)
        if count > 0:
            dirs_with_images += 1
            total_images += count

    # Require majority of subdirs to contain images
    if dirs_with_images < len(subdirs) * 0.5:
        return None

    class_names_all = sorted(
        sd.name for sd in subdirs if _count_images_in(sd) > 0
    )

    return DatasetInfo(
        format="classification",
        image_dirs=[str(root)],
        num_images=total_images,
        estimated_size_bytes=_estimate_size(root),
        splits={},
        num_classes=len(class_names_all),
        class_names=class_names_all[:10],
        annotations_path=None,
        extra={},
    )


def _build_flat(root: Path) -> DatasetInfo:
    """Fallback: flat directory with images."""
    num_images = _count_images_in(root)
    return DatasetInfo(
        format="flat",
        image_dirs=[str(root)],
        num_images=num_images,
        estimated_size_bytes=_estimate_size(root) if num_images > 0 else 0,
        splits={},
        num_classes=None,
        class_names=None,
        annotations_path=None,
        extra={},
    )
