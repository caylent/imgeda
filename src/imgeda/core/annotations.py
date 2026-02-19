"""Annotation analysis for YOLO, COCO, and Pascal VOC formats."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BBox:
    """Bounding box in normalized [0,1] coordinates."""

    class_name: str = ""
    class_id: int = 0
    x_center: float = 0.0
    y_center: float = 0.0
    width: float = 0.0
    height: float = 0.0
    area: float = 0.0  # normalized area (fraction of image)


@dataclass(slots=True)
class AnnotationStats:
    """Aggregated annotation statistics for a dataset."""

    total_images: int = 0
    annotated_images: int = 0
    unannotated_images: int = 0
    total_annotations: int = 0
    class_names: list[str] = field(default_factory=list)
    num_classes: int = 0

    # Per-class counts
    class_counts: dict[str, int] = field(default_factory=dict)

    # Objects per image distribution
    objects_per_image: list[int] = field(default_factory=list)
    mean_objects_per_image: float = 0.0
    max_objects_per_image: int = 0

    # Bounding box statistics
    bbox_widths: list[float] = field(default_factory=list)
    bbox_heights: list[float] = field(default_factory=list)
    bbox_areas: list[float] = field(default_factory=list)
    bbox_aspect_ratios: list[float] = field(default_factory=list)
    bbox_x_centers: list[float] = field(default_factory=list)
    bbox_y_centers: list[float] = field(default_factory=list)

    # COCO-style size breakdown (by area relative to image)
    small_count: int = 0  # area < 0.01 (< 1% of image)
    medium_count: int = 0  # 0.01 <= area < 0.1
    large_count: int = 0  # area >= 0.1

    # Class co-occurrence matrix (class_a -> class_b -> count)
    co_occurrence: dict[str, dict[str, int]] = field(default_factory=dict)

    # Per-split class distribution (split_name -> class -> count)
    split_class_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    # Orphan detection
    orphan_images: list[str] = field(default_factory=list)  # images with no annotations
    orphan_annotations: list[str] = field(default_factory=list)  # annotations with no images

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_images": self.total_images,
            "annotated_images": self.annotated_images,
            "unannotated_images": self.unannotated_images,
            "total_annotations": self.total_annotations,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "class_counts": self.class_counts,
            "mean_objects_per_image": round(self.mean_objects_per_image, 2),
            "max_objects_per_image": self.max_objects_per_image,
            "small_count": self.small_count,
            "medium_count": self.medium_count,
            "large_count": self.large_count,
            "orphan_images": self.orphan_images[:20],
            "orphan_annotations": self.orphan_annotations[:20],
        }


def _parse_yolo_labels(
    label_dir: str, class_names: list[str]
) -> tuple[dict[str, list[BBox]], list[str]]:
    """Parse YOLO format .txt label files.

    Returns:
        (image_stem -> list of BBox, list of class_names)
    """
    annotations: dict[str, list[BBox]] = {}
    errors: list[str] = []

    for txt_file in Path(label_dir).glob("*.txt"):
        stem = txt_file.stem
        boxes: list[BBox] = []
        try:
            for line in txt_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                area = w * h
                boxes.append(
                    BBox(
                        class_name=cls_name,
                        class_id=cls_id,
                        x_center=xc,
                        y_center=yc,
                        width=w,
                        height=h,
                        area=area,
                    )
                )
        except Exception:
            errors.append(str(txt_file))
        annotations[stem] = boxes

    return annotations, errors


def _parse_coco_annotations(
    json_path: str,
) -> tuple[dict[str, list[BBox]], list[str], dict[int, str]]:
    """Parse COCO JSON annotations.

    Returns:
        (image_filename -> list of BBox, errors, category_map)
    """
    with open(json_path) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    images = {img["id"]: img for img in data.get("images", [])}

    annotations: dict[str, list[BBox]] = {}
    # Initialize all images as having empty annotation lists
    for img in images.values():
        fname = img.get("file_name", "")
        if fname:
            annotations.setdefault(fname, [])

    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in images:
            continue
        img = images[img_id]
        fname = img.get("file_name", "")
        iw, ih = img.get("width", 1), img.get("height", 1)

        bbox = ann.get("bbox", [0, 0, 0, 0])  # COCO: [x, y, w, h] in pixels
        if len(bbox) < 4:
            continue

        bx, by, bw, bh = bbox
        # Normalize to [0,1]
        nw = bw / iw if iw > 0 else 0
        nh = bh / ih if ih > 0 else 0
        nx = (bx + bw / 2) / iw if iw > 0 else 0
        ny = (by + bh / 2) / ih if ih > 0 else 0
        area = nw * nh

        cat_id = ann.get("category_id", 0)
        cls_name = categories.get(cat_id, str(cat_id))

        annotations.setdefault(fname, []).append(
            BBox(
                class_name=cls_name,
                class_id=cat_id,
                x_center=nx,
                y_center=ny,
                width=nw,
                height=nh,
                area=area,
            )
        )

    return annotations, [], categories


def _parse_voc_annotations(
    annotation_dir: str,
) -> tuple[dict[str, list[BBox]], list[str]]:
    """Parse Pascal VOC XML annotations.

    Returns:
        (image_filename -> list of BBox, errors)
    """
    annotations: dict[str, list[BBox]] = {}
    errors: list[str] = []

    for xml_file in Path(annotation_dir).glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            fname_el = root.find("filename")
            fname = fname_el.text if fname_el is not None and fname_el.text else xml_file.stem

            size_el = root.find("size")
            iw = int(size_el.findtext("width", "1")) if size_el is not None else 1
            ih = int(size_el.findtext("height", "1")) if size_el is not None else 1

            boxes: list[BBox] = []
            for obj in root.findall("object"):
                name_el = obj.find("name")
                cls_name = name_el.text if name_el is not None and name_el.text else "unknown"

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))

                bw = xmax - xmin
                bh = ymax - ymin
                nw = bw / iw if iw > 0 else 0
                nh = bh / ih if ih > 0 else 0
                nx = (xmin + bw / 2) / iw if iw > 0 else 0
                ny = (ymin + bh / 2) / ih if ih > 0 else 0

                boxes.append(
                    BBox(
                        class_name=cls_name,
                        class_id=0,
                        x_center=nx,
                        y_center=ny,
                        width=nw,
                        height=nh,
                        area=nw * nh,
                    )
                )

            annotations[fname] = boxes
        except Exception:
            errors.append(str(xml_file))

    return annotations, errors


def analyze_annotations(
    dataset_dir: str,
    fmt: str,
    label_dir: str | None = None,
    annotation_file: str | None = None,
    class_names: list[str] | None = None,
    image_dir: str | None = None,
) -> AnnotationStats:
    """Analyze annotations for a dataset directory.

    Args:
        dataset_dir: Root dataset directory
        fmt: Format type ('yolo', 'coco', 'voc')
        label_dir: Path to labels directory (YOLO)
        annotation_file: Path to COCO JSON file
        class_names: Class names (YOLO)
        image_dir: Path to images directory for orphan detection
    """
    stats = AnnotationStats()
    annotations: dict[str, list[BBox]] = {}

    if fmt == "yolo":
        if label_dir is None:
            # Try standard YOLO paths
            for candidate in ["labels", "labels/train", "labels/val"]:
                p = os.path.join(dataset_dir, candidate)
                if os.path.isdir(p):
                    label_dir = p
                    break
        if label_dir and os.path.isdir(label_dir):
            annotations, _ = _parse_yolo_labels(label_dir, class_names or [])

    elif fmt == "coco":
        if annotation_file is None:
            ann_dir = os.path.join(dataset_dir, "annotations")
            if os.path.isdir(ann_dir):
                jsons = list(Path(ann_dir).glob("*.json"))
                if jsons:
                    annotation_file = str(jsons[0])
        if annotation_file and os.path.isfile(annotation_file):
            annotations, _, cats = _parse_coco_annotations(annotation_file)
            if not class_names:
                class_names = list(cats.values())

    elif fmt == "voc":
        ann_dir = os.path.join(dataset_dir, "Annotations")
        if os.path.isdir(ann_dir):
            annotations, _ = _parse_voc_annotations(ann_dir)

    if not annotations:
        return stats

    # Compute stats
    all_classes: Counter[str] = Counter()
    co_occur: dict[str, Counter[str]] = defaultdict(Counter)

    for stem, boxes in annotations.items():
        stats.objects_per_image.append(len(boxes))

        classes_in_image: set[str] = set()
        for box in boxes:
            all_classes[box.class_name] += 1
            stats.bbox_widths.append(box.width)
            stats.bbox_heights.append(box.height)
            stats.bbox_areas.append(box.area)
            if box.height > 0:
                stats.bbox_aspect_ratios.append(box.width / box.height)
            stats.bbox_x_centers.append(box.x_center)
            stats.bbox_y_centers.append(box.y_center)

            # Size classification
            if box.area < 0.01:
                stats.small_count += 1
            elif box.area < 0.1:
                stats.medium_count += 1
            else:
                stats.large_count += 1

            classes_in_image.add(box.class_name)

        # Co-occurrence
        for c1 in classes_in_image:
            for c2 in classes_in_image:
                if c1 != c2:
                    co_occur[c1][c2] += 1

    stats.total_annotations = sum(all_classes.values())
    stats.class_counts = dict(all_classes.most_common())
    stats.class_names = class_names or sorted(all_classes.keys())
    stats.num_classes = len(all_classes)

    # Objects per image stats
    annotated = [stem for stem, boxes in annotations.items() if boxes]
    unannotated = [stem for stem, boxes in annotations.items() if not boxes]
    stats.annotated_images = len(annotated)
    stats.unannotated_images = len(unannotated)
    stats.total_images = len(annotations)
    stats.orphan_images = unannotated

    if stats.objects_per_image:
        stats.mean_objects_per_image = sum(stats.objects_per_image) / len(stats.objects_per_image)
        stats.max_objects_per_image = max(stats.objects_per_image)

    # Co-occurrence matrix
    stats.co_occurrence = {k: dict(v) for k, v in co_occur.items()}

    # Orphan annotation detection (annotations without matching images)
    if image_dir and os.path.isdir(image_dir):
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
        image_stems = set()
        for f in os.listdir(image_dir):
            if os.path.splitext(f)[1].lower() in img_exts:
                image_stems.add(os.path.splitext(f)[0])

        ann_stems = set(annotations.keys())
        stats.orphan_annotations = sorted(ann_stems - image_stems)[:100]

    return stats
