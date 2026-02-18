"""Tests for dataset format detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from imgeda.core.format_detector import DatasetInfo, detect_format


def _create_image(path: Path, w: int = 100, h: int = 100) -> None:
    """Create a small test image at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


class TestYoloDetection:
    def test_yolo_with_data_yaml(self, tmp_path: Path) -> None:
        # Create data.yaml
        (tmp_path / "data.yaml").write_text(
            "train: images/train\n"
            "val: images/val\n"
            "nc: 3\n"
            "names: [cat, dog, bird]\n"
        )
        # Create images dirs
        for split in ("train", "val"):
            for i in range(3):
                _create_image(tmp_path / "images" / split / f"img_{i}.jpg")
        # Create labels dir
        (tmp_path / "labels" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "train" / "img_0.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        info = detect_format(str(tmp_path))
        assert info.format == "yolo"
        assert "train" in info.splits
        assert "val" in info.splits
        assert info.splits["train"] == 3
        assert info.splits["val"] == 3
        assert info.num_classes == 3
        assert info.class_names == ["cat", "dog", "bird"]
        assert info.annotations_path is not None
        assert info.num_images == 6

    def test_yolo_with_list_names(self, tmp_path: Path) -> None:
        (tmp_path / "data.yaml").write_text(
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  - cat\n"
            "  - dog\n"
            "  - bird\n"
        )
        for split in ("train", "val"):
            _create_image(tmp_path / "images" / split / "img_0.jpg")
        (tmp_path / "labels").mkdir()

        info = detect_format(str(tmp_path))
        assert info.format == "yolo"
        assert info.num_classes == 3
        assert info.class_names == ["cat", "dog", "bird"]

    def test_yolo_no_images(self, tmp_path: Path) -> None:
        """data.yaml exists but no images — still detects as YOLO."""
        (tmp_path / "data.yaml").write_text("nc: 2\nnames: [a, b]\n")
        info = detect_format(str(tmp_path))
        assert info.format == "yolo"
        assert info.num_images == 0


class TestCocoDetection:
    def test_coco_format(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        coco_data = {
            "images": [
                {"id": 1, "file_name": "img_0.jpg"},
                {"id": 2, "file_name": "img_1.jpg"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1},
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
            ],
        }
        (ann_dir / "instances_train.json").write_text(json.dumps(coco_data))

        for i in range(3):
            _create_image(img_dir / f"img_{i}.jpg")

        info = detect_format(str(tmp_path))
        assert info.format == "coco"
        assert info.num_classes == 2
        assert info.class_names == ["cat", "dog"]
        assert info.annotations_path is not None
        assert "train" in info.splits
        assert info.splits["train"] == 2

    def test_coco_no_categories(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        coco_data = {"images": [{"id": 1}], "annotations": [{"id": 1}]}
        (ann_dir / "data.json").write_text(json.dumps(coco_data))

        info = detect_format(str(tmp_path))
        assert info.format == "coco"
        assert info.num_classes is None

    def test_non_coco_json_skipped(self, tmp_path: Path) -> None:
        """JSON without COCO keys should not match."""
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        (ann_dir / "config.json").write_text(json.dumps({"key": "value"}))

        info = detect_format(str(tmp_path))
        assert info.format != "coco"


class TestVocDetection:
    def test_voc_format(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        img_dir = tmp_path / "JPEGImages"
        img_dir.mkdir()

        (ann_dir / "img_0.xml").write_text("<annotation><object></object></annotation>")
        (ann_dir / "img_1.xml").write_text("<annotation><object></object></annotation>")

        for i in range(5):
            _create_image(img_dir / f"img_{i}.jpg")

        info = detect_format(str(tmp_path))
        assert info.format == "voc"
        assert info.num_images == 5
        assert info.annotations_path is not None

    def test_voc_with_imagesets(self, tmp_path: Path) -> None:
        (tmp_path / "Annotations").mkdir()
        (tmp_path / "JPEGImages").mkdir()
        (tmp_path / "Annotations" / "a.xml").write_text("<annotation/>")
        _create_image(tmp_path / "JPEGImages" / "a.jpg")

        imagesets = tmp_path / "ImageSets" / "Main"
        imagesets.mkdir(parents=True)
        (imagesets / "train.txt").write_text("img_0\nimg_1\nimg_2\n")
        (imagesets / "val.txt").write_text("img_3\nimg_4\n")

        info = detect_format(str(tmp_path))
        assert info.format == "voc"
        assert info.splits.get("train") == 3
        assert info.splits.get("val") == 2

    def test_voc_needs_both_dirs(self, tmp_path: Path) -> None:
        """Only Annotations/ without JPEGImages/ should not match VOC."""
        (tmp_path / "Annotations").mkdir()
        (tmp_path / "Annotations" / "a.xml").write_text("<annotation/>")

        info = detect_format(str(tmp_path))
        assert info.format != "voc"


class TestClassificationDetection:
    def test_classification_format(self, tmp_path: Path) -> None:
        for cls in ("cat", "dog", "bird", "fish"):
            for i in range(3):
                _create_image(tmp_path / cls / f"img_{i}.jpg")

        info = detect_format(str(tmp_path))
        assert info.format == "classification"
        assert info.num_classes == 4
        assert info.class_names is not None
        assert "cat" in info.class_names
        assert info.num_images == 12

    def test_classification_needs_3_plus_subdirs(self, tmp_path: Path) -> None:
        """Only 2 subdirs should not match classification."""
        for cls in ("cat", "dog"):
            _create_image(tmp_path / cls / "img_0.jpg")

        info = detect_format(str(tmp_path))
        assert info.format != "classification"

    def test_classification_skipped_with_labels_dir(self, tmp_path: Path) -> None:
        """If labels/ exists, should not match classification (probably YOLO)."""
        for cls in ("cat", "dog", "bird", "fish"):
            _create_image(tmp_path / cls / "img_0.jpg")
        (tmp_path / "labels").mkdir()

        info = detect_format(str(tmp_path))
        assert info.format != "classification"


class TestFlatDetection:
    def test_flat_format(self, tmp_path: Path) -> None:
        for i in range(5):
            _create_image(tmp_path / f"img_{i}.jpg")

        info = detect_format(str(tmp_path))
        assert info.format == "flat"
        assert info.num_images == 5

    def test_empty_directory(self, tmp_path: Path) -> None:
        info = detect_format(str(tmp_path))
        assert info.format == "flat"
        assert info.num_images == 0
        assert info.estimated_size_bytes == 0

    def test_flat_with_subdirs(self, tmp_path: Path) -> None:
        """Only 1-2 subdirs with images -> flat, not classification."""
        _create_image(tmp_path / "img_0.jpg")
        _create_image(tmp_path / "subdir" / "img_1.jpg")

        info = detect_format(str(tmp_path))
        assert info.format == "flat"
        assert info.num_images == 2


class TestDatasetInfo:
    def test_dataclass_defaults(self) -> None:
        info = DatasetInfo(
            format="flat",
            image_dirs=["/tmp"],
            num_images=0,
            estimated_size_bytes=0,
            splits={},
        )
        assert info.num_classes is None
        assert info.class_names is None
        assert info.annotations_path is None
        assert info.extra == {}
