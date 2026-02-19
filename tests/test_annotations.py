"""Tests for annotation analysis (YOLO, COCO, VOC)."""

from __future__ import annotations

import json
from pathlib import Path

from imgeda.core.annotations import (
    AnnotationStats,
    BBox,
    _parse_coco_annotations,
    _parse_voc_annotations,
    _parse_yolo_labels,
    analyze_annotations,
)


class TestBBox:
    def test_defaults(self) -> None:
        box = BBox()
        assert box.class_name == ""
        assert box.area == 0.0


class TestAnnotationStats:
    def test_to_dict(self) -> None:
        stats = AnnotationStats(total_images=10, total_annotations=50, num_classes=3)
        d = stats.to_dict()
        assert d["total_images"] == 10
        assert d["total_annotations"] == 50
        assert d["num_classes"] == 3

    def test_to_dict_truncates_orphans(self) -> None:
        stats = AnnotationStats(orphan_images=[f"img_{i}" for i in range(50)])
        d = stats.to_dict()
        assert len(d["orphan_images"]) == 20


class TestParseYoloLabels:
    def test_basic_parsing(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "img_001.txt").write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.8 0.1 0.1\n")
        (label_dir / "img_002.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        annotations, errors = _parse_yolo_labels(str(label_dir), ["cat", "dog"])
        assert len(annotations) == 2
        assert len(annotations["img_001"]) == 2
        assert annotations["img_001"][0].class_name == "cat"
        assert annotations["img_001"][1].class_name == "dog"
        assert len(errors) == 0

    def test_unknown_class_id(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "img.txt").write_text("5 0.5 0.5 0.3 0.4\n")

        annotations, _ = _parse_yolo_labels(str(label_dir), ["cat", "dog"])
        assert annotations["img"][0].class_name == "5"

    def test_empty_file(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "empty.txt").write_text("")

        annotations, _ = _parse_yolo_labels(str(label_dir), [])
        assert annotations["empty"] == []

    def test_short_lines_skipped(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "bad.txt").write_text("0 0.5\n0 0.5 0.5 0.3 0.4\n")

        annotations, _ = _parse_yolo_labels(str(label_dir), ["x"])
        assert len(annotations["bad"]) == 1

    def test_area_computed(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n")

        annotations, _ = _parse_yolo_labels(str(label_dir), ["x"])
        assert abs(annotations["a"][0].area - 0.06) < 1e-6


class TestParseCocoAnnotations:
    def test_basic_parsing(self, tmp_path: Path) -> None:
        coco = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "b.jpg", "width": 800, "height": 600},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 200]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 50, 50]},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [0, 0, 800, 600]},
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
            ],
        }
        json_path = tmp_path / "instances.json"
        json_path.write_text(json.dumps(coco))

        annotations, errors, categories = _parse_coco_annotations(str(json_path))
        assert len(errors) == 0
        assert len(annotations) == 2
        assert len(annotations["a.jpg"]) == 2
        assert len(annotations["b.jpg"]) == 1
        assert categories[1] == "cat"
        assert categories[2] == "dog"

    def test_normalizes_coordinates(self, tmp_path: Path) -> None:
        coco = {
            "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            ],
            "categories": [{"id": 1, "name": "obj"}],
        }
        json_path = tmp_path / "coco.json"
        json_path.write_text(json.dumps(coco))

        annotations, _, _ = _parse_coco_annotations(str(json_path))
        box = annotations["img.jpg"][0]
        assert abs(box.width - 0.3) < 1e-6
        assert abs(box.height - 0.4) < 1e-6

    def test_empty_annotations(self, tmp_path: Path) -> None:
        coco = {
            "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [],
            "categories": [],
        }
        json_path = tmp_path / "empty.json"
        json_path.write_text(json.dumps(coco))

        annotations, _, _ = _parse_coco_annotations(str(json_path))
        assert annotations["img.jpg"] == []


class TestParseVocAnnotations:
    def test_basic_parsing(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        xml_content = """<annotation>
            <filename>img_001.jpg</filename>
            <size><width>640</width><height>480</height></size>
            <object>
                <name>cat</name>
                <bndbox>
                    <xmin>10</xmin><ymin>20</ymin>
                    <xmax>110</xmax><ymax>220</ymax>
                </bndbox>
            </object>
            <object>
                <name>dog</name>
                <bndbox>
                    <xmin>200</xmin><ymin>100</ymin>
                    <xmax>400</xmax><ymax>300</ymax>
                </bndbox>
            </object>
        </annotation>"""
        (ann_dir / "img_001.xml").write_text(xml_content)

        annotations, errors = _parse_voc_annotations(str(ann_dir))
        assert len(errors) == 0
        assert "img_001.jpg" in annotations
        assert len(annotations["img_001.jpg"]) == 2
        assert annotations["img_001.jpg"][0].class_name == "cat"

    def test_normalizes_coordinates(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        xml_content = """<annotation>
            <filename>img.jpg</filename>
            <size><width>100</width><height>100</height></size>
            <object>
                <name>obj</name>
                <bndbox>
                    <xmin>0</xmin><ymin>0</ymin>
                    <xmax>50</xmax><ymax>50</ymax>
                </bndbox>
            </object>
        </annotation>"""
        (ann_dir / "img.xml").write_text(xml_content)

        annotations, _ = _parse_voc_annotations(str(ann_dir))
        box = annotations["img.jpg"][0]
        assert abs(box.width - 0.5) < 1e-6
        assert abs(box.height - 0.5) < 1e-6
        assert abs(box.area - 0.25) < 1e-6


class TestMalformedInputs:
    """Tests for robustness against malformed annotation files."""

    def test_yolo_binary_garbage(self, tmp_path: Path) -> None:
        """Binary garbage in YOLO label files should be handled gracefully."""
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "garbage.txt").write_bytes(b"\x00\xff\x80\xfe" * 100)

        annotations, errors = _parse_yolo_labels(str(label_dir), ["x"])
        # Should not crash — error or empty boxes
        assert "garbage" in annotations or len(errors) > 0

    def test_yolo_non_numeric_values(self, tmp_path: Path) -> None:
        """Non-numeric values in YOLO labels should be handled."""
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "bad.txt").write_text("cat 0.5 0.5 abc 0.4\n")

        annotations, errors = _parse_yolo_labels(str(label_dir), ["x"])
        # Should not crash
        assert "bad" in annotations or len(errors) > 0

    def test_yolo_negative_coords(self, tmp_path: Path) -> None:
        """Negative coordinates should be parsed (they're technically valid in YOLO)."""
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "neg.txt").write_text("0 -0.1 0.5 0.3 0.4\n")

        annotations, _ = _parse_yolo_labels(str(label_dir), ["x"])
        assert len(annotations["neg"]) == 1
        assert annotations["neg"][0].x_center == -0.1

    def test_coco_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON should raise an exception in COCO parser."""
        json_path = tmp_path / "bad.json"
        json_path.write_text("{invalid json content")

        # Should raise json.JSONDecodeError
        import pytest

        with pytest.raises(Exception):
            _parse_coco_annotations(str(json_path))

    def test_coco_missing_keys(self, tmp_path: Path) -> None:
        """COCO JSON missing required keys should return empty results."""
        json_path = tmp_path / "incomplete.json"
        json_path.write_text(json.dumps({"info": "test"}))

        annotations, errors, categories = _parse_coco_annotations(str(json_path))
        assert len(annotations) == 0
        assert len(categories) == 0

    def test_coco_zero_dimensions(self, tmp_path: Path) -> None:
        """Images with zero width/height should not cause division by zero."""
        coco = {
            "images": [{"id": 1, "file_name": "zero.jpg", "width": 0, "height": 0}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            ],
            "categories": [{"id": 1, "name": "obj"}],
        }
        json_path = tmp_path / "zero.json"
        json_path.write_text(json.dumps(coco))

        annotations, _, _ = _parse_coco_annotations(str(json_path))
        box = annotations["zero.jpg"][0]
        assert box.width == 0  # Division by zero handled
        assert box.height == 0

    def test_coco_short_bbox(self, tmp_path: Path) -> None:
        """Annotations with too-short bbox arrays should be skipped."""
        coco = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20]},  # Too short
            ],
            "categories": [{"id": 1, "name": "obj"}],
        }
        json_path = tmp_path / "short_bbox.json"
        json_path.write_text(json.dumps(coco))

        annotations, _, _ = _parse_coco_annotations(str(json_path))
        assert len(annotations["a.jpg"]) == 0

    def test_coco_orphan_annotation(self, tmp_path: Path) -> None:
        """Annotations referencing non-existent images should be skipped."""
        coco = {
            "images": [{"id": 1, "file_name": "exists.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 999, "category_id": 1, "bbox": [10, 20, 30, 40]},
            ],
            "categories": [{"id": 1, "name": "obj"}],
        }
        json_path = tmp_path / "orphan.json"
        json_path.write_text(json.dumps(coco))

        annotations, _, _ = _parse_coco_annotations(str(json_path))
        assert len(annotations["exists.jpg"]) == 0

    def test_voc_truncated_xml(self, tmp_path: Path) -> None:
        """Truncated XML should be handled without crash."""
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        (ann_dir / "truncated.xml").write_text("<annotation><filename>test.jpg</filename>")

        annotations, errors = _parse_voc_annotations(str(ann_dir))
        # Should not crash — either errors or partial parse
        assert len(errors) > 0 or "test.jpg" in annotations

    def test_voc_binary_garbage(self, tmp_path: Path) -> None:
        """Binary garbage in XML files should be handled."""
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        (ann_dir / "garbage.xml").write_bytes(b"\x00\xff\x80\xfe" * 50)

        annotations, errors = _parse_voc_annotations(str(ann_dir))
        assert len(errors) > 0

    def test_voc_missing_bndbox(self, tmp_path: Path) -> None:
        """VOC objects without bndbox should be skipped."""
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        xml_content = """<annotation>
            <filename>img.jpg</filename>
            <size><width>100</width><height>100</height></size>
            <object>
                <name>cat</name>
            </object>
        </annotation>"""
        (ann_dir / "img.xml").write_text(xml_content)

        annotations, errors = _parse_voc_annotations(str(ann_dir))
        assert len(errors) == 0
        assert len(annotations["img.jpg"]) == 0

    def test_voc_missing_size(self, tmp_path: Path) -> None:
        """VOC annotations without size element should use defaults."""
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        xml_content = """<annotation>
            <filename>img.jpg</filename>
            <object>
                <name>cat</name>
                <bndbox>
                    <xmin>0</xmin><ymin>0</ymin>
                    <xmax>50</xmax><ymax>50</ymax>
                </bndbox>
            </object>
        </annotation>"""
        (ann_dir / "img.xml").write_text(xml_content)

        annotations, errors = _parse_voc_annotations(str(ann_dir))
        assert len(errors) == 0
        # With default size of 1x1, bbox will be normalized by 1
        assert len(annotations["img.jpg"]) == 1


class TestAnalyzeAnnotations:
    def test_yolo_analysis(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "img_001.txt").write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.8 0.1 0.1\n")
        (label_dir / "img_002.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        stats = analyze_annotations(
            str(tmp_path), "yolo", label_dir=str(label_dir), class_names=["cat", "dog"]
        )
        assert stats.total_annotations == 3
        assert stats.num_classes == 2
        assert stats.annotated_images == 2
        assert stats.mean_objects_per_image == 1.5
        assert stats.max_objects_per_image == 2

    def test_coco_analysis(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        coco = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 200]},
            ],
            "categories": [{"id": 1, "name": "cat"}],
        }
        (ann_dir / "instances.json").write_text(json.dumps(coco))

        stats = analyze_annotations(str(tmp_path), "coco")
        assert stats.total_annotations == 1
        assert "cat" in stats.class_names

    def test_voc_analysis(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        xml_content = """<annotation>
            <filename>img.jpg</filename>
            <size><width>100</width><height>100</height></size>
            <object>
                <name>person</name>
                <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
            </object>
        </annotation>"""
        (ann_dir / "img.xml").write_text(xml_content)

        stats = analyze_annotations(str(tmp_path), "voc")
        assert stats.total_annotations == 1
        assert "person" in stats.class_counts

    def test_no_annotations_found(self, tmp_path: Path) -> None:
        stats = analyze_annotations(str(tmp_path), "yolo")
        assert stats.total_annotations == 0

    def test_unknown_format(self, tmp_path: Path) -> None:
        stats = analyze_annotations(str(tmp_path), "unknown_format")
        assert stats.total_annotations == 0

    def test_size_classification(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        # Small: area < 0.01, Medium: 0.01-0.1, Large: >= 0.1
        (label_dir / "img.txt").write_text(
            "0 0.5 0.5 0.05 0.05\n"  # area = 0.0025 (small)
            "0 0.5 0.5 0.2 0.2\n"  # area = 0.04 (medium)
            "0 0.5 0.5 0.5 0.5\n"  # area = 0.25 (large)
        )

        stats = analyze_annotations(
            str(tmp_path), "yolo", label_dir=str(label_dir), class_names=["x"]
        )
        assert stats.small_count == 1
        assert stats.medium_count == 1
        assert stats.large_count == 1

    def test_co_occurrence(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        # Two classes in same image
        (label_dir / "img.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")

        stats = analyze_annotations(
            str(tmp_path), "yolo", label_dir=str(label_dir), class_names=["cat", "dog"]
        )
        assert "cat" in stats.co_occurrence
        assert "dog" in stats.co_occurrence["cat"]
