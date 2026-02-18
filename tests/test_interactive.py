"""Tests for the interactive configuration wizard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from imgeda.cli.interactive import (
    _build_split_choices,
    _format_dataset_panel,
    _resolve_image_dirs,
    run_interactive,
)
from imgeda.core.format_detector import DatasetInfo


class TestFormatDatasetPanel:
    def test_basic_panel(self) -> None:
        info = DatasetInfo(
            format="flat",
            image_dirs=["/data"],
            num_images=100,
            estimated_size_bytes=1024000,
            splits={},
        )
        panel = _format_dataset_panel(info)
        assert panel is not None

    def test_panel_with_splits(self) -> None:
        info = DatasetInfo(
            format="yolo",
            image_dirs=["/data/images/train", "/data/images/val"],
            num_images=500,
            estimated_size_bytes=5000000,
            splits={"train": 400, "val": 100},
            num_classes=10,
            class_names=["cat", "dog", "bird"],
        )
        panel = _format_dataset_panel(info)
        assert panel is not None

    def test_panel_with_many_classes(self) -> None:
        info = DatasetInfo(
            format="coco",
            image_dirs=["/data/images"],
            num_images=1000,
            estimated_size_bytes=10000000,
            splits={},
            num_classes=20,
            class_names=["c" + str(i) for i in range(20)],
        )
        panel = _format_dataset_panel(info)
        assert panel is not None

    def test_panel_with_annotations(self) -> None:
        info = DatasetInfo(
            format="voc",
            image_dirs=["/data/JPEGImages"],
            num_images=200,
            estimated_size_bytes=2000000,
            splits={},
            annotations_path="/data/Annotations",
        )
        panel = _format_dataset_panel(info)
        assert panel is not None


class TestBuildSplitChoices:
    def test_builds_choices(self) -> None:
        info = DatasetInfo(
            format="yolo",
            image_dirs=["/data/images/train", "/data/images/val"],
            num_images=300,
            estimated_size_bytes=3000000,
            splits={"train": 200, "val": 100},
        )
        choices = _build_split_choices(info)
        assert len(choices) == 2


class TestResolveImageDirs:
    def test_no_splits(self) -> None:
        info = DatasetInfo(
            format="flat",
            image_dirs=["/data/images"],
            num_images=100,
            estimated_size_bytes=1000000,
            splits={},
        )
        dirs = _resolve_image_dirs(info, [])
        assert dirs == ["/data/images"]

    def test_with_matching_splits(self) -> None:
        info = DatasetInfo(
            format="yolo",
            image_dirs=["/data/images/train", "/data/images/val"],
            num_images=300,
            estimated_size_bytes=3000000,
            splits={"train": 200, "val": 100},
        )
        dirs = _resolve_image_dirs(info, ["train"])
        assert dirs == ["/data/images/train"]

    def test_no_matching_splits_returns_all(self) -> None:
        info = DatasetInfo(
            format="yolo",
            image_dirs=["/data/custom_train", "/data/custom_val"],
            num_images=300,
            estimated_size_bytes=3000000,
            splits={"train": 200, "val": 100},
        )
        dirs = _resolve_image_dirs(info, ["train"])
        assert dirs == ["/data/custom_train", "/data/custom_val"]


class TestRunInteractive:
    @pytest.mark.timeout(60)
    def test_happy_path(self, tmp_image_dir: Path, tmp_path: Path) -> None:
        """Full happy path: user provides directory, accepts defaults, gets scan."""
        output = str(tmp_path / "manifest.jsonl")
        with patch("imgeda.cli.interactive.questionary") as mock_q:
            # questionary.path(...).ask() -> directory
            mock_q.path.return_value.ask.return_value = str(tmp_image_dir)

            # questionary.checkbox(...).ask() -> analyses list (called once for flat format)
            mock_q.checkbox.return_value.ask.return_value = [
                "Basic metadata (dimensions, format, file size)",
                "Pixel statistics (brightness, color channels)",
                "Perceptual hashing (duplicate detection)",
                "Corner/border artifact detection",
            ]

            # questionary.text(...).ask() -> called twice: workers, output path
            mock_q.text.return_value.ask.side_effect = [
                "2",  # workers
                output,  # output path
            ]

            # questionary.confirm(...).ask() -> no report
            mock_q.confirm.return_value.ask.return_value = False

            run_interactive()

        assert Path(output).exists()

    def test_cancel_at_directory(self) -> None:
        """User cancels at directory prompt."""
        with patch("imgeda.cli.interactive.questionary") as mock_q:
            mock_q.path.return_value.ask.return_value = None
            run_interactive()  # should return without error

    def test_invalid_directory(self, tmp_path: Path) -> None:
        """User provides nonexistent directory."""
        with patch("imgeda.cli.interactive.questionary") as mock_q:
            mock_q.path.return_value.ask.return_value = str(tmp_path / "nonexistent")
            run_interactive()  # should print error and return

    def test_cancel_at_analyses(self, tmp_image_dir: Path) -> None:
        """User cancels at analyses selection."""
        with patch("imgeda.cli.interactive.questionary") as mock_q:
            mock_q.path.return_value.ask.return_value = str(tmp_image_dir)
            mock_q.checkbox.return_value.ask.return_value = None
            run_interactive()  # should return without error

    def test_cancel_at_workers(self, tmp_image_dir: Path) -> None:
        """User cancels at workers prompt."""
        with patch("imgeda.cli.interactive.questionary") as mock_q:
            mock_q.path.return_value.ask.return_value = str(tmp_image_dir)
            mock_q.checkbox.return_value.ask.return_value = [
                "Basic metadata (dimensions, format, file size)",
            ]
            mock_q.text.return_value.ask.return_value = None
            run_interactive()

    def test_cancel_at_output(self, tmp_image_dir: Path) -> None:
        """User cancels at output path prompt."""
        with patch("imgeda.cli.interactive.questionary") as mock_q:
            mock_q.path.return_value.ask.return_value = str(tmp_image_dir)
            mock_q.checkbox.return_value.ask.return_value = [
                "Basic metadata (dimensions, format, file size)",
            ]
            # First text call (workers) returns "2", second (output) returns None
            mock_q.text.return_value.ask.side_effect = ["2", None]
            run_interactive()

    def test_cancel_at_confirm(self, tmp_image_dir: Path) -> None:
        """User cancels at report confirmation prompt."""
        output_path = str(tmp_image_dir.parent / "out.jsonl")
        with patch("imgeda.cli.interactive.questionary") as mock_q:
            mock_q.path.return_value.ask.return_value = str(tmp_image_dir)
            mock_q.checkbox.return_value.ask.return_value = [
                "Basic metadata (dimensions, format, file size)",
            ]
            mock_q.text.return_value.ask.side_effect = ["2", output_path]
            mock_q.confirm.return_value.ask.return_value = None
            run_interactive()
