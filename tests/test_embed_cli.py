"""Tests for the embed CLI command and compute_embeddings function."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from imgeda.cli.app import app
from imgeda.io.manifest_io import append_records, create_manifest
from imgeda.models.manifest import ImageRecord, ManifestMeta

runner = CliRunner()

_has_torch = importlib.util.find_spec("torch") is not None
_has_open_clip = importlib.util.find_spec("open_clip") is not None
_has_umap = importlib.util.find_spec("umap") is not None

requires_embeddings = pytest.mark.skipif(
    not (_has_torch and _has_open_clip),
    reason="Requires torch and open_clip (pip install imgeda[embeddings])",
)
requires_umap = pytest.mark.skipif(
    not _has_umap,
    reason="Requires umap-learn (pip install umap-learn)",
)


@pytest.fixture
def manifest_with_images(tmp_path: Path) -> tuple[str, list[str]]:
    """Create a manifest pointing to real image files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    from PIL import Image

    paths = []
    for i in range(3):
        arr = np.random.randint(60, 200, (64, 64, 3), dtype=np.uint8)
        p = img_dir / f"img_{i}.jpg"
        Image.fromarray(arr).save(p)
        paths.append(str(p))

    manifest = tmp_path / "manifest.jsonl"
    meta = ManifestMeta(input_dir=str(img_dir), created_at="now")
    create_manifest(str(manifest), meta)
    records = [
        ImageRecord(path=p, filename=Path(p).name, width=64, height=64, format="JPEG")
        for p in paths
    ]
    append_records(str(manifest), records)
    return str(manifest), paths


class TestEmbedCLI:
    """Test embed CLI using mocks for the heavy embedding functions."""

    @patch("imgeda.core.embeddings._check_deps")
    @patch("imgeda.core.embeddings.compute_embeddings")
    @patch("imgeda.core.embeddings.save_embeddings")
    @patch("imgeda.core.embeddings.find_outliers")
    def test_embed_no_plot(
        self,
        mock_outliers: MagicMock,
        mock_save: MagicMock,
        mock_compute: MagicMock,
        mock_check: MagicMock,
        manifest_with_images: tuple[str, list[str]],
        tmp_path: Path,
    ) -> None:
        manifest, paths = manifest_with_images
        n = len(paths)
        mock_compute.return_value = np.random.randn(n, 512).astype(np.float32)
        mock_outliers.return_value = np.zeros(n, dtype=np.bool_)

        result = runner.invoke(
            app,
            ["embed", "-m", manifest, "-o", str(tmp_path / "emb.npz"), "--no-plot"],
        )
        assert result.exit_code == 0
        mock_check.assert_called_once()
        mock_compute.assert_called_once()
        mock_save.assert_called_once()
        mock_outliers.assert_called_once()

    @patch("imgeda.core.embeddings._check_deps")
    @patch("imgeda.core.embeddings.compute_umap_projection")
    @patch("imgeda.core.embeddings.compute_embeddings")
    @patch("imgeda.core.embeddings.save_embeddings")
    @patch("imgeda.core.embeddings.find_outliers")
    @patch("imgeda.plotting.embeddings.plot_umap")
    def test_embed_with_plot(
        self,
        mock_plot: MagicMock,
        mock_outliers: MagicMock,
        mock_save: MagicMock,
        mock_compute: MagicMock,
        mock_umap: MagicMock,
        mock_check: MagicMock,
        manifest_with_images: tuple[str, list[str]],
        tmp_path: Path,
    ) -> None:
        manifest, paths = manifest_with_images
        n = len(paths)
        mock_compute.return_value = np.random.randn(n, 512).astype(np.float32)
        mock_outliers.return_value = np.array([True, False, False])
        mock_umap.return_value = np.random.randn(n, 2).astype(np.float32)
        mock_plot.return_value = str(tmp_path / "umap.png")

        result = runner.invoke(
            app,
            [
                "embed",
                "-m",
                manifest,
                "-o",
                str(tmp_path / "emb.npz"),
                "--plot",
                "--plot-dir",
                str(tmp_path / "plots"),
            ],
        )
        assert result.exit_code == 0
        mock_umap.assert_called_once()

    @patch("imgeda.core.embeddings._check_deps")
    @patch("imgeda.core.embeddings.compute_embeddings")
    @patch("imgeda.core.embeddings.save_embeddings")
    @patch("imgeda.core.embeddings.find_outliers")
    def test_embed_shows_outliers(
        self,
        mock_outliers: MagicMock,
        mock_save: MagicMock,
        mock_compute: MagicMock,
        mock_check: MagicMock,
        manifest_with_images: tuple[str, list[str]],
        tmp_path: Path,
    ) -> None:
        manifest, paths = manifest_with_images
        n = len(paths)
        mock_compute.return_value = np.random.randn(n, 512).astype(np.float32)
        mock_outliers.return_value = np.array([True, False, False])

        result = runner.invoke(
            app,
            ["embed", "-m", manifest, "-o", str(tmp_path / "emb.npz"), "--no-plot"],
        )
        assert result.exit_code == 0
        assert "outlier" in result.output.lower()

    @patch("imgeda.core.embeddings._check_deps", side_effect=ImportError("No torch"))
    def test_embed_missing_deps_shows_friendly_message(
        self, mock_check: MagicMock, tmp_path: Path
    ) -> None:
        manifest = tmp_path / "m.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)

        result = runner.invoke(
            app,
            ["embed", "-m", str(manifest), "-o", str(tmp_path / "emb.npz")],
        )
        assert result.exit_code == 1
        assert "pip install imgeda[embeddings]" in result.output

    def test_embed_missing_manifest(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["embed", "-m", str(tmp_path / "nope.jsonl"), "-o", str(tmp_path / "emb.npz")],
        )
        assert result.exit_code == 1

    def test_embed_empty_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "empty.jsonl"
        meta = ManifestMeta(input_dir="/data", created_at="now")
        create_manifest(str(manifest), meta)

        result = runner.invoke(
            app,
            ["embed", "-m", str(manifest), "-o", str(tmp_path / "emb.npz")],
        )
        assert result.exit_code == 1

    def test_embed_help(self) -> None:
        result = runner.invoke(app, ["embed", "--help"])
        assert result.exit_code == 0
        assert "embeddings" in result.output.lower()


class TestComputeEmbeddingsReal:
    """Test actual embedding computation — only runs when torch+open_clip are installed."""

    @requires_embeddings
    @pytest.mark.timeout(120)
    def test_compute_embeddings_small(self, tmp_path: Path) -> None:
        """Run real CLIP inference on tiny images."""
        from PIL import Image

        from imgeda.core.embeddings import compute_embeddings

        paths = []
        for i in range(2):
            arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            p = tmp_path / f"img_{i}.jpg"
            Image.fromarray(arr).save(p)
            paths.append(str(p))

        embeddings = compute_embeddings(paths, batch_size=2, device="cpu")
        assert embeddings.shape == (2, 512)
        assert embeddings.dtype == np.float32
        # Embeddings should be normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    @requires_umap
    def test_umap_projection(self) -> None:
        """Test UMAP projection."""
        from imgeda.core.embeddings import compute_umap_projection

        embeddings = np.random.randn(20, 128).astype(np.float32)
        embeddings = (embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)).astype(
            np.float32
        )
        projection = compute_umap_projection(embeddings)
        assert projection.shape == (20, 2)
        assert projection.dtype == np.float32

    @requires_embeddings
    def test_check_deps_passes(self) -> None:
        from imgeda.core.embeddings import _check_deps

        _check_deps()  # Should not raise
