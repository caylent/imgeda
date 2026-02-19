"""Tests for embedding utilities (non-CLIP parts that don't require torch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from imgeda.core.embeddings import (
    find_outliers,
    find_semantic_duplicates,
    load_embeddings,
    save_embeddings,
)


class TestFindOutliers:
    def test_cluster_with_outlier(self) -> None:
        """An obvious outlier should be flagged."""
        # 49 points near origin, 1 far away
        embeddings = np.random.randn(50, 128).astype(np.float32) * 0.1
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Make one outlier
        embeddings[0] = -embeddings[0]

        mask = find_outliers(embeddings, threshold_percentile=5.0)
        assert mask.shape == (50,)
        assert mask.dtype == np.bool_
        # The outlier should be flagged (at 5th percentile threshold)
        assert mask.sum() >= 1

    def test_uniform_cluster(self) -> None:
        """A tight cluster should have few outliers."""
        embeddings = np.ones((20, 64), dtype=np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        mask = find_outliers(embeddings, threshold_percentile=5.0)
        # With identical vectors, the percentile threshold should flag ~5%
        assert mask.sum() <= 2

    def test_returns_bool_mask(self) -> None:
        embeddings = np.random.randn(10, 32).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        mask = find_outliers(embeddings)
        assert mask.dtype == np.bool_
        assert mask.shape == (10,)


class TestFindSemanticDuplicates:
    def test_identical_vectors(self) -> None:
        """Identical vectors should be flagged as duplicates."""
        embeddings = np.random.randn(5, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings[2] = embeddings[0]  # Make idx 2 a duplicate of idx 0

        dupes = find_semantic_duplicates(embeddings, threshold=0.99)
        assert len(dupes) >= 1
        pairs = [(a, b) for a, b, _ in dupes]
        assert (0, 2) in pairs

    def test_no_duplicates(self) -> None:
        """Orthogonal vectors should not be flagged."""
        n = 10
        # Pad to 64 dims
        padded = np.zeros((n, 64), dtype=np.float32)
        padded[:, :n] = np.eye(n, dtype=np.float32)
        padded = padded / np.linalg.norm(padded, axis=1, keepdims=True)

        dupes = find_semantic_duplicates(padded, threshold=0.95)
        assert len(dupes) == 0

    def test_returns_similarity(self) -> None:
        embeddings = np.random.randn(5, 32).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings[1] = embeddings[0]

        dupes = find_semantic_duplicates(embeddings, threshold=0.99)
        for idx_a, idx_b, sim in dupes:
            assert 0 <= sim <= 1.01  # allow small float imprecision
            assert idx_a < idx_b


class TestSaveLoadEmbeddings:
    def test_roundtrip(self, tmp_path: Path) -> None:
        embeddings = np.random.randn(20, 128).astype(np.float32)
        paths = [f"/img_{i}.jpg" for i in range(20)]
        out = str(tmp_path / "embeddings.npz")

        save_embeddings(embeddings, paths, out)
        assert Path(out).exists()

        loaded_emb, loaded_paths = load_embeddings(out)
        np.testing.assert_array_almost_equal(loaded_emb, embeddings)
        assert loaded_paths == paths

    def test_empty_embeddings(self, tmp_path: Path) -> None:
        embeddings = np.zeros((0, 128), dtype=np.float32)
        paths: list[str] = []
        out = str(tmp_path / "empty.npz")

        save_embeddings(embeddings, paths, out)
        loaded_emb, loaded_paths = load_embeddings(out)
        assert loaded_emb.shape == (0, 128)
        assert loaded_paths == []
