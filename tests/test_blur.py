"""Tests for blur detection."""

from __future__ import annotations

import numpy as np

from imgeda.core.detector import compute_blur_score


class TestComputeBlurScore:
    def test_sharp_image_high_score(self) -> None:
        """A sharp image with high-frequency content should have a high blur score."""
        # Checkerboard pattern = lots of edges
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[::2, ::2] = 255
        arr[1::2, 1::2] = 255
        score = compute_blur_score(arr)
        assert score > 1000  # Very sharp

    def test_uniform_image_low_score(self) -> None:
        """A uniform (blurry) image should have a very low score."""
        arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        score = compute_blur_score(arr)
        assert score < 1.0  # Nearly zero variance

    def test_gradient_moderate_score(self) -> None:
        """A smooth gradient should have a moderate blur score."""
        gradient = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
        arr = np.stack([gradient, gradient, gradient], axis=2)
        score = compute_blur_score(arr)
        assert 0 < score < 5000

    def test_returns_float(self) -> None:
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        score = compute_blur_score(arr)
        assert isinstance(score, float)

    def test_sharp_vs_blurry_ordering(self) -> None:
        """Sharp images should score higher than blurry ones."""
        # Sharp: random noise
        sharp = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Blurry: uniform with slight noise
        blurry = np.full((100, 100, 3), 128, dtype=np.uint8)
        blurry += np.random.randint(-2, 3, (100, 100, 3), dtype=np.int8).view(np.uint8)

        assert compute_blur_score(sharp) > compute_blur_score(blurry)
