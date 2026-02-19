"""Embedding visualization plots — UMAP projection, outliers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from imgeda.models.config import PlotConfig
from imgeda.plotting.base import (
    COLORS,
    apply_theme,
    save_figure,
    tufte_axes,
)

import matplotlib.pyplot as plt  # noqa: E402


def plot_umap(
    projection: NDArray[np.float32],
    config: PlotConfig,
    labels: list[str] | None = None,
    outlier_mask: NDArray[np.bool_] | None = None,
) -> str:
    """2D scatter plot of UMAP-projected embeddings."""
    apply_theme()

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if outlier_mask is not None:
        # Plot normal points
        normal = ~outlier_mask
        ax.scatter(
            projection[normal, 0],
            projection[normal, 1],
            s=6,
            alpha=0.4,
            color=COLORS["primary"],
            edgecolors="none",
            label=f"Normal ({normal.sum():,})",
        )
        # Plot outliers
        ax.scatter(
            projection[outlier_mask, 0],
            projection[outlier_mask, 1],
            s=12,
            alpha=0.8,
            color=COLORS["highlight"],
            edgecolors="none",
            label=f"Outliers ({outlier_mask.sum():,})",
        )
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            s=6,
            alpha=0.4,
            color=COLORS["primary"],
            edgecolors="none",
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Dataset Embedding Space")
    ax.set_xticks([])
    ax.set_yticks([])
    tufte_axes(ax)

    return save_figure(fig, "embedding_umap", config)
