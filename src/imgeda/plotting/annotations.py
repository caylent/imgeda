"""Annotation analysis plots — class frequency, bbox distributions, co-occurrence."""

from __future__ import annotations

import numpy as np

from imgeda.core.annotations import AnnotationStats
from imgeda.models.config import PlotConfig
from imgeda.plotting.base import (
    COLORS,
    apply_theme,
    save_figure,
    tufte_axes,
)

import matplotlib.pyplot as plt  # noqa: E402


def plot_class_frequency(stats: AnnotationStats, config: PlotConfig) -> str:
    """Horizontal bar chart of class frequencies (top 30)."""
    apply_theme()

    if not stats.class_counts:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.text(0.5, 0.5, "No class data", ha="center", va="center", transform=ax.transAxes)
        return save_figure(fig, "class_frequency", config)

    top = list(stats.class_counts.items())[:30]
    labels, counts = zip(*reversed(top))

    fig_h = max(config.figsize[1], len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(config.figsize[0], fig_h), dpi=config.dpi)

    bars = ax.barh(range(len(labels)), counts, color=COLORS["primary"], edgecolor="none")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Annotation Count")
    ax.set_title("Class Frequency Distribution")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            fontsize=8,
            color=COLORS["text_secondary"],
        )

    tufte_axes(ax)
    return save_figure(fig, "class_frequency", config)


def plot_bbox_sizes(stats: AnnotationStats, config: PlotConfig) -> str:
    """Scatter plot of bbox width vs height + size category pie chart."""
    apply_theme()

    if not stats.bbox_widths:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.text(0.5, 0.5, "No bbox data", ha="center", va="center", transform=ax.transAxes)
        return save_figure(fig, "bbox_sizes", config)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(config.figsize[0] * 1.3, config.figsize[1]), dpi=config.dpi
    )

    # Scatter: width vs height
    w = np.array(stats.bbox_widths)
    h = np.array(stats.bbox_heights)
    if len(w) > 5000:
        ax1.hexbin(w, h, gridsize=40, cmap="YlOrRd", mincnt=1)
    else:
        ax1.scatter(w, h, alpha=0.3, s=8, color=COLORS["primary"], edgecolors="none")

    ax1.set_xlabel("BBox Width (normalized)")
    ax1.set_ylabel("BBox Height (normalized)")
    ax1.set_title("Bounding Box Dimensions")
    tufte_axes(ax1)

    # Pie: small/medium/large
    sizes = [stats.small_count, stats.medium_count, stats.large_count]
    labels = ["Small (<1%)", "Medium (1-10%)", "Large (>10%)"]
    colors = [COLORS["secondary"], COLORS["primary"], COLORS["highlight"]]
    nonzero = [(s, lb, c) for s, lb, c in zip(sizes, labels, colors) if s > 0]
    if nonzero:
        s_vals, s_labels, s_colors = zip(*nonzero)
        ax2.pie(
            s_vals, labels=s_labels, colors=s_colors, autopct="%1.0f%%", textprops={"fontsize": 10}
        )
    ax2.set_title("Object Size Distribution")

    fig.tight_layout()
    return save_figure(fig, "bbox_sizes", config)


def plot_objects_per_image(stats: AnnotationStats, config: PlotConfig) -> str:
    """Histogram of objects per image."""
    apply_theme()

    if not stats.objects_per_image:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return save_figure(fig, "objects_per_image", config)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    max_obj = max(stats.objects_per_image)
    bins = min(max_obj, 50)
    ax.hist(
        stats.objects_per_image, bins=bins, color=COLORS["primary"], edgecolor="none", alpha=0.85
    )

    mean_val = stats.mean_objects_per_image
    ax.axvline(mean_val, color=COLORS["neutral"], linewidth=1.2, linestyle="--")
    ax.text(
        mean_val,
        ax.get_ylim()[1] * 0.95,
        f"  mean={mean_val:.1f}",
        fontsize=10,
        fontstyle="italic",
        color=COLORS["neutral"],
        va="top",
    )

    ax.set_xlabel("Objects per Image")
    ax.set_ylabel("Count")
    ax.set_title("Objects per Image Distribution")
    tufte_axes(ax)

    return save_figure(fig, "objects_per_image", config)


def plot_co_occurrence(stats: AnnotationStats, config: PlotConfig) -> str:
    """Heatmap of class co-occurrence matrix."""
    apply_theme()

    if not stats.co_occurrence:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.text(0.5, 0.5, "No co-occurrence data", ha="center", va="center", transform=ax.transAxes)
        return save_figure(fig, "co_occurrence", config)

    # Build matrix from top N classes
    top_classes = list(stats.class_counts.keys())[:20]
    n = len(top_classes)
    matrix = np.zeros((n, n))

    for i, c1 in enumerate(top_classes):
        for j, c2 in enumerate(top_classes):
            if c1 in stats.co_occurrence and c2 in stats.co_occurrence[c1]:
                matrix[i, j] = stats.co_occurrence[c1][c2]

    fig_size = max(config.figsize[0], n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=config.dpi)

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(top_classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(top_classes, fontsize=8)
    ax.set_title("Class Co-occurrence Matrix")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    return save_figure(fig, "co_occurrence", config)


def plot_annotation_coverage(stats: AnnotationStats, config: PlotConfig) -> str:
    """2D histogram of bbox center positions — reveals spatial bias."""
    apply_theme()

    if not stats.bbox_x_centers:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.text(0.5, 0.5, "No bbox data", ha="center", va="center", transform=ax.transAxes)
        return save_figure(fig, "annotation_coverage", config)

    fig, ax = plt.subplots(figsize=(config.figsize[1], config.figsize[1]), dpi=config.dpi)

    x = np.array(stats.bbox_x_centers)
    y = np.array(stats.bbox_y_centers)

    ax.hist2d(x, y, bins=20, cmap="YlOrRd", range=[[0, 1], [0, 1]])
    ax.set_xlabel("X Position (normalized)")
    ax.set_ylabel("Y Position (normalized)")
    ax.set_title("Annotation Coverage Heatmap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # Image coordinates: origin at top-left
    ax.set_aspect("equal")

    return save_figure(fig, "annotation_coverage", config)
