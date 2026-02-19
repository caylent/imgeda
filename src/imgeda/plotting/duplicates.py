"""Duplicate group visualization."""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt

from imgeda.core.duplicates import find_exact_duplicates
from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import COLORS, apply_theme, save_figure, tufte_axes


def plot_duplicates(records: list[ImageRecord], config: PlotConfig) -> str:
    groups = find_exact_duplicates(records)

    apply_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.figsize, dpi=config.dpi)

    # Left panel: group size distribution
    if groups:
        sizes = [len(v) for v in groups.values()]
        size_counts = Counter(sizes)
        sorted_keys = sorted(size_counts.keys())
        bars = ax1.bar(
            [str(k) for k in sorted_keys],
            [size_counts[k] for k in sorted_keys],
            color=COLORS["primary"],
            alpha=0.85,
            edgecolor="none",
        )
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.3,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=COLORS["text_secondary"],
            )
        ax1.set_xlabel("Group Size")
        ax1.set_ylabel("Number of Groups")
        ax1.set_title("Duplicate Group Sizes")
    else:
        ax1.text(
            0.5,
            0.5,
            "No duplicates found",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
            color=COLORS["text_secondary"],
            fontstyle="italic",
        )
        ax1.set_title("Duplicate Group Sizes")

    # Right panel: horizontal bar chart (replaces donut/pie)
    total = len(records)
    dup_count = sum(len(v) - 1 for v in groups.values())
    unique_count = total - dup_count

    categories = ["Unique", "Duplicates"]
    values = [unique_count, dup_count]
    bar_colors = [COLORS["primary"], COLORS["highlight"]]

    bars2 = ax2.barh(categories, values, color=bar_colors, alpha=0.85, edgecolor="none")

    for bar, val in zip(bars2, values):
        pct = val / total * 100 if total > 0 else 0
        ax2.text(
            bar.get_width() + max(total * 0.01, 1),
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:,}  ({pct:.1f}%)",
            va="center",
            fontsize=10,
            color=COLORS["text_secondary"],
        )

    ax2.set_xlabel("Image Count")
    ax2.set_title("Unique vs Duplicate Images")

    tufte_axes(ax1)
    tufte_axes(ax2)
    fig.tight_layout(w_pad=3)
    return save_figure(fig, "duplicates", config)
