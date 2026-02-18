"""Duplicate group visualization."""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt

from imgeda.core.duplicates import find_exact_duplicates
from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import COLORS, apply_theme, save_figure


def plot_duplicates(records: list[ImageRecord], config: PlotConfig) -> str:
    groups = find_exact_duplicates(records)

    apply_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.figsize, dpi=config.dpi)

    # Bar chart: group size distribution
    if groups:
        sizes = [len(v) for v in groups.values()]
        size_counts = Counter(sizes)
        sorted_keys = sorted(size_counts.keys())
        bars = ax1.bar(
            [str(k) for k in sorted_keys],
            [size_counts[k] for k in sorted_keys],
            color=COLORS["primary"],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.3,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color=COLORS["neutral"],
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
            fontsize=14,
            color=COLORS["neutral"],
            fontstyle="italic",
        )
        ax1.set_title("Duplicate Group Sizes")

    # Donut chart: unique vs duplicate
    total = len(records)
    dup_count = sum(len(v) - 1 for v in groups.values())
    unique_count = total - dup_count

    wedges, texts, autotexts = ax2.pie(
        [unique_count, dup_count],
        labels=[f"Unique ({unique_count:,})", f"Duplicates ({dup_count:,})"],
        colors=[COLORS["success"], COLORS["danger"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=12),
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(12)

    ax2.set_title("Unique vs Duplicate Images")

    fig.tight_layout(w_pad=3)
    return save_figure(fig, "duplicates", config)
