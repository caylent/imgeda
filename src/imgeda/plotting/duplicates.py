"""Duplicate group visualization."""

from __future__ import annotations

from collections import Counter

from imgeda.core.duplicates import find_exact_duplicates
from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import create_figure, save_figure


def plot_duplicates(records: list[ImageRecord], config: PlotConfig) -> str:
    groups = find_exact_duplicates(records)

    fig, axes = create_figure(config)
    # Re-create with 2 subplots
    import matplotlib.pyplot as plt

    plt.close(fig)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.figsize, dpi=config.dpi)

    # Bar chart: group size distribution
    if groups:
        sizes = [len(v) for v in groups.values()]
        size_counts = Counter(sizes)
        ax1.bar(
            [str(k) for k in sorted(size_counts.keys())],
            [size_counts[k] for k in sorted(size_counts.keys())],
            color="steelblue",
        )
        ax1.set_xlabel("Group Size")
        ax1.set_ylabel("Number of Groups")
        ax1.set_title("Duplicate Group Sizes")
    else:
        ax1.text(0.5, 0.5, "No duplicates found", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Duplicate Group Sizes")

    # Pie chart: unique vs duplicate
    total = len(records)
    dup_count = sum(len(v) - 1 for v in groups.values())  # excess copies
    unique_count = total - dup_count

    ax2.pie(
        [unique_count, dup_count],
        labels=[f"Unique ({unique_count:,})", f"Duplicates ({dup_count:,})"],
        colors=["#51cf66", "#ff6b6b"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax2.set_title("Unique vs Duplicate Images")

    fig.tight_layout()
    return save_figure(fig, "duplicates", config)
