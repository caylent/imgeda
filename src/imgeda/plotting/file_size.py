"""File size distribution plot."""

from __future__ import annotations

import numpy as np

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import COLORS, create_figure, prepare_records, save_figure


def plot_file_size(records: list[ImageRecord], config: PlotConfig) -> str:
    recs = prepare_records(records, config)
    sizes_kb = [r.file_size_bytes / 1024 for r in recs]

    fig, ax = create_figure(config)

    if sizes_kb:
        arr = np.array(sizes_kb)
        lo = max(arr.min(), 0.1)
        hi = arr.max()
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
        ax.hist(
            sizes_kb,
            bins=bins,
            color=COLORS["primary"],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.set_xscale("log")

        # Percentile annotations — staggered vertically, placed to the right
        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        y_max = ax.get_ylim()[1]

        stats = [
            (median, f"Median  {median:,.0f} KB", COLORS["success"], 0.92),
            (p95, f"P95  {p95:,.0f} KB", COLORS["secondary"], 0.76),
            (p99, f"P99  {p99:,.0f} KB", COLORS["danger"], 0.60),
        ]
        for val, label, color, y_frac in stats:
            ax.axvline(val, color=color, linestyle="--", alpha=0.7, linewidth=1.2)
            ax.annotate(
                label,
                xy=(val, y_max * y_frac),
                xytext=(val * 2.2, y_max * y_frac),
                fontsize=12,
                fontweight="bold",
                color=color,
                va="center",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
            )

    ax.set_xlabel("File Size (KB, log scale)")
    ax.set_ylabel("Count")
    ax.set_title(f"File Size Distribution  ({len(recs):,} images)")
    fig.tight_layout()

    return save_figure(fig, "file_size", config)
