"""File size distribution plot."""

from __future__ import annotations

import numpy as np

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import create_figure, prepare_records, save_figure


def plot_file_size(records: list[ImageRecord], config: PlotConfig) -> str:
    recs = prepare_records(records, config)
    sizes_kb = [r.file_size_bytes / 1024 for r in recs]

    fig, ax = create_figure(config)

    if sizes_kb:
        arr = np.array(sizes_kb)
        # Use log-space bins for proper log-scale histogram
        lo = max(arr.min(), 0.1)
        hi = arr.max()
        bins = np.logspace(np.log10(lo), np.log10(hi), 80)
        ax.hist(sizes_kb, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")

        # Annotate percentiles
        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        for val, label, color in [
            (median, f"Median: {median:.0f} KB", "green"),
            (p95, f"P95: {p95:.0f} KB", "orange"),
            (p99, f"P99: {p99:.0f} KB", "red"),
        ]:
            ax.axvline(val, color=color, linestyle="--", alpha=0.8)
            ax.annotate(label, (val, ax.get_ylim()[1] * 0.9), fontsize=8, color=color, rotation=90)

    ax.set_xlabel("File Size (KB, log scale)")
    ax.set_ylabel("Count")
    ax.set_title(f"File Size Distribution ({len(recs):,} images)")
    fig.tight_layout()

    return save_figure(fig, "file_size", config)
