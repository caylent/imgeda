"""File size distribution plot."""

from __future__ import annotations

import math

import numpy as np

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import COLORS, create_figure, prepare_records, save_figure, tufte_axes


def _auto_unit(kb: float) -> tuple[float, str]:
    """Pick human-friendly unit for a file size in KB."""
    if kb >= 1_048_576:  # 1 GB in KB
        return kb / 1_048_576, "GB"
    if kb >= 1024:
        return kb / 1024, "MB"
    return kb, "KB"


def _format_size(kb: float) -> str:
    """Format a KB value with auto-selected unit."""
    val, unit = _auto_unit(kb)
    if val >= 100:
        return f"{val:,.0f} {unit}"
    if val >= 10:
        return f"{val:,.1f} {unit}"
    return f"{val:,.2f} {unit}"


def _adaptive_bins(n: int, spread_ratio: float) -> int:
    """Pick bin count from data size and spread.

    spread_ratio = max/min — larger spread needs more bins.
    """
    base = max(15, min(80, int(math.sqrt(n) * 1.5)))
    # Widen for high-spread data (e.g. 1 KB to 100 MB)
    if spread_ratio > 1000:
        base = min(100, int(base * 1.3))
    return base


def plot_file_size(records: list[ImageRecord], config: PlotConfig) -> str:
    recs = prepare_records(records, config)
    sizes_kb = [r.file_size_bytes / 1024 for r in recs]

    fig, ax = create_figure(config)

    if sizes_kb:
        arr = np.array(sizes_kb)
        lo = max(float(arr.min()), 0.1)
        hi = float(arr.max())
        spread = hi / lo if lo > 0 else 1.0
        num_bins = _adaptive_bins(len(arr), spread)
        bins = np.logspace(np.log10(lo), np.log10(hi), num_bins)

        ax.hist(
            sizes_kb,
            bins=bins,
            color=COLORS["primary"],
            edgecolor="none",
            alpha=0.85,
        )
        ax.set_xscale("log")

        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        y_max = ax.get_ylim()[1]

        stats = [
            (median, f"Median  {_format_size(median)}", COLORS["highlight"], 0.92),
            (p95, f"P95  {_format_size(p95)}", COLORS["text_secondary"], 0.76),
            (p99, f"P99  {_format_size(p99)}", COLORS["text_secondary"], 0.60),
        ]

        # Nudge labels apart if percentiles are too close on log scale
        placed: list[tuple[float, float]] = []
        for val, label, color, y_frac in stats:
            x_pos = val * 1.15
            y_pos = y_max * y_frac

            # Check for overlap with already-placed labels
            for px, py in placed:
                # "too close" on log scale = ratio < 2x, and y within 12%
                if 0.5 < (x_pos / px) < 2.0 and abs(y_pos - py) < y_max * 0.12:
                    y_pos = py - y_max * 0.13

            ax.axvline(val, color=color, linestyle="--", alpha=0.5, linewidth=0.8)
            ax.text(
                x_pos,
                y_pos,
                label,
                fontsize=10,
                fontstyle="italic",
                color=color,
                va="center",
            )
            placed.append((x_pos, y_pos))

        # Auto-select x-axis label unit
        _, dominant_unit = _auto_unit(median)
        ax.set_xlabel(f"File Size ({dominant_unit}, log scale)")
    else:
        ax.set_xlabel("File Size (log scale)")

    ax.set_ylabel("Count")
    ax.set_title(f"File Size Distribution  ({len(recs):,} images)")
    tufte_axes(ax)
    fig.tight_layout()

    return save_figure(fig, "file_size", config)
