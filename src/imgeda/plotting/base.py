"""Headless matplotlib setup, figure helpers, sampling."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from imgeda.models.config import PlotConfig  # noqa: E402
from imgeda.models.manifest import ImageRecord  # noqa: E402

# Tufte-inspired palette — muted defaults, color used with purpose
COLORS = {
    "primary": "#4c72b0",
    "secondary": "#8da0cb",
    "highlight": "#e41a1c",
    "neutral": "#555555",
    "light": "#cccccc",
    "channel_r": "#c44e52",
    "channel_g": "#55a868",
    "channel_b": "#4c72b0",
    "bg": "#fffff8",
    "text": "#333333",
    "text_secondary": "#555555",
    "text_tertiary": "#888888",
    # Backward-compat aliases
    "danger": "#e41a1c",
    "success": "#55a868",
    "bg_accent": "#fffff8",
}


def apply_theme() -> None:
    """Apply Tufte-inspired theme: serif fonts, no grid, minimal chrome."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "font.family": "serif",
            "font.serif": [
                "Palatino",
                "Georgia",
                "DejaVu Serif",
                "serif",
            ],
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "normal",
            "axes.titlepad": 14,
            "axes.labelsize": 12,
            "axes.labelpad": 8,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["light"],
            "axes.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.color": COLORS["light"],
            "ytick.color": COLORS["light"],
            "xtick.labelcolor": COLORS["text"],
            "ytick.labelcolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.dpi": 150,
        }
    )


def tufte_axes(ax: Axes) -> None:
    """Trim bottom/left spines to data range (range-frame effect)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom spine: trim to x data range
    xmin, xmax = ax.get_xlim()
    ax.spines["bottom"].set_bounds(xmin, xmax)

    # Left spine: trim to y data range
    ymin, ymax = ax.get_ylim()
    ax.spines["left"].set_bounds(ymin, ymax)


def direct_label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    color: str = COLORS["text_secondary"],
) -> None:
    """Place italic serif text directly at a data point (replaces legend)."""
    ax.text(
        x,
        y,
        text,
        fontsize=10,
        fontstyle="italic",
        color=color,
        va="center",
    )


def create_figure(config: PlotConfig) -> tuple[Figure, Any]:
    apply_theme()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    return fig, ax


def save_figure(fig: Figure, name: str, config: PlotConfig) -> str:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{config.format}"
    fig.savefig(path, bbox_inches="tight", dpi=config.dpi, facecolor=COLORS["bg"])
    plt.close(fig)
    return str(path)


def sample_records(
    records: list[ImageRecord], max_samples: int | None = None, seed: int = 42
) -> list[ImageRecord]:
    """Sample records if dataset is too large for plotting."""
    if max_samples is None or len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    return rng.sample(records, max_samples)


def valid_records(records: list[ImageRecord]) -> list[ImageRecord]:
    """Filter to non-corrupt records."""
    return [r for r in records if not r.is_corrupt]


def prepare_records(records: list[ImageRecord], config: PlotConfig) -> list[ImageRecord]:
    """Filter to valid records and apply sampling from config."""
    recs = valid_records(records)
    return sample_records(recs, config.sample, seed=config.seed)
