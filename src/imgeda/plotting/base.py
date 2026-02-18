"""Headless matplotlib setup, figure helpers, sampling."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from imgeda.models.config import PlotConfig  # noqa: E402
from imgeda.models.manifest import ImageRecord  # noqa: E402

COLORS = {
    "primary": "#2563eb",
    "secondary": "#f59e0b",
    "success": "#10b981",
    "danger": "#ef4444",
    "neutral": "#6b7280",
    "channel_r": "#ef4444",
    "channel_g": "#10b981",
    "channel_b": "#3b82f6",
}


def apply_theme() -> None:
    """Apply consistent professional theme to all plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 9,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "grid.alpha": 0.3,
        }
    )


def create_figure(config: PlotConfig) -> tuple[Figure, Any]:
    apply_theme()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    return fig, ax


def save_figure(fig: Figure, name: str, config: PlotConfig) -> str:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{config.format}"
    fig.savefig(path, bbox_inches="tight", dpi=config.dpi)
    plt.close(fig)
    return str(path)


def sample_records(records: list[ImageRecord], max_samples: int | None = None) -> list[ImageRecord]:
    """Sample records if dataset is too large for plotting."""
    if max_samples is None or len(records) <= max_samples:
        return records
    return random.sample(records, max_samples)


def valid_records(records: list[ImageRecord]) -> list[ImageRecord]:
    """Filter to non-corrupt records."""
    return [r for r in records if not r.is_corrupt]


def prepare_records(records: list[ImageRecord], config: PlotConfig) -> list[ImageRecord]:
    """Filter to valid records and apply sampling from config."""
    recs = valid_records(records)
    return sample_records(recs, config.sample)
