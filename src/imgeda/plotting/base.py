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

# Refined, muted palette — professional and accessible
COLORS = {
    "primary": "#4361ee",
    "secondary": "#f77f00",
    "success": "#2a9d8f",
    "danger": "#e63946",
    "neutral": "#6c757d",
    "light": "#adb5bd",
    "channel_r": "#e63946",
    "channel_g": "#2a9d8f",
    "channel_b": "#4361ee",
    "bg_accent": "#f8f9fa",
}


def apply_theme() -> None:
    """Apply consistent professional theme to all plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.titlepad": 16,
            "axes.labelsize": 13,
            "axes.labelpad": 8,
            "axes.labelweight": "medium",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#dee2e6",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#dee2e6",
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
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
    fig.savefig(path, bbox_inches="tight", dpi=config.dpi, facecolor="white")
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
