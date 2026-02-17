"""Headless matplotlib setup, figure helpers, sampling."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from imgeda.models.config import PlotConfig  # noqa: E402
from imgeda.models.manifest import ImageRecord  # noqa: E402


def create_figure(config: PlotConfig) -> tuple[Figure, plt.Axes]:
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
