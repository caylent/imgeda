"""EXIF metadata visualization plots."""

from __future__ import annotations

from collections import Counter

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import (
    COLORS,
    create_figure,
    prepare_records,
    save_figure,
    tufte_axes,
)


def plot_camera_distribution(records: list[ImageRecord], config: PlotConfig) -> str:
    """Bar chart of camera make/model distribution (top 15)."""
    recs = prepare_records(records, config)
    models: Counter[str] = Counter()
    for r in recs:
        if r.camera_model:
            label = r.camera_model
            if r.camera_make and not r.camera_model.startswith(r.camera_make):
                label = f"{r.camera_make} {r.camera_model}"
            models[label] += 1

    if not models:
        fig, ax = create_figure(config)
        ax.text(0.5, 0.5, "No EXIF camera data", ha="center", va="center", transform=ax.transAxes)
        return save_figure(fig, "exif_camera", config)

    top = models.most_common(15)
    labels, counts = zip(*reversed(top))

    fig, ax = create_figure(config)
    bars = ax.barh(range(len(labels)), counts, color=COLORS["primary"], edgecolor="none")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Image Count")
    ax.set_title("Camera Distribution")

    # Direct labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            fontsize=9,
            color=COLORS["text_secondary"],
        )

    tufte_axes(ax)
    return save_figure(fig, "exif_camera", config)


def plot_focal_length(records: list[ImageRecord], config: PlotConfig) -> str:
    """Histogram of 35mm-equivalent focal lengths with zone annotations."""
    recs = prepare_records(records, config)
    focal_lengths = [
        r.focal_length_35mm
        for r in recs
        if r.focal_length_35mm is not None and r.focal_length_35mm > 0
    ]

    if not focal_lengths:
        fig, ax = create_figure(config)
        ax.text(0.5, 0.5, "No focal length data", ha="center", va="center", transform=ax.transAxes)
        return save_figure(fig, "exif_focal_length", config)

    fig, ax = create_figure(config)
    ax.hist(focal_lengths, bins=50, color=COLORS["primary"], edgecolor="none", alpha=0.85)

    # Zone annotations
    zones = [
        (0, 24, "Ultra-wide"),
        (24, 35, "Wide"),
        (35, 70, "Normal"),
        (70, 200, "Telephoto"),
    ]
    ymax = ax.get_ylim()[1]
    for lo, hi, label in zones:
        mid = (lo + hi) / 2
        if lo < max(focal_lengths) and hi > min(focal_lengths):
            ax.axvline(lo, color=COLORS["light"], linewidth=0.5, linestyle=":")
            ax.text(
                mid,
                ymax * 0.97,
                label,
                ha="center",
                fontsize=8,
                fontstyle="italic",
                color=COLORS["text_tertiary"],
                va="top",
            )

    ax.set_xlabel("Focal Length (35mm equiv.)")
    ax.set_ylabel("Count")
    ax.set_title("Focal Length Distribution")
    tufte_axes(ax)

    return save_figure(fig, "exif_focal_length", config)


def plot_iso_distribution(records: list[ImageRecord], config: PlotConfig) -> str:
    """Histogram of ISO speeds with noise risk zones."""
    recs = prepare_records(records, config)
    isos = [r.iso_speed for r in recs if r.iso_speed is not None and r.iso_speed > 0]

    if not isos:
        fig, ax = create_figure(config)
        ax.text(0.5, 0.5, "No ISO data", ha="center", va="center", transform=ax.transAxes)
        return save_figure(fig, "exif_iso", config)

    fig, ax = create_figure(config)
    ax.hist(isos, bins=50, color=COLORS["primary"], edgecolor="none", alpha=0.85)

    # High ISO warning zone
    high_iso_thresh = 3200
    high_count = sum(1 for i in isos if i >= high_iso_thresh)
    if high_count > 0:
        ax.axvspan(high_iso_thresh, max(isos) * 1.05, alpha=0.08, color=COLORS["highlight"])
        ax.text(
            high_iso_thresh,
            ax.get_ylim()[1] * 0.95,
            f"  {high_count} high-noise",
            fontsize=10,
            fontstyle="italic",
            color=COLORS["highlight"],
            va="top",
        )

    ax.set_xlabel("ISO Speed")
    ax.set_ylabel("Count")
    ax.set_title("ISO Distribution")
    tufte_axes(ax)

    return save_figure(fig, "exif_iso", config)
