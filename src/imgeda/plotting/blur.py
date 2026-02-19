"""Blur score distribution plot."""

from __future__ import annotations

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import (
    COLORS,
    create_figure,
    prepare_records,
    save_figure,
    tufte_axes,
)


def plot_blur(records: list[ImageRecord], config: PlotConfig) -> str:
    """Histogram of blur scores with threshold line."""
    recs = prepare_records(records, config)
    scores = [r.blur_score for r in recs if r.blur_score is not None]

    if not scores:
        fig, ax = create_figure(config)
        ax.text(0.5, 0.5, "No blur data", ha="center", va="center", transform=ax.transAxes)
        return save_figure(fig, "blur", config)

    fig, ax = create_figure(config)
    ax.hist(scores, bins=80, color=COLORS["primary"], edgecolor="none", alpha=0.85)

    blur_thresh = 100.0
    ax.axvline(blur_thresh, color=COLORS["highlight"], linewidth=1.2, linestyle="--")
    blurry_count = sum(1 for s in scores if s < blur_thresh)
    ax.text(
        blur_thresh,
        ax.get_ylim()[1] * 0.95,
        f"  {blurry_count} blurry",
        fontsize=10,
        fontstyle="italic",
        color=COLORS["highlight"],
        va="top",
    )

    # Shade blurry region
    ax.axvspan(0, blur_thresh, alpha=0.08, color=COLORS["highlight"])

    ax.set_xlabel("Blur Score (Laplacian Variance)")
    ax.set_ylabel("Count")
    ax.set_title("Blur Score Distribution")
    tufte_axes(ax)

    return save_figure(fig, "blur", config)
