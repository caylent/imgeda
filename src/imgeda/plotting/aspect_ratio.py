"""Aspect ratio histogram."""

from __future__ import annotations

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import COLORS, create_figure, prepare_records, save_figure

COMMON_RATIOS = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:2": 3 / 2,
    "16:9": 16 / 9,
}


def plot_aspect_ratio(records: list[ImageRecord], config: PlotConfig) -> str:
    recs = prepare_records(records, config)
    ratios = [r.aspect_ratio for r in recs if r.aspect_ratio > 0]

    fig, ax = create_figure(config)

    num_bins = min(80, max(20, len(ratios) // 5)) if ratios else 20
    ax.hist(ratios, bins=num_bins, color=COLORS["primary"], edgecolor="white", linewidth=0.3)

    for label, val in COMMON_RATIOS.items():
        ax.axvline(val, color=COLORS["danger"], linestyle="--", alpha=0.6, linewidth=1)
        ax.text(
            val,
            ax.get_ylim()[1] * 1.02,
            label,
            fontsize=8,
            color=COLORS["danger"],
            ha="center",
            va="bottom",
            clip_on=False,
        )

    ax.set_xlabel("Aspect Ratio (width / height)")
    ax.set_ylabel("Count")
    ax.set_title(f"Aspect Ratio Distribution ({len(recs):,} images)")
    fig.tight_layout()

    return save_figure(fig, "aspect_ratio", config)
