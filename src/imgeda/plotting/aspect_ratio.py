"""Aspect ratio histogram."""

from __future__ import annotations

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import COLORS, create_figure, prepare_records, save_figure, tufte_axes

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

    num_bins = min(60, max(20, len(ratios) // 5)) if ratios else 20
    ax.hist(
        ratios,
        bins=num_bins,
        color=COLORS["primary"],
        edgecolor="none",
        alpha=0.85,
    )

    y_max = ax.get_ylim()[1]
    y_fracs = [0.94, 0.82, 0.70, 0.58]
    for i, (label, val) in enumerate(COMMON_RATIOS.items()):
        ax.axvline(val, color=COLORS["light"], linestyle=":", alpha=0.5, linewidth=0.8)
        ax.text(
            val + 0.03,
            y_max * y_fracs[i],
            label,
            fontsize=10,
            fontstyle="italic",
            color=COLORS["text_secondary"],
            va="center",
        )

    ax.set_xlabel("Aspect Ratio (width / height)")
    ax.set_ylabel("Count")
    ax.set_title(f"Aspect Ratio Distribution  ({len(recs):,} images)")
    tufte_axes(ax)
    fig.tight_layout()

    return save_figure(fig, "aspect_ratio", config)
