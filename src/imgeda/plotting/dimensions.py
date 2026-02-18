"""Width x Height scatter/hexbin plot."""

from __future__ import annotations

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import COLORS, create_figure, prepare_records, save_figure


RESOLUTION_REFS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4K": (3840, 2160),
}


def plot_dimensions(records: list[ImageRecord], config: PlotConfig) -> str:
    recs = prepare_records(records, config)
    widths = [r.width for r in recs]
    heights = [r.height for r in recs]

    fig, ax = create_figure(config)

    if len(recs) > 5000:
        hb = ax.hexbin(widths, heights, gridsize=50, cmap="Blues", mincnt=1, alpha=0.85)
        fig.colorbar(hb, ax=ax, label="Count", shrink=0.8)
    else:
        from imgeda.plotting.base import sample_records

        recs_sampled = sample_records(recs, 5000)
        w = [r.width for r in recs_sampled]
        h = [r.height for r in recs_sampled]
        ax.scatter(
            w,
            h,
            alpha=0.45,
            s=18,
            c=COLORS["primary"],
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    # Reference lines — subtle and clean
    for label, (rw, rh) in RESOLUTION_REFS.items():
        ax.axvline(rw, color=COLORS["light"], linestyle="--", alpha=0.6, linewidth=0.8)
        ax.axhline(rh, color=COLORS["light"], linestyle="--", alpha=0.6, linewidth=0.8)
        ax.annotate(
            label,
            (rw, rh),
            fontsize=10,
            fontweight="bold",
            color=COLORS["neutral"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["light"], alpha=0.9),
        )

    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title(f"Image Dimensions  ({len(recs):,} images)")
    fig.tight_layout()

    return save_figure(fig, "dimensions", config)
