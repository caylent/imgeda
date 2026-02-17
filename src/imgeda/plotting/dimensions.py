"""Width x Height scatter/hexbin plot."""

from __future__ import annotations

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import create_figure, save_figure, sample_records, valid_records


RESOLUTION_REFS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4K": (3840, 2160),
}


def plot_dimensions(records: list[ImageRecord], config: PlotConfig) -> str:
    recs = valid_records(records)
    widths = [r.width for r in recs]
    heights = [r.height for r in recs]

    fig, ax = create_figure(config)

    if len(recs) > 5000:
        # Hexbin for large datasets
        hb = ax.hexbin(widths, heights, gridsize=50, cmap="YlOrRd", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Count")
    else:
        recs_sampled = sample_records(recs, 5000)
        w = [r.width for r in recs_sampled]
        h = [r.height for r in recs_sampled]
        ax.scatter(w, h, alpha=0.3, s=8, c="steelblue")

    # Reference lines
    for label, (rw, rh) in RESOLUTION_REFS.items():
        ax.axvline(rw, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axhline(rh, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.annotate(label, (rw, rh), fontsize=8, color="gray", alpha=0.7)

    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title(f"Image Dimensions ({len(recs):,} images)")
    fig.tight_layout()

    return save_figure(fig, "dimensions", config)
