"""Width x Height scatter/hexbin plot."""

from __future__ import annotations

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import COLORS, create_figure, prepare_records, save_figure, tufte_axes


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
        hb = ax.hexbin(widths, heights, gridsize=50, cmap="Greys", mincnt=1, alpha=0.85)
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
            edgecolors="none",
            zorder=3,
        )

    # Only show reference lines that fall within (or near) the data range
    if widths and heights:
        w_min, w_max = min(widths), max(widths)
        h_min, h_max = min(heights), max(heights)
        # Pad by 20% so refs near the edge still appear
        w_pad = (w_max - w_min) * 0.2 or 100
        h_pad = (h_max - h_min) * 0.2 or 100

        for label, (rw, rh) in RESOLUTION_REFS.items():
            in_w = (w_min - w_pad) <= rw <= (w_max + w_pad)
            in_h = (h_min - h_pad) <= rh <= (h_max + h_pad)
            if in_w or in_h:
                if in_w:
                    ax.axvline(rw, color=COLORS["light"], linestyle="--", alpha=0.4, linewidth=0.6)
                if in_h:
                    ax.axhline(rh, color=COLORS["light"], linestyle="--", alpha=0.4, linewidth=0.6)
                # Place label at the intersection, clamped into view
                lx = rw if in_w else w_max * 0.95
                ly = rh if in_h else h_max * 0.95
                ax.text(
                    lx,
                    ly,
                    f"  {label}",
                    fontsize=10,
                    fontstyle="italic",
                    color=COLORS["text_secondary"],
                )

    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title(f"Image Dimensions  ({len(recs):,} images)")
    tufte_axes(ax)
    fig.tight_layout()

    return save_figure(fig, "dimensions", config)
