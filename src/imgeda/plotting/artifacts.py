"""Corner/border artifact analysis plot."""

from __future__ import annotations

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import (
    COLORS,
    create_figure,
    direct_label,
    prepare_records,
    save_figure,
    tufte_axes,
)


def plot_artifacts(records: list[ImageRecord], config: PlotConfig) -> str:
    recs = [r for r in prepare_records(records, config) if r.corner_stats]
    deltas = [r.corner_stats.delta for r in recs]  # type: ignore[union-attr]
    threshold = config.artifact_threshold

    fig, ax = create_figure(config)

    ax.hist(deltas, bins=60, color=COLORS["primary"], edgecolor="none", alpha=0.85)
    ax.axvline(
        threshold,
        color=COLORS["highlight"],
        linestyle="--",
        alpha=0.8,
        linewidth=1.2,
    )

    artifact_count = sum(1 for d in deltas if d > threshold)
    y_max = ax.get_ylim()[1]
    direct_label(
        ax,
        threshold * 1.1 if threshold > 0 else 5,
        y_max * 0.85,
        f"Threshold ({threshold:.0f}) — {artifact_count:,} flagged",
        color=COLORS["highlight"],
    )

    ax.set_xlabel("Corner\u2013Center Brightness Delta")
    ax.set_ylabel("Count")
    ax.set_title(f"Border Artifact Analysis  ({len(recs):,} images)")
    tufte_axes(ax)
    fig.tight_layout()

    return save_figure(fig, "artifacts", config)
