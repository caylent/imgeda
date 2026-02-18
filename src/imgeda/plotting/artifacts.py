"""Corner/border artifact analysis plot."""

from __future__ import annotations

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import COLORS, create_figure, prepare_records, save_figure


def plot_artifacts(records: list[ImageRecord], config: PlotConfig) -> str:
    recs = [r for r in prepare_records(records, config) if r.corner_stats]
    deltas = [r.corner_stats.delta for r in recs]  # type: ignore[union-attr]
    threshold = config.artifact_threshold

    fig, ax = create_figure(config)

    ax.hist(deltas, bins=60, color=COLORS["primary"], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(
        threshold,
        color=COLORS["danger"],
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label=f"Threshold ({threshold:.0f})",
    )

    artifact_count = sum(1 for d in deltas if d > threshold)
    y_max = ax.get_ylim()[1]
    ax.annotate(
        f"{artifact_count:,} flagged",
        xy=(threshold, y_max * 0.85),
        xytext=(threshold * 1.3 if threshold > 0 else 10, y_max * 0.85),
        fontsize=13,
        fontweight="bold",
        color=COLORS["danger"],
        va="center",
        arrowprops=dict(arrowstyle="-|>", color=COLORS["danger"], lw=1.2),
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=COLORS["danger"], alpha=0.9, lw=1.2),
    )

    ax.set_xlabel("Corner\u2013Center Brightness Delta")
    ax.set_ylabel("Count")
    ax.set_title(f"Border Artifact Analysis  ({len(recs):,} images)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    return save_figure(fig, "artifacts", config)
