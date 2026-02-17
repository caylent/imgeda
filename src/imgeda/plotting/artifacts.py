"""Corner/border artifact analysis plot."""

from __future__ import annotations

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import create_figure, prepare_records, save_figure


def plot_artifacts(records: list[ImageRecord], config: PlotConfig) -> str:
    recs = [r for r in prepare_records(records, config) if r.corner_stats]
    deltas = [r.corner_stats.delta for r in recs]  # type: ignore[union-attr]
    threshold = config.artifact_threshold

    fig, ax = create_figure(config)

    ax.hist(deltas, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(
        threshold, color="red", linestyle="--", alpha=0.8, label=f"Threshold ({threshold:.0f})"
    )

    artifact_count = sum(1 for d in deltas if d > threshold)
    ax.annotate(
        f"{artifact_count:,} images above threshold",
        xy=(threshold, ax.get_ylim()[1] * 0.8),
        fontsize=9,
        color="red",
    )

    ax.set_xlabel("Corner-Center Brightness Delta")
    ax.set_ylabel("Count")
    ax.set_title(f"Border Artifact Analysis ({len(recs):,} images)")
    ax.legend()
    fig.tight_layout()

    return save_figure(fig, "artifacts", config)
