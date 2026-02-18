"""Brightness + channel distribution plots."""

from __future__ import annotations

from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import COLORS, create_figure, prepare_records, save_figure


def plot_brightness(records: list[ImageRecord], config: PlotConfig) -> str:
    """Histogram of mean brightness with shaded dark/normal/overexposed regions."""
    recs = [r for r in prepare_records(records, config) if r.pixel_stats]
    brightness = [r.pixel_stats.mean_brightness for r in recs]  # type: ignore[union-attr]

    fig, ax = create_figure(config)

    ax.hist(brightness, bins=80, color=COLORS["primary"], edgecolor="white", linewidth=0.3)

    # Shaded regions
    ax.axvspan(0, 40, alpha=0.07, color="navy", label="Dark (<40)")
    ax.axvspan(220, 255, alpha=0.07, color=COLORS["danger"], label="Overexposed (>220)")

    ax.set_xlabel("Mean Brightness")
    ax.set_ylabel("Count")
    ax.set_title(f"Brightness Distribution ({len(recs):,} images)")
    ax.legend()
    fig.tight_layout()

    return save_figure(fig, "brightness", config)


def plot_channels(records: list[ImageRecord], config: PlotConfig) -> str:
    """Violin plot of R/G/B channel means across all images."""
    recs = [r for r in prepare_records(records, config) if r.pixel_stats]

    r_means = [r.pixel_stats.mean_r for r in recs]  # type: ignore[union-attr]
    g_means = [r.pixel_stats.mean_g for r in recs]  # type: ignore[union-attr]
    b_means = [r.pixel_stats.mean_b for r in recs]  # type: ignore[union-attr]

    fig, ax = create_figure(config)

    parts = ax.violinplot(
        [r_means, g_means, b_means],
        showmeans=True,
        showmedians=True,
    )

    channel_colors = [COLORS["channel_r"], COLORS["channel_g"], COLORS["channel_b"]]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(channel_colors[i])
        pc.set_alpha(0.6)

    # Style the stat lines
    for key in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(COLORS["neutral"])

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Red", "Green", "Blue"])
    ax.set_ylabel("Mean Channel Value")
    ax.set_title(f"Channel Distributions ({len(recs):,} images)")
    fig.tight_layout()

    return save_figure(fig, "channels", config)
