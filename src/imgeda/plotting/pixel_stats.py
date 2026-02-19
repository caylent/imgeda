"""Brightness + channel distribution plots."""

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


def plot_brightness(records: list[ImageRecord], config: PlotConfig) -> str:
    """Histogram of mean brightness with shaded dark/normal/overexposed regions."""
    recs = [r for r in prepare_records(records, config) if r.pixel_stats]
    brightness = [r.pixel_stats.mean_brightness for r in recs]  # type: ignore[union-attr]

    fig, ax = create_figure(config)

    ax.hist(
        brightness,
        bins=60,
        color=COLORS["primary"],
        edgecolor="none",
        alpha=0.85,
    )

    # Shaded regions — neutral gray, labeled directly
    ax.axvspan(0, 40, alpha=0.06, color=COLORS["primary"])
    ax.axvspan(220, 255, alpha=0.06, color=COLORS["primary"])

    y_max = ax.get_ylim()[1]
    direct_label(ax, 2, y_max * 0.92, "Dark (<40)")
    direct_label(ax, 222, y_max * 0.92, "Overexposed (>220)")

    ax.set_xlabel("Mean Brightness")
    ax.set_ylabel("Count")
    ax.set_title(f"Brightness Distribution  ({len(recs):,} images)")
    tufte_axes(ax)
    fig.tight_layout()

    return save_figure(fig, "brightness", config)


def plot_channels(records: list[ImageRecord], config: PlotConfig) -> str:
    """Violin plot of R/G/B channel means across all images."""
    recs = [r for r in prepare_records(records, config) if r.pixel_stats]

    r_means = [r.pixel_stats.mean_r for r in recs]  # type: ignore[union-attr]
    g_means = [r.pixel_stats.mean_g for r in recs]  # type: ignore[union-attr]
    b_means = [r.pixel_stats.mean_b for r in recs]  # type: ignore[union-attr]

    fig, ax = create_figure(config)

    if recs:
        parts = ax.violinplot(
            [r_means, g_means, b_means],
            showmeans=True,
            showmedians=True,
        )

        channel_colors = [COLORS["channel_r"], COLORS["channel_g"], COLORS["channel_b"]]
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(channel_colors[i])
            pc.set_alpha(0.55)
            pc.set_edgecolor(channel_colors[i])
            pc.set_linewidth(0.8)

        for key in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color(COLORS["light"])
                parts[key].set_linewidth(0.8)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Red", "Green", "Blue"])
    ax.set_ylabel("Mean Channel Value")
    ax.set_title(f"Channel Distributions  ({len(recs):,} images)")
    tufte_axes(ax)
    fig.tight_layout()

    return save_figure(fig, "channels", config)
