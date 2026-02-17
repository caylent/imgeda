"""Brightness + channel distribution plots."""

from __future__ import annotations


from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord
from imgeda.plotting.base import create_figure, save_figure, valid_records


def plot_brightness(records: list[ImageRecord], config: PlotConfig) -> str:
    """Histogram of mean brightness with shaded dark/normal/overexposed regions."""
    recs = [r for r in valid_records(records) if r.pixel_stats]
    brightness = [r.pixel_stats.mean_brightness for r in recs]  # type: ignore[union-attr]

    fig, ax = create_figure(config)

    ax.hist(brightness, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)

    # Shaded regions
    ax.axvspan(0, 40, alpha=0.1, color="navy", label="Dark (<40)")
    ax.axvspan(220, 255, alpha=0.1, color="red", label="Overexposed (>220)")

    ax.set_xlabel("Mean Brightness")
    ax.set_ylabel("Count")
    ax.set_title(f"Brightness Distribution ({len(recs):,} images)")
    ax.legend()
    fig.tight_layout()

    return save_figure(fig, "brightness", config)


def plot_channels(records: list[ImageRecord], config: PlotConfig) -> str:
    """Box plot of R/G/B channel means across all images."""
    recs = [r for r in valid_records(records) if r.pixel_stats]

    r_means = [r.pixel_stats.mean_r for r in recs]  # type: ignore[union-attr]
    g_means = [r.pixel_stats.mean_g for r in recs]  # type: ignore[union-attr]
    b_means = [r.pixel_stats.mean_b for r in recs]  # type: ignore[union-attr]

    fig, ax = create_figure(config)

    bp = ax.boxplot(
        [r_means, g_means, b_means],
        tick_labels=["Red", "Green", "Blue"],
        patch_artist=True,
    )
    colors = ["#ff6b6b", "#51cf66", "#339af0"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Mean Channel Value")
    ax.set_title(f"Channel Distributions ({len(recs):,} images)")
    fig.tight_layout()

    return save_figure(fig, "channels", config)


def plot_pixel_histogram(records: list[ImageRecord], config: PlotConfig) -> str:
    """Combined brightness histogram — alias for plot_brightness."""
    return plot_brightness(records, config)
