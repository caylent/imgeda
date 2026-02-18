"""imgeda plot command group."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from imgeda.io.manifest_io import read_manifest
from imgeda.models.config import PlotConfig
from imgeda.models.manifest import ImageRecord

plot_app = typer.Typer(help="Generate plots from a manifest.")
console = Console()


def _load_and_config(
    manifest: str,
    output: str,
    fmt: str,
    dpi: int,
    sample: Optional[int],
    seed: int = 42,
) -> tuple[list[ImageRecord], PlotConfig]:
    _meta, records = read_manifest(manifest)
    if not records:
        console.print("[red]No records found in manifest.[/red]")
        raise typer.Exit(1)
    # Carry artifact threshold from scan settings if available
    artifact_threshold = 50.0
    if _meta and _meta.settings.get("artifact_threshold"):
        artifact_threshold = float(_meta.settings["artifact_threshold"])
    config = PlotConfig(
        output_dir=output,
        format=fmt,
        dpi=dpi,
        sample=sample,
        artifact_threshold=artifact_threshold,
        seed=seed,
    )
    return records, config


# Common options
_manifest_opt = typer.Option(..., "-m", "--manifest", help="Path to manifest JSONL")
_output_opt = typer.Option("./plots", "-o", "--output", help="Output directory")
_format_opt = typer.Option("png", "--format", help="Output format (png, pdf, svg)")
_dpi_opt = typer.Option(150, "--dpi", help="DPI for output")
_sample_opt = typer.Option(None, "--sample", help="Sample N records for large datasets")
_seed_opt = typer.Option(42, "--seed", help="Random seed for sampling reproducibility")


@plot_app.command()
def dimensions(
    manifest: str = _manifest_opt,
    output: str = _output_opt,
    fmt: str = _format_opt,
    dpi: int = _dpi_opt,
    sample: Optional[int] = _sample_opt,
    seed: int = _seed_opt,
) -> None:
    """Plot image dimensions (width x height)."""
    from imgeda.plotting.dimensions import plot_dimensions

    records, config = _load_and_config(manifest, output, fmt, dpi, sample, seed)
    path = plot_dimensions(records, config)
    console.print(f"[green]Saved:[/green] {path}")


@plot_app.command()
def file_size(
    manifest: str = _manifest_opt,
    output: str = _output_opt,
    fmt: str = _format_opt,
    dpi: int = _dpi_opt,
    sample: Optional[int] = _sample_opt,
    seed: int = _seed_opt,
) -> None:
    """Plot file size distribution."""
    from imgeda.plotting.file_size import plot_file_size

    records, config = _load_and_config(manifest, output, fmt, dpi, sample, seed)
    path = plot_file_size(records, config)
    console.print(f"[green]Saved:[/green] {path}")


@plot_app.command()
def aspect_ratio(
    manifest: str = _manifest_opt,
    output: str = _output_opt,
    fmt: str = _format_opt,
    dpi: int = _dpi_opt,
    sample: Optional[int] = _sample_opt,
    seed: int = _seed_opt,
) -> None:
    """Plot aspect ratio distribution."""
    from imgeda.plotting.aspect_ratio import plot_aspect_ratio

    records, config = _load_and_config(manifest, output, fmt, dpi, sample, seed)
    path = plot_aspect_ratio(records, config)
    console.print(f"[green]Saved:[/green] {path}")


@plot_app.command()
def brightness(
    manifest: str = _manifest_opt,
    output: str = _output_opt,
    fmt: str = _format_opt,
    dpi: int = _dpi_opt,
    sample: Optional[int] = _sample_opt,
    seed: int = _seed_opt,
) -> None:
    """Plot brightness distribution."""
    from imgeda.plotting.pixel_stats import plot_brightness

    records, config = _load_and_config(manifest, output, fmt, dpi, sample, seed)
    path = plot_brightness(records, config)
    console.print(f"[green]Saved:[/green] {path}")


@plot_app.command()
def channels(
    manifest: str = _manifest_opt,
    output: str = _output_opt,
    fmt: str = _format_opt,
    dpi: int = _dpi_opt,
    sample: Optional[int] = _sample_opt,
    seed: int = _seed_opt,
) -> None:
    """Plot R/G/B channel distributions."""
    from imgeda.plotting.pixel_stats import plot_channels

    records, config = _load_and_config(manifest, output, fmt, dpi, sample, seed)
    path = plot_channels(records, config)
    console.print(f"[green]Saved:[/green] {path}")


@plot_app.command()
def artifacts(
    manifest: str = _manifest_opt,
    output: str = _output_opt,
    fmt: str = _format_opt,
    dpi: int = _dpi_opt,
    sample: Optional[int] = _sample_opt,
    seed: int = _seed_opt,
) -> None:
    """Plot border artifact analysis."""
    from imgeda.plotting.artifacts import plot_artifacts

    records, config = _load_and_config(manifest, output, fmt, dpi, sample, seed)
    path = plot_artifacts(records, config)
    console.print(f"[green]Saved:[/green] {path}")


@plot_app.command()
def duplicates(
    manifest: str = _manifest_opt,
    output: str = _output_opt,
    fmt: str = _format_opt,
    dpi: int = _dpi_opt,
    sample: Optional[int] = _sample_opt,
    seed: int = _seed_opt,
) -> None:
    """Plot duplicate analysis."""
    from imgeda.plotting.duplicates import plot_duplicates

    records, config = _load_and_config(manifest, output, fmt, dpi, sample, seed)
    path = plot_duplicates(records, config)
    console.print(f"[green]Saved:[/green] {path}")


@plot_app.command(name="all")
def all_plots(
    manifest: str = _manifest_opt,
    output: str = _output_opt,
    fmt: str = _format_opt,
    dpi: int = _dpi_opt,
    sample: Optional[int] = _sample_opt,
    seed: int = _seed_opt,
) -> None:
    """Generate all plots."""
    from imgeda.plotting.artifacts import plot_artifacts
    from imgeda.plotting.aspect_ratio import plot_aspect_ratio
    from imgeda.plotting.dimensions import plot_dimensions
    from imgeda.plotting.duplicates import plot_duplicates
    from imgeda.plotting.file_size import plot_file_size
    from imgeda.plotting.pixel_stats import plot_brightness, plot_channels

    records, config = _load_and_config(manifest, output, fmt, dpi, sample, seed)

    plots = [
        ("Dimensions", plot_dimensions),
        ("File size", plot_file_size),
        ("Aspect ratio", plot_aspect_ratio),
        ("Brightness", plot_brightness),
        ("Channels", plot_channels),
        ("Artifacts", plot_artifacts),
        ("Duplicates", plot_duplicates),
    ]
    for name, fn in plots:
        try:
            path = fn(records, config)
            console.print(f"  [green]{name}:[/green] {path}")
        except Exception as e:
            console.print(f"  [red]{name}: Failed — {e}[/red]")

    console.print(f"\n[bold green]All plots saved to {output}/[/bold green]")
