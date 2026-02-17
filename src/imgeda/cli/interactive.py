"""Interactive configurator — questionary-based wizard for imgeda (no args)."""

from __future__ import annotations

import os
from pathlib import Path

import questionary
from rich.console import Console

from imgeda.io.image_reader import discover_images
from imgeda.models.config import ScanConfig

console = Console()

DEFAULT_EXTENSIONS = ScanConfig().extensions


def run_interactive() -> None:
    """Launch the interactive configuration wizard."""
    console.print("\n[bold blue]Welcome to imgeda — Image Dataset EDA Tool[/bold blue]\n")

    # 1. Directory
    directory = questionary.path(
        "Where are your images?",
        default=".",
    ).ask()

    if directory is None:
        return  # user cancelled

    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.is_dir():
        console.print(f"[red]Error: {directory} is not a valid directory[/red]")
        return

    # Quick count
    images = discover_images(str(dir_path), DEFAULT_EXTENSIONS)
    total_size = sum(os.path.getsize(p) for p in images[:1000])
    est_size = total_size * len(images) / min(len(images), 1000) if images else 0

    def fmt_bytes(b: float) -> str:
        if b > 1_000_000_000:
            return f"{b / 1_000_000_000:.1f} GB"
        if b > 1_000_000:
            return f"{b / 1_000_000:.1f} MB"
        return f"{b / 1_000:.1f} KB"

    console.print(f"  Found [bold]{len(images):,}[/bold] images (~{fmt_bytes(est_size)})\n")

    if not images:
        console.print("[yellow]No images found. Check the directory path.[/yellow]")
        return

    # 2. Analyses
    analyses = questionary.checkbox(
        "What analyses would you like to run?",
        choices=[
            questionary.Choice("Basic metadata (dimensions, format, file size)", checked=True),
            questionary.Choice("Pixel statistics (brightness, color channels)", checked=True),
            questionary.Choice("Perceptual hashing (duplicate detection)", checked=True),
            questionary.Choice("Corner/border artifact detection", checked=True),
        ],
    ).ask()

    if analyses is None:
        return

    skip_pixel_stats = "Pixel statistics (brightness, color channels)" not in analyses
    include_hashes = "Perceptual hashing (duplicate detection)" in analyses

    # 3. Workers
    cpu = os.cpu_count() or 4
    workers_str = questionary.text(
        f"How many workers? (default: {cpu}, your CPU has {cpu} cores)",
        default=str(cpu),
    ).ask()

    if workers_str is None:
        return

    workers = int(workers_str) if workers_str.isdigit() else cpu

    # 4. Output
    output = questionary.text(
        "Output manifest path?",
        default="./imgeda_manifest.jsonl",
    ).ask()

    if output is None:
        return

    # 5. Plots
    generate_plots = questionary.confirm(
        "Generate plots after scanning?",
        default=True,
    ).ask()

    if generate_plots is None:
        return

    # Build config and run
    config = ScanConfig(
        workers=workers,
        include_hashes=include_hashes,
        skip_pixel_stats=skip_pixel_stats,
    )

    console.print("\n[bold]Starting scan...[/bold]\n")

    from imgeda.pipeline.runner import run_scan

    run_scan(str(dir_path), output, config)

    # Generate plots if requested
    if generate_plots:
        from imgeda.io.manifest_io import read_manifest
        from imgeda.models.config import PlotConfig
        from imgeda.plotting.aspect_ratio import plot_aspect_ratio
        from imgeda.plotting.artifacts import plot_artifacts
        from imgeda.plotting.dimensions import plot_dimensions
        from imgeda.plotting.duplicates import plot_duplicates
        from imgeda.plotting.file_size import plot_file_size
        from imgeda.plotting.pixel_stats import plot_brightness, plot_channels

        console.print("\n[bold]Generating plots...[/bold]\n")
        _meta, records = read_manifest(output)
        plot_config = PlotConfig(output_dir="./plots")

        for name, fn in [
            ("Dimensions", plot_dimensions),
            ("File size", plot_file_size),
            ("Aspect ratio", plot_aspect_ratio),
            ("Brightness", plot_brightness),
            ("Channels", plot_channels),
            ("Artifacts", plot_artifacts),
            ("Duplicates", plot_duplicates),
        ]:
            try:
                path = fn(records, plot_config)
                console.print(f"  [green]{name}:[/green] {path}")
            except Exception as e:
                console.print(f"  [red]{name}: {e}[/red]")

    console.print("\n[bold green]Done![/bold green]")
