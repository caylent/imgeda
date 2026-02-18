"""Interactive configurator — questionary-based wizard for imgeda (no args)."""

from __future__ import annotations

import os
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from imgeda.core.format_detector import DatasetInfo, detect_format
from imgeda.io.image_reader import discover_images
from imgeda.models.config import ScanConfig
from imgeda.utils import fmt_bytes

console = Console()

DEFAULT_EXTENSIONS = ScanConfig().extensions


def _format_dataset_panel(info: DatasetInfo) -> Panel:
    """Build a Rich panel showing detected dataset info."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Format", info.format.upper())
    table.add_row("Images", f"{info.num_images:,} (~{fmt_bytes(info.estimated_size_bytes)})")

    if info.splits:
        split_parts = [f"{k} ({v:,})" for k, v in info.splits.items()]
        table.add_row("Splits", " \u00b7 ".join(split_parts))

    if info.num_classes is not None:
        names_str = ""
        if info.class_names:
            preview = ", ".join(info.class_names[:5])
            if info.num_classes > 5:
                preview += ", \u2026"
            names_str = f" ({preview})"
        table.add_row("Classes", f"{info.num_classes}{names_str}")

    if info.annotations_path:
        table.add_row("Annotations", str(Path(info.annotations_path).name) + "/")

    return Panel(table, title="Dataset Info", border_style="blue")


def _build_split_choices(info: DatasetInfo) -> list[questionary.Choice]:
    """Build questionary choices for split selection."""
    choices = []
    for split_name, count in info.splits.items():
        choices.append(
            questionary.Choice(
                f"{split_name} ({count:,} images)",
                value=split_name,
                checked=True,
            )
        )
    return choices


def _resolve_image_dirs(info: DatasetInfo, selected_splits: list[str]) -> list[str]:
    """Resolve which image directories to scan based on selected splits."""
    if not info.splits or not selected_splits:
        return info.image_dirs

    # For YOLO/COCO/VOC with splits, filter image_dirs to selected splits
    selected: list[str] = []
    for img_dir in info.image_dirs:
        dir_name = Path(img_dir).name
        if dir_name in selected_splits:
            selected.append(img_dir)

    # If we couldn't match by dir name, return all image_dirs
    return selected if selected else info.image_dirs


def run_interactive() -> None:
    """Launch the interactive configuration wizard."""
    console.print("\n[bold blue]Welcome to imgeda \u2014 Image Dataset EDA Tool[/bold blue]\n")

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

    # 2. Format detection
    console.print("  Detecting dataset format\u2026\n")
    info = detect_format(str(dir_path))
    console.print(_format_dataset_panel(info))
    console.print()

    if info.num_images == 0:
        # Fallback: try discover_images on root in case format detector missed something
        images = discover_images(str(dir_path), DEFAULT_EXTENSIONS)
        if not images:
            console.print("[yellow]No images found. Check the directory path.[/yellow]")
            return
        # Update info with discovered count
        try:
            total_size = sum(os.path.getsize(p) for p in images[:1000])
        except OSError:
            total_size = 0
        est_size = total_size * len(images) / min(len(images), 1000) if images else 0
        console.print(f"  Found [bold]{len(images):,}[/bold] images (~{fmt_bytes(est_size)})\n")

    # 3. Split selection (if splits detected)
    selected_splits: list[str] = []
    if info.splits:
        split_choices = _build_split_choices(info)
        selected_splits = questionary.checkbox(
            "Which splits to analyze?",
            choices=split_choices,
        ).ask()

        if selected_splits is None:
            return

        if not selected_splits:
            console.print("[yellow]No splits selected. Exiting.[/yellow]")
            return

    # 4. Analyses
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

    # 5. Workers
    cpu = os.cpu_count() or 4
    workers_str = questionary.text(
        f"How many workers? ({cpu} cores available)",
        default=str(cpu),
    ).ask()

    if workers_str is None:
        return

    workers = int(workers_str) if workers_str.isdigit() else cpu

    # 6. Output
    output = questionary.text(
        "Output manifest path?",
        default="./imgeda_manifest.jsonl",
    ).ask()

    if output is None:
        return

    # 7. Combined report prompt
    generate_report = questionary.confirm(
        "Generate plots and HTML report after scanning?",
        default=True,
    ).ask()

    if generate_report is None:
        return

    # Build config and run
    config = ScanConfig(
        workers=workers,
        include_hashes=include_hashes,
        skip_pixel_stats=skip_pixel_stats,
    )

    # Resolve which directories to scan
    scan_dirs = _resolve_image_dirs(info, selected_splits)

    console.print("\n[bold]Starting scan\u2026[/bold]\n")

    from imgeda.pipeline.runner import run_scan

    # Scan each directory (or the root if flat)
    for scan_dir in scan_dirs:
        run_scan(scan_dir, output, config)

    # Generate plots and report if requested
    if generate_report:
        from imgeda.io.manifest_io import read_manifest
        from imgeda.models.config import PlotConfig
        from imgeda.plotting.aspect_ratio import plot_aspect_ratio
        from imgeda.plotting.artifacts import plot_artifacts
        from imgeda.plotting.dimensions import plot_dimensions
        from imgeda.plotting.duplicates import plot_duplicates
        from imgeda.plotting.file_size import plot_file_size
        from imgeda.plotting.pixel_stats import plot_brightness, plot_channels

        console.print("\n[bold]Generating plots\u2026[/bold]\n")
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

        # Generate HTML report
        console.print("\n[bold]Generating HTML report\u2026[/bold]\n")
        from imgeda.cli.report import report as report_cmd

        try:
            report_cmd(manifest=output, output="./imgeda_report.html")
        except SystemExit:
            pass  # typer.Exit raised when manifest has no records

    console.print("\n[bold green]Done![/bold green]")
