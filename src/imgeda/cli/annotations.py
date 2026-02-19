"""imgeda annotations command — annotation analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import orjson
import typer
from rich.console import Console

from imgeda.core.annotations import analyze_annotations
from imgeda.core.format_detector import detect_format

console = Console()


def annotations_cmd(
    directory: str = typer.Argument(..., help="Dataset directory"),
    fmt: Optional[str] = typer.Option(
        None, "--format", "-f", help="Format (yolo, coco, voc). Auto-detected if omitted."
    ),
    label_dir: Optional[str] = typer.Option(None, "--labels", help="Labels directory (YOLO)"),
    annotation_file: Optional[str] = typer.Option(
        None, "--annotation-file", help="Annotation JSON file (COCO)"
    ),
    output: Optional[str] = typer.Option(None, "-o", "--out", help="Output JSON path"),
) -> None:
    """Analyze annotations in a dataset (YOLO, COCO, VOC)."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        console.print(f"[red]Not a directory: {directory}[/red]")
        raise typer.Exit(1)

    # Auto-detect format if not provided
    if fmt is None:
        info = detect_format(directory)
        fmt = info.format
        console.print(f"[dim]Detected format: {fmt}[/dim]")
    else:
        fmt = fmt.lower()

    if fmt not in ("yolo", "coco", "voc"):
        console.print(f"[yellow]Format '{fmt}' does not have annotations to analyze.[/yellow]")
        raise typer.Exit(0)

    # Get class names from format detector for YOLO
    class_names: list[str] | None = None
    if fmt == "yolo":
        info = detect_format(directory)
        class_names = info.class_names or None

    stats = analyze_annotations(
        dataset_dir=directory,
        fmt=fmt,
        label_dir=label_dir,
        annotation_file=annotation_file,
        class_names=class_names,
    )

    if stats.total_annotations == 0:
        console.print("[yellow]No annotations found.[/yellow]")
        raise typer.Exit(0)

    # Display summary
    console.print("\n[bold]Annotation Analysis[/bold]")
    console.print(f"  Images: {stats.total_images:,} ({stats.annotated_images:,} annotated)")
    console.print(f"  Total annotations: {stats.total_annotations:,}")
    console.print(f"  Classes: {stats.num_classes}")
    console.print(f"  Avg objects/image: {stats.mean_objects_per_image:.1f}")
    console.print(f"  Max objects/image: {stats.max_objects_per_image}")
    console.print(
        f"  Size breakdown: {stats.small_count:,} small / "
        f"{stats.medium_count:,} medium / {stats.large_count:,} large"
    )

    # Class distribution (top 20)
    console.print("\n[bold]Class Distribution (top 20):[/bold]")
    for cls, count in list(stats.class_counts.items())[:20]:
        pct = count / stats.total_annotations * 100
        bar = "█" * int(pct / 2)
        console.print(f"  {cls:30s} {count:>8,} ({pct:5.1f}%) {bar}")

    if stats.orphan_images:
        console.print(f"\n[yellow]  {len(stats.orphan_images)} unannotated images[/yellow]")
    if stats.orphan_annotations:
        console.print(
            f"[yellow]  {len(stats.orphan_annotations)} annotations without images[/yellow]"
        )

    if output:
        data = orjson.dumps(stats.to_dict(), option=orjson.OPT_INDENT_2)
        with open(output, "wb") as f:
            f.write(data)
        console.print(f"\n[green]Saved to {output}[/green]")
