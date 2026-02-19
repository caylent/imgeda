"""imgeda export command group."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from imgeda.io.manifest_io import read_manifest

export_app = typer.Typer(help="Export manifest to other formats.")
console = Console()


@export_app.command()
def parquet(
    manifest: str = typer.Option(..., "-m", "--manifest", help="Path to manifest JSONL"),
    output: str = typer.Option(..., "-o", "--out", help="Output Parquet file path"),
) -> None:
    """Export manifest to Parquet format.

    Requires pyarrow: pip install imgeda[parquet]
    """
    if not Path(manifest).exists():
        console.print(f"[red]Manifest not found: {manifest}[/red]")
        raise typer.Exit(1)

    _, records = read_manifest(manifest)

    if not records:
        console.print("[yellow]No records found in manifest.[/yellow]")
        raise typer.Exit(1)

    try:
        from imgeda.io.parquet_io import records_to_parquet
    except ImportError:
        console.print(
            "[red]pyarrow is required for Parquet export.[/red]\n"
            "Install with: pip install imgeda[parquet]"
        )
        raise typer.Exit(1)

    row_count = records_to_parquet(records, output)
    console.print(f"[green]Exported {row_count:,} records to {output}[/green]")


@export_app.command()
def csv(
    manifest: str = typer.Option(..., "-m", "--manifest", help="Path to manifest JSONL"),
    output: str = typer.Option(..., "-o", "--out", help="Output CSV file path"),
) -> None:
    """Export manifest to CSV format."""
    if not Path(manifest).exists():
        console.print(f"[red]Manifest not found: {manifest}[/red]")
        raise typer.Exit(1)

    _, records = read_manifest(manifest)

    if not records:
        console.print("[yellow]No records found in manifest.[/yellow]")
        raise typer.Exit(1)

    from imgeda.io.csv_io import records_to_csv

    row_count = records_to_csv(records, output)
    console.print(f"[green]Exported {row_count:,} records to {output}[/green]")
