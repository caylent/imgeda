"""imgeda diff — manifest comparison command."""

from __future__ import annotations

from typing import Optional

import orjson
import typer
from rich.console import Console

from imgeda.core.diff import diff_manifests
from imgeda.io.manifest_io import read_manifest

console = Console()


def diff(
    old: str = typer.Option(..., "--old", help="Path to old manifest JSONL"),
    new: str = typer.Option(..., "--new", help="Path to new manifest JSONL"),
    output: Optional[str] = typer.Option(None, "-o", "--out", help="Output JSON path"),
) -> None:
    """Compare two manifests and show differences."""
    _, old_records = read_manifest(old)
    _, new_records = read_manifest(new)

    if not old_records and not new_records:
        console.print("[yellow]Both manifests are empty.[/yellow]")
        raise typer.Exit(0)

    result = diff_manifests(old_records, new_records)
    summary = result.summary

    # Display summary
    console.print("\n[bold]Manifest Diff[/bold]")
    console.print(f"  Old: {summary.total_old:,} images")
    console.print(f"  New: {summary.total_new:,} images")
    console.print()
    console.print(f"  [green]Added:[/green]     {summary.added_count:,}")
    console.print(f"  [red]Removed:[/red]   {summary.removed_count:,}")
    console.print(f"  [yellow]Changed:[/yellow]   {summary.changed_count:,}")
    console.print(f"  Unchanged: {result.unchanged_count:,}")

    if summary.corrupt_old != summary.corrupt_new:
        console.print(f"\n  Corrupt: {summary.corrupt_old} -> {summary.corrupt_new}")
    if summary.duplicate_groups_old != summary.duplicate_groups_new:
        console.print(
            f"  Duplicate groups: {summary.duplicate_groups_old} -> {summary.duplicate_groups_new}"
        )

    if output:
        data = orjson.dumps(result.to_dict(), option=orjson.OPT_INDENT_2)
        with open(output, "wb") as f:
            f.write(data)
        console.print(f"\n[green]Saved to {output}[/green]")
