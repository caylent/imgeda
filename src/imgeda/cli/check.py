"""imgeda check command group."""

from __future__ import annotations

from typing import Any, Optional

import orjson
import typer
from rich.console import Console

from imgeda.io.manifest_io import read_manifest
from imgeda.models.manifest import ImageRecord

check_app = typer.Typer(help="Check for issues in a manifest.")
console = Console()


def _load_records(manifest: str) -> list[ImageRecord]:
    _meta, records = read_manifest(manifest)
    if not records:
        console.print("[red]No records found in manifest.[/red]")
        raise typer.Exit(1)
    return records


def _output_results(results: list[dict[str, Any]], output: Optional[str], label: str) -> None:
    console.print(f"[bold]{label}:[/bold] {len(results):,} found")
    if output:
        with open(output, "wb") as f:
            f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
        console.print(f"[green]Saved to {output}[/green]")
    elif results:
        for item in results[:20]:
            console.print(f"  {item.get('path', item.get('paths', ''))}")
        if len(results) > 20:
            console.print(f"  ... and {len(results) - 20} more (use --output to save all)")


_manifest_opt = typer.Option(..., "-m", "--manifest", help="Path to manifest JSONL")
_output_opt = typer.Option(None, "-o", "--output", help="Output JSON path")


@check_app.command()
def corrupt(manifest: str = _manifest_opt, output: Optional[str] = _output_opt) -> None:
    """List corrupt images."""
    records = _load_records(manifest)
    results = [{"path": r.path, "filename": r.filename} for r in records if r.is_corrupt]
    _output_results(results, output, "Corrupt images")


@check_app.command()
def exposure(manifest: str = _manifest_opt, output: Optional[str] = _output_opt) -> None:
    """List dark and overexposed images."""
    records = _load_records(manifest)
    results = []
    for r in records:
        issues = []
        if r.is_dark:
            issues.append("dark")
        if r.is_overexposed:
            issues.append("overexposed")
        if issues:
            brightness = r.pixel_stats.mean_brightness if r.pixel_stats else None
            results.append({"path": r.path, "issues": issues, "mean_brightness": brightness})
    _output_results(results, output, "Exposure issues")


@check_app.command()
def artifacts(manifest: str = _manifest_opt, output: Optional[str] = _output_opt) -> None:
    """List images with border artifacts."""
    records = _load_records(manifest)
    results = [
        {
            "path": r.path,
            "delta": r.corner_stats.delta if r.corner_stats else None,
        }
        for r in records
        if r.has_border_artifact
    ]
    _output_results(results, output, "Border artifacts")


@check_app.command(name="duplicates")
def duplicates_cmd(manifest: str = _manifest_opt, output: Optional[str] = _output_opt) -> None:
    """Find duplicate image groups."""
    from imgeda.core.duplicates import find_exact_duplicates, find_near_duplicates

    records = _load_records(manifest)

    exact = find_exact_duplicates(records)
    near = find_near_duplicates(records)

    results = []
    for phash, group in exact.items():
        results.append(
            {
                "type": "exact",
                "phash": phash,
                "count": len(group),
                "paths": [r.path for r in group],
            }
        )
    for group in near:
        results.append(
            {
                "type": "near",
                "count": len(group),
                "paths": [r.path for r in group],
            }
        )

    _output_results(results, output, "Duplicate groups")


@check_app.command()
def blur(manifest: str = _manifest_opt, output: Optional[str] = _output_opt) -> None:
    """List blurry images."""
    records = _load_records(manifest)
    results = [{"path": r.path, "blur_score": r.blur_score} for r in records if r.is_blurry]
    _output_results(results, output, "Blurry images")


@check_app.command()
def leakage(
    manifests: list[str] = typer.Option(
        ..., "-m", "--manifest", help="Manifest paths (provide 2+)"
    ),
    threshold: int = typer.Option(8, "--threshold", help="Hamming distance threshold"),
    output: Optional[str] = _output_opt,
) -> None:
    """Detect duplicate images across splits (data leakage)."""
    if len(manifests) < 2:
        console.print("[red]Provide at least 2 manifests for leakage detection.[/red]")
        raise typer.Exit(1)

    from imgeda.core.leakage import detect_leakage

    all_records: dict[str, list[ImageRecord]] = {}
    for m in manifests:
        meta, recs = read_manifest(m)
        label = meta.input_dir if meta else m
        all_records[label] = recs

    result = detect_leakage(all_records, threshold)
    console.print(f"\n[bold]Cross-split leakage:[/bold] {len(result):,} leaked images")
    if output:
        with open(output, "wb") as f:
            f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2))
        console.print(f"[green]Saved to {output}[/green]")
    elif result:
        for item in result[:20]:
            console.print(f"  {item['path']} appears in: {', '.join(item['found_in'])}")
        if len(result) > 20:
            console.print(f"  ... and {len(result) - 20} more")


@check_app.command(name="all")
def all_checks(manifest: str = _manifest_opt, output: Optional[str] = _output_opt) -> None:
    """Run all checks."""
    from imgeda.core.duplicates import find_exact_duplicates, find_near_duplicates

    console.print("[bold]Running all checks...[/bold]\n")

    records = _load_records(manifest)

    # Corrupt
    corrupt_list = [r for r in records if r.is_corrupt]
    console.print(
        f"  Corrupt: [{'red' if corrupt_list else 'green'}]{len(corrupt_list):,}"
        f"[/{'red' if corrupt_list else 'green'}]"
    )

    # Exposure
    dark = [r for r in records if r.is_dark]
    overexposed = [r for r in records if r.is_overexposed]
    console.print(f"  Dark: [yellow]{len(dark):,}[/yellow]")
    console.print(f"  Overexposed: [yellow]{len(overexposed):,}[/yellow]")

    # Blur
    blurry = [r for r in records if r.is_blurry]
    console.print(f"  Blurry: [yellow]{len(blurry):,}[/yellow]")

    # Artifacts
    artifacts_list = [r for r in records if r.has_border_artifact]
    console.print(f"  Border artifacts: [yellow]{len(artifacts_list):,}[/yellow]")

    # Duplicates (both exact and near)
    exact = find_exact_duplicates(records)
    near = find_near_duplicates(records)
    exact_dup_count = sum(len(v) - 1 for v in exact.values())
    near_dup_groups = len(near)
    console.print(
        f"  Exact duplicates: [yellow]{exact_dup_count:,}[/yellow] (in {len(exact):,} groups)"
    )
    console.print(f"  Near-duplicate groups: [yellow]{near_dup_groups:,}[/yellow]")
