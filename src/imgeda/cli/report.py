"""imgeda info and report commands."""

from __future__ import annotations

import base64
import tempfile

import typer
from rich.console import Console
from rich.table import Table

from imgeda.core.aggregator import aggregate
from imgeda.io.manifest_io import read_manifest
from imgeda.utils import escape_html, fmt_bytes

console = Console()


def info(
    manifest: str = typer.Option(..., "-m", "--manifest", help="Path to manifest JSONL"),
) -> None:
    """Show a quick summary of the manifest."""
    meta, records = read_manifest(manifest)
    if not records:
        console.print("[red]No records found.[/red]")
        raise typer.Exit(1)

    summary = aggregate(records)

    table = Table(title="Dataset Summary", show_header=False, border_style="blue")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Total images", f"{summary.total_images:,}")
    table.add_row("Total size", fmt_bytes(summary.total_size_bytes))
    table.add_row("Corrupt", f"{summary.corrupt_count:,}")
    table.add_row("Dark", f"{summary.dark_count:,}")
    table.add_row("Overexposed", f"{summary.overexposed_count:,}")
    table.add_row("Border artifacts", f"{summary.artifact_count:,}")
    table.add_row(
        "Dimensions",
        f"{summary.min_width}x{summary.min_height} — {summary.max_width}x{summary.max_height}",
    )
    table.add_row("Formats", ", ".join(f"{k} ({v})" for k, v in summary.format_counts.items()))
    table.add_row("Color modes", ", ".join(f"{k} ({v})" for k, v in summary.mode_counts.items()))

    if meta:
        table.add_row("Input dir", meta.input_dir)
        table.add_row("Created", meta.created_at)

    console.print(table)


def report(
    manifest: str = typer.Option(..., "-m", "--manifest", help="Path to manifest JSONL"),
    output: str = typer.Option("./imgeda_report.html", "-o", "--output", help="Output HTML path"),
) -> None:
    """Generate a single-page HTML report with embedded plots and stats."""
    from imgeda.core.duplicates import find_exact_duplicates
    from imgeda.models.config import PlotConfig
    from imgeda.plotting.aspect_ratio import plot_aspect_ratio
    from imgeda.plotting.artifacts import plot_artifacts
    from imgeda.plotting.dimensions import plot_dimensions
    from imgeda.plotting.duplicates import plot_duplicates
    from imgeda.plotting.file_size import plot_file_size
    from imgeda.plotting.pixel_stats import plot_brightness, plot_channels

    meta, records = read_manifest(manifest)
    if not records:
        console.print("[red]No records found.[/red]")
        raise typer.Exit(1)

    summary = aggregate(records)
    dupes = find_exact_duplicates(records)

    # Generate plots as base64 PNGs in auto-cleaned temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_config = PlotConfig(output_dir=tmpdir, format="png", dpi=100, figsize=(10, 6))

        plot_funcs = [
            ("Dimensions", plot_dimensions),
            ("File Size", plot_file_size),
            ("Aspect Ratio", plot_aspect_ratio),
            ("Brightness", plot_brightness),
            ("Channels", plot_channels),
            ("Artifacts", plot_artifacts),
            ("Duplicates", plot_duplicates),
        ]

        plot_images: list[tuple[str, str]] = []  # (title, base64)
        for title, fn in plot_funcs:
            try:
                path = fn(records, plot_config)
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                plot_images.append((title, b64))
            except Exception as exc:
                console.print(f"  [yellow]Warning: {title} plot failed — {exc}[/yellow]")

    # Build HTML with escaped user data
    esc = escape_html
    plots_html = "\n".join(
        f'<div class="plot"><h3>{esc(t)}</h3><img src="data:image/png;base64,{b}" /></div>'
        for t, b in plot_images
    )

    dup_count = sum(len(v) - 1 for v in dupes.values())

    format_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{v:,}</td></tr>" for k, v in summary.format_counts.items()
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>imgeda Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
h1 {{ color: #2c3e50; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
.stat {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.stat .label {{ color: #666; font-size: 0.85em; }}
.stat .value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
.plot {{ background: white; padding: 16px; border-radius: 8px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.plot img {{ width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #f1f3f5; }}
</style></head><body>
<h1>imgeda Dataset Report</h1>
<div class="stats">
<div class="stat"><div class="label">Total Images</div><div class="value">{summary.total_images:,}</div></div>
<div class="stat"><div class="label">Total Size</div><div class="value">{esc(fmt_bytes(summary.total_size_bytes))}</div></div>
<div class="stat"><div class="label">Corrupt</div><div class="value">{summary.corrupt_count:,}</div></div>
<div class="stat"><div class="label">Dark</div><div class="value">{summary.dark_count:,}</div></div>
<div class="stat"><div class="label">Overexposed</div><div class="value">{summary.overexposed_count:,}</div></div>
<div class="stat"><div class="label">Artifacts</div><div class="value">{summary.artifact_count:,}</div></div>
<div class="stat"><div class="label">Duplicates</div><div class="value">{dup_count:,}</div></div>
<div class="stat"><div class="label">Dimensions Range</div><div class="value">{summary.min_width}x{summary.min_height} — {summary.max_width}x{summary.max_height}</div></div>
</div>
<h2>Format Breakdown</h2>
<table><tr><th>Format</th><th>Count</th></tr>
{format_rows}
</table>
<h2>Plots</h2>
{plots_html}
</body></html>"""

    with open(output, "w") as f:
        f.write(html)

    console.print(f"[bold green]Report saved to {output}[/bold green]")
