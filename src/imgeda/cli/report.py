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
    table.add_row("Blurry", f"{summary.blurry_count:,}")
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
    from imgeda.plotting.artifacts import plot_artifacts
    from imgeda.plotting.aspect_ratio import plot_aspect_ratio
    from imgeda.plotting.blur import plot_blur
    from imgeda.plotting.dimensions import plot_dimensions
    from imgeda.plotting.duplicates import plot_duplicates
    from imgeda.plotting.exif import (
        plot_camera_distribution,
        plot_focal_length,
        plot_iso_distribution,
    )
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
            ("Blur", plot_blur),
            ("Artifacts", plot_artifacts),
            ("Duplicates", plot_duplicates),
            ("Camera Distribution", plot_camera_distribution),
            ("Focal Length", plot_focal_length),
            ("ISO Distribution", plot_iso_distribution),
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

    # Build flagged images table data (worst offenders)
    flagged = []
    for r in records:
        issues = []
        if r.is_corrupt:
            issues.append("corrupt")
        if r.is_dark:
            issues.append("dark")
        if r.is_overexposed:
            issues.append("overexposed")
        if r.is_blurry:
            issues.append("blurry")
        if r.has_border_artifact:
            issues.append("artifact")
        if issues:
            flagged.append((r.path, r.filename, ", ".join(issues), r.width, r.height))

    # Build HTML
    esc = escape_html
    plots_html = "\n".join(
        f'<div class="plot"><h3>{esc(t)}</h3><img src="data:image/png;base64,{b}" /></div>'
        for t, b in plot_images
    )

    dup_count = sum(len(v) - 1 for v in dupes.values())

    format_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{v:,}</td></tr>" for k, v in summary.format_counts.items()
    )

    # Flagged images table rows (limit 200 for file size)
    flagged_rows = "".join(
        f'<tr><td title="{esc(p)}">{esc(fn)}</td><td>{esc(issues)}</td><td>{w}x{h}</td></tr>'
        for p, fn, issues, w, h in flagged[:200]
    )
    flagged_note = (
        f"<p><em>Showing {min(len(flagged), 200)} of {len(flagged)} flagged images</em></p>"
        if len(flagged) > 200
        else ""
    )

    # EXIF summary rows
    exif_rows = ""
    if summary.camera_make_counts:
        for make, count in list(summary.camera_make_counts.items())[:10]:
            exif_rows += f"<tr><td>{esc(make)}</td><td>{count:,}</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>imgeda Report</title>
<style>
:root {{ --bg: #f8f9fa; --card: white; --border: #e9ecef; --text: #2c3e50; --muted: #666; --accent: #4c72b0; --danger: #e41a1c; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: var(--bg); color: var(--text); }}
h1 {{ margin-bottom: 4px; }} h1 small {{ font-weight: normal; color: var(--muted); font-size: 0.5em; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
.stat {{ background: var(--card); padding: 14px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.stat .label {{ color: var(--muted); font-size: 0.82em; margin-bottom: 2px; }}
.stat .value {{ font-size: 1.4em; font-weight: 700; }}
.stat .value.warn {{ color: var(--danger); }}
.plot {{ background: var(--card); padding: 16px; border-radius: 8px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.plot img {{ width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; background: var(--card); border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ background: #f1f3f5; cursor: pointer; user-select: none; }}
th:hover {{ background: #e2e6ea; }}
th::after {{ content: ' ⇅'; color: #ccc; font-size: 0.8em; }}
.section {{ margin: 32px 0; }}
.tabs {{ display: flex; gap: 4px; margin-bottom: -1px; }}
.tab {{ padding: 8px 16px; background: var(--border); border-radius: 6px 6px 0 0; cursor: pointer; font-size: 0.9em; }}
.tab.active {{ background: var(--card); font-weight: 600; }}
.tab-content {{ display: none; }} .tab-content.active {{ display: block; }}
.filter-bar {{ margin: 12px 0; }}
.filter-bar input {{ padding: 6px 12px; border: 1px solid var(--border); border-radius: 4px; width: 300px; font-size: 0.9em; }}
</style></head><body>
<h1>imgeda Report <small>Dataset EDA</small></h1>

<div class="stats">
<div class="stat"><div class="label">Total Images</div><div class="value">{summary.total_images:,}</div></div>
<div class="stat"><div class="label">Total Size</div><div class="value">{esc(fmt_bytes(summary.total_size_bytes))}</div></div>
<div class="stat"><div class="label">Corrupt</div><div class="value{" warn" if summary.corrupt_count else ""}">{summary.corrupt_count:,}</div></div>
<div class="stat"><div class="label">Dark</div><div class="value{" warn" if summary.dark_count else ""}">{summary.dark_count:,}</div></div>
<div class="stat"><div class="label">Overexposed</div><div class="value{" warn" if summary.overexposed_count else ""}">{summary.overexposed_count:,}</div></div>
<div class="stat"><div class="label">Blurry</div><div class="value{" warn" if summary.blurry_count else ""}">{summary.blurry_count:,}</div></div>
<div class="stat"><div class="label">Artifacts</div><div class="value{" warn" if summary.artifact_count else ""}">{summary.artifact_count:,}</div></div>
<div class="stat"><div class="label">Duplicates</div><div class="value{" warn" if dup_count else ""}">{dup_count:,}</div></div>
<div class="stat"><div class="label">Dimensions</div><div class="value" style="font-size:1em">{summary.min_width}x{summary.min_height} — {summary.max_width}x{summary.max_height}</div></div>
</div>

<div class="section">
<h2>Format Breakdown</h2>
<table><tr><th>Format</th><th>Count</th></tr>
{format_rows}
</table>
</div>

{'<div class="section"><h2>Camera Models (EXIF)</h2><table><tr><th>Camera</th><th>Count</th></tr>' + exif_rows + "</table></div>" if exif_rows else ""}

<div class="section">
<h2>Flagged Images</h2>
{flagged_note}
<div class="filter-bar"><input type="text" id="flagFilter" placeholder="Filter by filename or issue..." oninput="filterFlagged()"></div>
<table id="flaggedTable">
<tr><th onclick="sortTable('flaggedTable',0)">Filename</th><th onclick="sortTable('flaggedTable',1)">Issues</th><th onclick="sortTable('flaggedTable',2)">Size</th></tr>
{flagged_rows}
</table>
</div>

<div class="section">
<h2>Plots</h2>
{plots_html}
</div>

<script>
function sortTable(id,col){{
  const t=document.getElementById(id),rows=[...t.rows].slice(1);
  const dir=t.dataset.sortDir==col?-1:1;t.dataset.sortDir=dir==1?col:-1;
  rows.sort((a,b)=>{{
    let x=a.cells[col].textContent,y=b.cells[col].textContent;
    const nx=parseFloat(x.replace(/,/g,'')),ny=parseFloat(y.replace(/,/g,''));
    return isNaN(nx)?dir*x.localeCompare(y):dir*(nx-ny);
  }});
  rows.forEach(r=>t.appendChild(r));
}}
function filterFlagged(){{
  const q=document.getElementById('flagFilter').value.toLowerCase();
  const rows=document.querySelectorAll('#flaggedTable tr:not(:first-child)');
  rows.forEach(r=>{{r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';}});
}}
</script>
</body></html>"""

    with open(output, "w") as f:
        f.write(html)

    console.print(f"[bold green]Report saved to {output}[/bold green]")
