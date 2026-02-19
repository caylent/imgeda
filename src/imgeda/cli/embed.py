"""imgeda embed command — compute and visualize image embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress

from imgeda.io.manifest_io import read_manifest
from imgeda.models.config import PlotConfig

console = Console()


def embed(
    manifest: str = typer.Option(..., "-m", "--manifest", help="Path to manifest JSONL"),
    output: str = typer.Option("./embeddings.npz", "-o", "--out", help="Output .npz file path"),
    model: str = typer.Option("ViT-B-32", "--model", help="OpenCLIP model name"),
    pretrained: str = typer.Option("laion2b_s34b_b79k", "--pretrained", help="Pretrained weights"),
    batch_size: int = typer.Option(32, "--batch-size", help="Inference batch size"),
    device: Optional[str] = typer.Option(None, "--device", help="Torch device (auto-detected)"),
    plot: bool = typer.Option(True, "--plot/--no-plot", help="Generate UMAP plot"),
    plot_dir: str = typer.Option("./plots", "--plot-dir", help="Plot output directory"),
) -> None:
    """Compute CLIP embeddings for images in a manifest.

    Requires: pip install imgeda[embeddings]
    """
    from imgeda.core.embeddings import (
        _check_deps,
        compute_embeddings,
        compute_umap_projection,
        find_outliers,
        save_embeddings,
    )

    try:
        _check_deps()
    except ImportError:
        console.print(
            "[red]Embedding support requires open_clip and torch.[/red]\n"
            "Install with: [bold]pip install imgeda\\[embeddings][/bold]"
        )
        raise typer.Exit(1)

    if not Path(manifest).exists():
        console.print(f"[red]Manifest not found: {manifest}[/red]")
        raise typer.Exit(1)

    _, records = read_manifest(manifest)
    if not records:
        console.print("[yellow]No records in manifest.[/yellow]")
        raise typer.Exit(1)

    # Filter to non-corrupt images that exist
    valid = [r for r in records if not r.is_corrupt and Path(r.path).exists()]
    if not valid:
        console.print("[yellow]No valid images found.[/yellow]")
        raise typer.Exit(1)

    paths = [r.path for r in valid]
    console.print(f"Computing embeddings for {len(paths):,} images...")

    with Progress() as progress:
        task = progress.add_task("Embedding", total=len(paths))

        def callback(current: int, total: int) -> None:
            progress.update(task, completed=current)

        embeddings = compute_embeddings(
            paths,
            model_name=model,
            pretrained=pretrained,
            batch_size=batch_size,
            device=device,
            progress_callback=callback,
        )

    save_embeddings(embeddings, paths, output)
    console.print(f"[green]Saved embeddings ({embeddings.shape}) to {output}[/green]")

    # Outlier detection
    outlier_mask = find_outliers(embeddings)
    outlier_count = int(outlier_mask.sum())
    if outlier_count > 0:
        console.print(f"[yellow]Found {outlier_count} outlier images[/yellow]")
        for i, is_outlier in enumerate(outlier_mask):
            if is_outlier and i < 10:
                console.print(f"  {paths[i]}")

    if plot:
        console.print("Computing UMAP projection...")
        try:
            projection = compute_umap_projection(embeddings)
        except ImportError:
            console.print(
                "[yellow]UMAP requires umap-learn.[/yellow]\n"
                "Install with: [bold]pip install umap-learn[/bold]\n"
                "Skipping UMAP plot.",
            )
            return

        from imgeda.plotting.embeddings import plot_umap

        plot_config = PlotConfig(output_dir=plot_dir)
        plot_path = plot_umap(projection, plot_config, outlier_mask=outlier_mask)
        console.print(f"[green]Saved UMAP plot: {plot_path}[/green]")
