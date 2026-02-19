"""imgeda scan command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from imgeda.models.config import ScanConfig


def scan(
    directory: str = typer.Argument(..., help="Directory to scan for images"),
    output: str = typer.Option(
        "./imgeda_manifest.jsonl", "-o", "--output", help="Manifest output path"
    ),
    workers: Optional[int] = typer.Option(None, "--workers", help="Number of parallel workers"),
    checkpoint_every: int = typer.Option(500, "--checkpoint-every", help="Flush interval"),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Auto-resume from existing manifest"
    ),
    force: bool = typer.Option(False, "--force", help="Force rescan, ignoring existing manifest"),
    skip_pixel_stats: bool = typer.Option(
        False, "--skip-pixel-stats", help="Metadata-only scan (faster)"
    ),
    skip_exif: bool = typer.Option(False, "--skip-exif", help="Skip EXIF metadata extraction"),
    no_hashes: bool = typer.Option(False, "--no-hashes", help="Skip perceptual hashing"),
    extensions: Optional[str] = typer.Option(
        None, "--extensions", help="Comma-separated extensions"
    ),
    dark_threshold: float = typer.Option(40.0, "--dark-threshold"),
    overexposed_threshold: float = typer.Option(220.0, "--overexposed-threshold"),
    artifact_threshold: float = typer.Option(50.0, "--artifact-threshold"),
    blur_threshold: float = typer.Option(100.0, "--blur-threshold", help="Blur score threshold"),
    skip_blur: bool = typer.Option(False, "--skip-blur", help="Skip blur detection"),
    max_image_dim: int = typer.Option(2048, "--max-image-dim", help="Downsample threshold"),
) -> None:
    """Scan a directory of images and produce a JSONL manifest."""
    from imgeda.pipeline.runner import run_scan

    dir_path = Path(directory)
    if not dir_path.is_dir():
        typer.echo(f"Error: {directory} is not a valid directory", err=True)
        raise typer.Exit(1)

    config = ScanConfig(
        checkpoint_every=checkpoint_every,
        include_hashes=not no_hashes,
        skip_pixel_stats=skip_pixel_stats,
        skip_exif=skip_exif,
        dark_threshold=dark_threshold,
        overexposed_threshold=overexposed_threshold,
        artifact_threshold=artifact_threshold,
        blur_threshold=blur_threshold,
        skip_blur=skip_blur,
        max_image_dimension=max_image_dim,
        resume=resume,
        force=force,
    )
    if workers is not None:
        config.workers = workers
    if extensions:
        config.extensions = tuple(f".{e.strip().lstrip('.')}" for e in extensions.split(","))

    run_scan(str(dir_path), output, config)
