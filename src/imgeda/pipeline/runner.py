"""ProcessPoolExecutor orchestration with Rich progress."""

from __future__ import annotations

import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from imgeda.core.analyzer import analyze_image
from imgeda.io.image_reader import discover_images
from imgeda.io.manifest_io import append_records, create_manifest
from imgeda.models.config import ScanConfig
from imgeda.models.manifest import ImageRecord, ManifestMeta
from imgeda.pipeline.checkpoint import filter_pending, load_processed_set
from imgeda.pipeline.signals import ShutdownHandler, worker_init

console = Console()

# Max futures in flight at once to bound memory usage
_BATCH_SIZE = 5000


def run_scan(
    input_dir: str,
    output_path: str,
    config: ScanConfig,
) -> tuple[int, int]:
    """Run the scan pipeline. Returns (total_processed, corrupt_count)."""
    shutdown = ShutdownHandler()
    shutdown.install()

    try:
        return _run_scan_inner(input_dir, output_path, config, shutdown)
    finally:
        shutdown.uninstall()


def _run_scan_inner(
    input_dir: str,
    output_path: str,
    config: ScanConfig,
    shutdown: ShutdownHandler,
) -> tuple[int, int]:
    output = Path(output_path)

    # Discover images
    console.print(f"[bold]Discovering images in[/bold] {input_dir} ...")
    all_images = discover_images(input_dir, config.extensions)
    total_discovered = len(all_images)
    console.print(f"  Found [bold]{total_discovered:,}[/bold] images")

    if total_discovered == 0:
        console.print("[yellow]No images found.[/yellow]")
        return 0, 0

    # Resume logic
    already_processed = 0
    if config.resume and not config.force and output.exists():
        processed_set, existing_records = load_processed_set(output_path)
        already_processed = len(existing_records)
        pending = filter_pending(all_images, processed_set)
        if already_processed > 0:
            console.print(
                f"  Resuming: [green]{already_processed:,}[/green] already processed, "
                f"[bold]{len(pending):,}[/bold] remaining"
            )
    else:
        pending = all_images
        # Truncate and write fresh manifest header
        meta = ManifestMeta(
            input_dir=os.path.abspath(input_dir),
            total_files=total_discovered,
            created_at=datetime.now(timezone.utc).isoformat(),
            settings={
                "workers": config.workers,
                "include_hashes": config.include_hashes,
                "skip_pixel_stats": config.skip_pixel_stats,
                "artifact_threshold": config.artifact_threshold,
                "dark_threshold": config.dark_threshold,
                "overexposed_threshold": config.overexposed_threshold,
            },
        )
        create_manifest(output_path, meta)

    if not pending:
        console.print("[green]All images already processed![/green]")
        return already_processed, 0

    total_to_process = len(pending)
    total_done = 0
    corrupt_count = 0
    dark_count = 0
    overexposed_count = 0
    buffer: list[ImageRecord] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Scanning images"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Scanning", total=total_to_process)

        workers = min(config.workers, total_to_process)
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
            # Submit in batches to bound memory usage
            for batch_start in range(0, len(pending), _BATCH_SIZE):
                if shutdown.is_shutting_down:
                    break

                batch = pending[batch_start : batch_start + _BATCH_SIZE]
                futures: dict[Future[ImageRecord], str] = {
                    executor.submit(analyze_image, path, config): path for path in batch
                }

                for future in as_completed(futures):
                    if shutdown.is_shutting_down:
                        progress.update(task, description="[yellow]Shutting down gracefully...")
                        for f in futures:
                            f.cancel()
                        break

                    try:
                        record = future.result(timeout=60)
                    except Exception:
                        record = ImageRecord(
                            path=futures[future],
                            filename=os.path.basename(futures[future]),
                            is_corrupt=True,
                            analyzed_at=datetime.now(timezone.utc).isoformat(),
                        )

                    buffer.append(record)
                    if record.is_corrupt:
                        corrupt_count += 1
                    if record.is_dark:
                        dark_count += 1
                    if record.is_overexposed:
                        overexposed_count += 1

                    total_done += 1
                    progress.update(task, advance=1)

                    # Checkpoint flush
                    if len(buffer) >= config.checkpoint_every:
                        append_records(output_path, buffer)
                        buffer.clear()

        # Flush remaining buffer
        if buffer:
            append_records(output_path, buffer)
            buffer.clear()

    final_count = already_processed + total_done

    # Summary
    console.print()
    console.print("[bold green]Scan complete![/bold green]")
    console.print(f"  Total processed: [bold]{final_count:,}[/bold]")
    if corrupt_count:
        console.print(f"  Corrupt: [red]{corrupt_count:,}[/red]")
    if dark_count:
        console.print(f"  Dark: [yellow]{dark_count:,}[/yellow]")
    if overexposed_count:
        console.print(f"  Overexposed: [yellow]{overexposed_count:,}[/yellow]")
    console.print(f"  Manifest: {output_path}")

    if shutdown.is_shutting_down:
        console.print("\n[yellow]Saved progress. Resume with: imgeda scan {input_dir}[/yellow]")

    return final_count, corrupt_count
