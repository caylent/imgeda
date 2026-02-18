# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**imgeda** is a CLI tool for exploratory data analysis (EDA) of image datasets. It scans image directories, generates JSONL manifests with metadata/pixel statistics, detects quality issues, finds duplicates, and produces visualizations. Built for both local use and AWS Lambda deployment.

## Commands

```bash
# Install dependencies
uv sync --all-extras

# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_analyzer.py

# Run single test
uv run pytest tests/test_analyzer.py::TestAnalyzeImage::test_normal_image

# Run with coverage
uv run pytest --cov=src/imgeda --cov-report=html

# Lint
uv run ruff check src/ tests/

# Lint with autofix
uv run ruff check --fix src/ tests/

# Format
uv run ruff format src/ tests/

# Type check (strict mode)
uv run mypy src/imgeda/
```

## Architecture

The codebase follows a layered architecture: **CLI → Pipeline → Core (pure functions)**.

- **`src/imgeda/cli/`** — Typer-based CLI commands (`scan`, `check`, `plot`, `report`, `info`, `diff`, `gate`, `export`, interactive wizard). Entry point: `cli/app.py`.
- **`src/imgeda/core/`** — Pure analysis functions with zero CLI dependencies, designed to be Lambda-compatible. Includes `analyzer.py` (single-image analysis + EXIF extraction), `detector.py` (exposure/artifact detection), `hasher.py` (perceptual hashing), `duplicates.py` (hash-based clustering with sub-hash bucketing to avoid O(n²)), `aggregator.py` (dataset summary), `diff.py` (manifest comparison), `gate.py` (policy-as-code evaluation), and `format_detector.py` (YOLO/COCO/VOC/classification detection).
- **`src/imgeda/pipeline/`** — Orchestration layer: `ProcessPoolExecutor` parallelism with Rich progress bars, crash-tolerant resume via checkpoint logic, and graceful Ctrl+C signal handling. Batched processing with memory-bounded futures (batch size 5000).
- **`src/imgeda/io/`** — JSONL manifest I/O with atomic writes (temp file + rename) and corruption-tolerant parsing (skips malformed lines). Parquet export via `parquet_io.py` (optional pyarrow dependency).
- **`src/imgeda/models/`** — Dataclasses with `__slots__`: `ImageRecord` (including EXIF fields), `PixelStats`, `CornerStats`, `ManifestMeta`, `ScanConfig`, `PlotConfig`, `Policy`.
- **`src/imgeda/plotting/`** — Seven plot types (dimensions, file_size, aspect_ratio, brightness, channels, artifacts, duplicates), each in its own module with a shared theme engine.
- **`src/imgeda/lambda_handler/`** — AWS Lambda handlers (5 functions sharing one Docker image, routed by ACTION env var). CDK infrastructure in `cdk/`.

## Key Design Patterns

- **Core functions never raise exceptions** — corrupt/unreadable files are flagged in the `ImageRecord` rather than throwing.
- **Resume is keyed on `(path, file_size, mtime)`** — modified files are automatically re-analyzed on resume.
- **JSONL manifest format** — first line is metadata (`__manifest_meta__: true`), remaining lines are `ImageRecord` entries. Append-only with atomic metadata updates.
- **Serialization uses orjson** for performance.

## Pre-push Checklist

Before pushing any commits or creating/updating PRs, **always** run the full CI checks locally and confirm they pass:

```bash
uv run ruff format --check src/ tests/   # Format check (fix with: uv run ruff format src/ tests/)
uv run ruff check src/ tests/            # Lint check
uv run mypy src/imgeda/                  # Type check
uv run pytest                            # All tests
```

Do not push until all four pass.

## Code Style

- Line length: 100 characters
- Target Python: 3.10+ (uses `from __future__ import annotations`)
- Type annotations throughout; mypy strict mode
- `__slots__` on all dataclasses
- Tests are class-based with pytest fixtures; test images are generated programmatically in `conftest.py`
