# imgeda

High-performance CLI tool for exploratory data analysis of image datasets.

Scan folders of images, generate JSONL manifests with metadata and pixel statistics, detect quality issues, find duplicates, and produce publication-ready visualizations — all from the command line.

[![PyPI](https://img.shields.io/pypi/v/imgeda)](https://pypi.org/project/imgeda/)
[![Python](https://img.shields.io/pypi/pyversions/imgeda)](https://pypi.org/project/imgeda/)
[![License](https://img.shields.io/pypi/l/imgeda)](https://github.com/caylent/imgeda/blob/main/LICENSE)

## Installation

```bash
pip install imgeda
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install imgeda
```

## Quick Start

```bash
# Scan a directory of images
imgeda scan ./images -o manifest.jsonl

# View dataset summary
imgeda info -m manifest.jsonl

# Check for quality issues
imgeda check all -m manifest.jsonl

# Generate all plots
imgeda plot all -m manifest.jsonl

# Generate an HTML report
imgeda report -m manifest.jsonl
```

Or just run `imgeda` with no arguments for an interactive wizard that walks you through everything:

```bash
# Interactive mode — auto-detects dataset format (YOLO, COCO, VOC, classification, flat)
imgeda
```

The wizard detects your dataset structure, shows a summary panel with image counts, splits, and class info, then lets you pick which splits and analyses to run.

## Features

- **Fast parallel scanning** with multi-core `ProcessPoolExecutor` and Rich progress bars
- **Resumable** — Ctrl+C anytime, progress is saved. Re-run and it picks up where it left off
- **JSONL manifest** — append-only, crash-tolerant, one record per image
- **Per-image analysis**: dimensions, file size, pixel statistics (mean/std per channel), brightness, perceptual hashing (phash + dhash), border artifact detection
- **Quality checks**: corrupt files, dark/overexposed images, border artifacts, exact and near-duplicate detection
- **7 plot types** with automatic large-dataset adaptations
- **Single-page HTML report** with embedded plots and summary tables
- **Dataset format detection** — auto-detects YOLO, COCO, Pascal VOC, classification, and flat image directories with split-aware scanning
- **Interactive configurator** with Rich panels, split selection, and smart defaults
- **Lambda-compatible core** — the analysis functions have zero CLI dependencies, ready for serverless deployment

## Example Output

All examples below were generated from the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (3,680 images).

### Dimensions

Width vs. height scatter plot with reference lines for 720p, 1080p, and 4K resolutions.

![Dimensions](https://raw.githubusercontent.com/caylent/imgeda/main/docs/examples/dimensions.png)

### Brightness Distribution

Histogram of mean brightness per image, with shaded regions for dark (<40) and overexposed (>220) images.

![Brightness](https://raw.githubusercontent.com/caylent/imgeda/main/docs/examples/brightness.png)

### File Size Distribution

Log-scale histogram with annotated median, P95, and P99 percentile lines.

![File Size](https://raw.githubusercontent.com/caylent/imgeda/main/docs/examples/file_size.png)

### Aspect Ratio Distribution

Histogram with reference lines at common ratios (1:1, 4:3, 3:2, 16:9).

![Aspect Ratio](https://raw.githubusercontent.com/caylent/imgeda/main/docs/examples/aspect_ratio.png)

### Channel Distributions

Box plots of mean R/G/B channel values across the dataset.

![Channels](https://raw.githubusercontent.com/caylent/imgeda/main/docs/examples/channels.png)

### Border Artifact Analysis

Corner-to-center brightness delta histogram with configurable threshold line.

![Artifacts](https://raw.githubusercontent.com/caylent/imgeda/main/docs/examples/artifacts.png)

### Duplicate Analysis

Duplicate group sizes and unique vs. duplicate breakdown.

![Duplicates](https://raw.githubusercontent.com/caylent/imgeda/main/docs/examples/duplicates.png)

## CLI Reference

### `imgeda scan <DIR>`

Scan a directory of images and produce a JSONL manifest.

```
Options:
  -o, --output PATH           Output manifest path [default: imgeda_manifest.jsonl]
  --workers INTEGER           Parallel workers [default: CPU count]
  --checkpoint-every INTEGER  Flush interval [default: 500]
  --resume / --no-resume      Auto-resume from existing manifest [default: resume]
  --force                     Force full rescan (ignore existing manifest)
  --skip-pixel-stats          Metadata-only scan (faster)
  --no-hashes                 Skip perceptual hashing
  --extensions TEXT            Comma-separated extensions to include
  --dark-threshold FLOAT      Dark image threshold [default: 40.0]
  --overexposed-threshold FLOAT  Overexposed threshold [default: 220.0]
  --artifact-threshold FLOAT  Border artifact threshold [default: 50.0]
  --max-image-dim INTEGER     Downsample threshold for pixel stats [default: 2048]
```

### `imgeda info -m <MANIFEST>`

Print a Rich-formatted dataset summary.

### `imgeda check <SUBCOMMAND> -m <MANIFEST>`

Subcommands: `corrupt`, `exposure`, `artifacts`, `duplicates`, `all`

### `imgeda plot <SUBCOMMAND> -m <MANIFEST>`

Subcommands: `dimensions`, `file-size`, `aspect-ratio`, `brightness`, `channels`, `artifacts`, `duplicates`, `all`

```
Common options:
  -o, --output PATH    Output directory [default: ./plots]
  --format TEXT         Output format: png, pdf, svg [default: png]
  --dpi INTEGER         DPI for output [default: 150]
  --sample INTEGER      Sample N records for large datasets
```

### `imgeda report -m <MANIFEST>`

Generate a single-page HTML report with embedded plots and statistics.

## Manifest Format

The manifest is a JSONL file (one JSON object per line):

- **Line 1**: Metadata header (input directory, scan settings, schema version)
- **Lines 2+**: One `ImageRecord` per image with all computed fields

```jsonl
{"__manifest_meta__": true, "input_dir": "./images", "created_at": "2026-02-17T12:00:00", ...}
{"path": "./images/cat.jpg", "width": 500, "height": 375, "format": "JPEG", "phash": "a1b2c3d4", ...}
```

The manifest is append-only and crash-tolerant. Resume is keyed on `(path, file_size, mtime)` — modified files are automatically re-analyzed.

## Performance

Tested on a 10-core Apple M1 Pro with SSD:

| Operation | 3,680 images |
|-----------|-------------|
| Full scan (metadata + pixels + hashes) | ~8s |
| Plot generation | ~3s |
| HTML report | ~4s |

The tool is designed to handle 100K+ image datasets with batched processing, memory-bounded futures, and automatic plot adaptations for large datasets.

## Development

```bash
git clone https://github.com/caylent/imgeda.git
cd imgeda
uv sync --all-extras
uv run pytest
uv run ruff check src/ tests/
```

## License

MIT
