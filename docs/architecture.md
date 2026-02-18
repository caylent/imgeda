# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         imgeda CLI (typer)                          │
│                                                                     │
│  scan ─ check ─ plot ─ report ─ info ─ diff ─ gate ─ export        │
│                     interactive wizard                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
┌──────────────────┐ ┌────────────────┐ ┌──────────────────┐
│    Pipeline      │ │   Plotting     │ │   I/O Layer      │
│                  │ │                │ │                  │
│ ProcessPool      │ │ 7 plot types   │ │ JSONL manifest   │
│ executor         │ │ + theme engine │ │ (atomic writes)  │
│ checkpointing    │ │                │ │                  │
│ signal handling  │ │ dimensions     │ │ Parquet export   │
│ batched futures  │ │ file_size      │ │ (streaming)      │
│ (5000/batch)     │ │ aspect_ratio   │ │                  │
│                  │ │ brightness     │ │ Image discovery  │
│ resume via       │ │ channels       │ │ (walk + filter)  │
│ (path,size,mtime)│ │ artifacts      │ │                  │
└────────┬─────────┘ │ duplicates     │ └──────────────────┘
         │           └────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Core (pure functions, no CLI deps)                │
│                                                                     │
│  analyzer.py ── Single-image analysis + EXIF extraction (never raises)│
│  detector.py ── Pixel stats, exposure, corner/border detection      │
│  hasher.py ──── Perceptual hashing (phash + dhash)                  │
│  duplicates.py ─ Exact + near-duplicate clustering                  │
│  aggregator.py ─ Dataset-level summary statistics                   │
│  diff.py ─────── Manifest comparison (added/removed/changed)        │
│  gate.py ─────── Policy-as-code QA evaluation                       │
│  format_detector.py ─ YOLO/COCO/VOC/classification/flat detection   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Models (dataclasses with __slots__)               │
│                                                                     │
│  ImageRecord ─ PixelStats ─ CornerStats ─ ManifestMeta              │
│  ScanConfig ─ PlotConfig ─ Policy ─ DatasetInfo                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Local CLI Flow

```
User runs: imgeda scan ./images -o manifest.jsonl

  1. CLI parses args → builds ScanConfig
  2. Pipeline discovers images (walk + extension filter)
  3. Pipeline loads existing manifest for resume (if any)
  4. Pipeline fans out to ProcessPoolExecutor workers
     ┌──────────────┐
     │   Worker 1   │──→ analyze_image(path, config) → ImageRecord
     │   Worker 2   │──→ analyze_image(path, config) → ImageRecord
     │   Worker N   │──→ analyze_image(path, config) → ImageRecord
     └──────────────┘
  5. Records batched (5000) → appended to JSONL manifest
  6. Ctrl+C → graceful shutdown, progress saved
  7. Re-run → resumes from (path, size, mtime) keys
```

## Serverless (AWS) Flow

```
S3 Input Bucket
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              Step Functions State Machine                 │
│                                                          │
│  ┌─────────────────┐                                     │
│  │  ListImages      │  List S3 objects, chunk into        │
│  │  Lambda          │  batches of ~20 keys                │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │  Map State       │  Fan out to N parallel Lambdas      │
│  │  (maxConc=10)    │                                     │
│  │  ┌─────────────┐ │                                     │
│  │  │AnalyzeBatch │ │  Download images → analyze_image()  │
│  │  │ Lambda      │ │  → write partial JSONL to S3        │
│  │  └─────────────┘ │                                     │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │ MergeManifests   │  Concatenate partials → final       │
│  │ Lambda           │  manifest with ManifestMeta header  │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │ Aggregate        │  Compute DatasetSummary from        │
│  │ Lambda           │  manifest → write JSON summary      │
│  └────────┬────────┘                                     │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │ GeneratePlots    │  Render all 7 plot types             │
│  │ Lambda           │  → upload PNGs to S3                │
│  └─────────────────┘                                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
S3 Output Bucket
├── manifests/manifest.jsonl
├── summary/summary.json
└── plots/*.png
```

## CI/CD Quality Gate Flow

```
imgeda scan ./images -o manifest.jsonl
    │
    ▼
imgeda gate -m manifest.jsonl -p policy.yml
    │
    ├── exit 0 → all checks pass → pipeline continues
    └── exit 2 → check failed → pipeline fails

Policy checks:
  ✓ min_images_total    (e.g., ≥100)
  ✓ max_corrupt_pct     (e.g., ≤1%)
  ✓ max_overexposed_pct (e.g., ≤5%)
  ✓ max_underexposed_pct(e.g., ≤5%)
  ✓ max_duplicate_pct   (e.g., ≤10%)
```

## Manifest Diff Flow

```
imgeda diff --old v1.jsonl --new v2.jsonl --out diff.json
    │
    ▼
Key records by path → compute:
  • Added paths   (in new, not in old)
  • Removed paths (in old, not in new)
  • Changed paths (same path, different fields)
  • Summary deltas (corrupt count, duplicate groups, totals)
```

## Module Dependency Graph

```
cli/
├── app.py ────────→ cli/{scan,check,plot,report,diff,gate,export,interactive}
├── scan.py ───────→ pipeline/runner
├── check.py ──────→ core/duplicates, io/manifest_io
├── plot.py ───────→ plotting/*, io/manifest_io
├── report.py ─────→ plotting/*, io/manifest_io, core/aggregator
├── diff.py ───────→ core/diff, io/manifest_io
├── gate.py ───────→ core/gate, io/manifest_io
├── export.py ─────→ io/parquet_io, io/manifest_io
└── interactive.py ─→ core/format_detector, pipeline/runner, plotting/*

pipeline/
├── runner.py ─────→ core/analyzer, io/manifest_io, pipeline/{checkpoint,signals}
├── checkpoint.py ──→ io/manifest_io
└── signals.py      (standalone)

core/                             (zero CLI dependencies)
├── analyzer.py ───→ core/{detector,hasher}, models/{config,manifest}
├── detector.py ───→ models/manifest (PixelStats, CornerStats)
├── hasher.py       (standalone, uses imagehash)
├── duplicates.py ──→ models/manifest
├── aggregator.py ──→ models/manifest
├── diff.py ───────→ core/duplicates, models/manifest
├── gate.py ───────→ core/duplicates, models/{manifest,policy}
└── format_detector.py (standalone)

io/
├── manifest_io.py ─→ models/manifest (orjson serialization)
├── parquet_io.py ──→ models/manifest (pyarrow, optional)
└── image_reader.py  (standalone, os.walk)

lambda_handler/
├── handler.py ────→ lambda_handler/handlers/*
└── handlers/
    ├── list_images.py ────→ (boto3 only)
    ├── analyze_batch.py ──→ core/analyzer, models/{config,manifest}
    ├── merge_manifests.py ─→ models/manifest
    ├── aggregate.py ──────→ core/aggregator, models/manifest
    └── generate_plots.py ──→ plotting/*, models/{config,manifest}

models/                           (standalone dataclasses)
├── manifest.py ── ImageRecord (+ EXIF fields), PixelStats, CornerStats, ManifestMeta
├── config.py ──── ScanConfig, PlotConfig
└── policy.py ──── Policy

plotting/
├── base.py ───────→ models/{config,manifest} (theme, figure helpers)
├── dimensions.py ──→ base
├── file_size.py ───→ base
├── aspect_ratio.py ─→ base
├── pixel_stats.py ──→ base
├── artifacts.py ────→ base
└── duplicates.py ───→ base, core/duplicates
```

## Key Design Principles

1. **Core never raises** — corrupt/unreadable images are flagged in ImageRecord, not thrown
2. **Resume by identity** — keyed on `(path, file_size, mtime)`, modified files re-analyzed
3. **Append-only JSONL** — crash-tolerant, truncated lines skipped on read
4. **Atomic metadata writes** — temp file + `os.replace()` to avoid partial writes
5. **Memory-bounded** — batched futures (5000), streaming Parquet writes (10K chunks)
6. **Layered isolation** — core has zero CLI deps, runs identically in Lambda
7. **Optional heavy deps** — `pyarrow` via `imgeda[parquet]`, `boto3` lazy-imported in Lambda
