# Latency test

Measure per-chunk wall-clock time for the online pipeline stages (IIR / ASR /
ORICA / ICLabel / reconstruct) and aggregate mean±SD tables and figures across
subjects and chunk sizes.

This folder lives under `benchmarks/` because it reuses the same config and
subject loaders as the validation benchmarks.

## Setup

From the repo root:

```bash
python setup_env.py
source .venv/bin/activate   # Windows: .venv\Scripts\activate
export PYORICA_NCTU_DATA=/path/to/dataset_2019_TBME
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYORICA_NCTU_DATA = "D:\path\to\input_data"
```

## Run (one subject, one chunk size)

`--output-dir` is only the **parent**. The run subfolder name is built from the
other flags, e.g. `s01` + chunk `250` + stream `0` →
`benchmarks/latency_test/results/s01_chunk250_full`.

```bash
python benchmarks/latency_test/run_latency_test.py \
  --subject s01 \
  --chunk-size 250 \
  --stream-seconds 0 \
  --force-iclabel \
  --output-dir benchmarks/latency_test/results
```

Shorter smoke test (`stream-seconds 60` → folder suffix `_sec60`):

```bash
python benchmarks/latency_test/run_latency_test.py \
  --subject s01 \
  --chunk-size 1000 \
  --stream-seconds 60 \
  --output-dir benchmarks/latency_test/results
```

### Flags

| Flag | Meaning |
|------|---------|
| `--subject` | Subject under `PYORICA_NCTU_DATA` (default: `s01`) |
| `--chunk-size N` | Samples per chunk (overrides YAML). @250 Hz: 250=1 s, 1000=4 s |
| `--stream-seconds S` | Online length after calib; `0` = full recording (default) |
| `--output-dir DIR` | Parent folder only (default: `benchmarks/latency_test/results`) |
| `--force-iclabel` | Run ICLabel even if chunk &lt; ~3.5 s |
| `--config PATH` | Default: `benchmarks/config/reference.yaml` |

### Per-run outputs

Written under the auto-named run folder:

- `latency_table_row.csv` / `.png` — compact ASR / ORICA / ICLabel / Total row
- `latency_realtime.png` — per-chunk realtime plot
- `latency_per_chunk.csv` — detailed per-chunk timings
- `latency_summary.csv` — mean/std for all online stages (used by aggregate)
- `latency_report.txt`, `run_meta.yaml`, `config.yaml`

## Aggregate (subjects × chunk sizes)

After several `sXX_chunk{N}_full` runs exist under
`benchmarks/latency_test/results`:

```bash
python benchmarks/latency_test/aggregate_latency_results.py
```

Reads each run's `latency_summary.csv` for online stages:

`iir`, `asr`, `orica_update`, `orica_transform`, `unmixing_pinv`,
`icalabel`, `reconstruct`, `chunk_total`

Writes under `benchmarks/latency_test/results/aggregate/`:

- `per_chunk/` — ms per chunk
- `per_sec/` — ms per 1 s of data @ 250 Hz

Each subfolder contains `latency_all_runs.csv`,
`latency_table_by_chunk.csv` / `.tex` / `.md`, and variability figures.
