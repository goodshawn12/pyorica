# Data analysis

Scripts for IC source-energy analysis on pyorica pipeline outputs.

## Scripts

| Script | Purpose |
|--------|---------|
| `analyze_ic_source_energy.py` | Cross-subject overview of `*_ic_source_energy.csv` results (and optional recomputation from `*_stages.npz`) |
| `ica_source_energy_analysis_stages.py` | Per-recording ICA / ICLabel / source-MS analysis from a single `*_stages.npz` |

## Setup

From the repo root:

```bash
python setup_env.py
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Extra plotting / ICLabel deps (if not already installed):

```bash
pip install matplotlib mne mne-icalabel
```

## Cross-subject overview (`analyze_ic_source_energy.py`)

Point `--run-dir` at a benchmark run directory that contains
`*_ic_source_energy.csv` files (and optionally `*_stages.npz` + `config.yaml`):

```bash
python data_analysis/analyze_ic_source_energy.py \
  --run-dir benchmarks/result/all/s05_iclabel_interval__asr_fit
```

Outputs (written into `--run-dir` unless otherwise configured):

- `analysis_overview.png`
- `analysis_stage_comparison.png`
- `analysis_all_ics.png`
- `analysis_summary.csv`
- `analysis_per_subject.csv`
- `analysis_ic_counts.csv`

## Per-recording stages analysis (`ica_source_energy_analysis_stages.py`)

```bash
python data_analysis/ica_source_energy_analysis_stages.py \
  --stages path/to/s01_stages.npz
```

Outputs go under `data_analysis/result/<stages_stem>/`.

Tune channel include/exclude, exclude-time masks, and figure options via the
constants at the top of the script.
