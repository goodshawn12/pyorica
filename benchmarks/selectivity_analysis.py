"""Artifact-brain selectivity index (delta) from per-subject IC source energy CSVs.

Reads the ``*_ic_source_energy.csv`` files produced by ``run_validation.py`` /
``run_all_subjects.py`` and computes the artifact-brain selectivity index
described in the research plan, overall and per artifact class.

Definition
----------
For each IC *k*, the log power reduction of a cleaning stage relative to the
IIR stage is::

    R_k = 10 * log10(ms_iir / ms_stage)      [dB]

where ``ms_stage`` is ``ms_asr`` or ``ms_orica``. Within each subject:

    delta_stage = mean_{k in A} R_k - mean_{k in B} R_k

with A = artifact-labeled ICs (muscle artifact, eye blink, heart beat,
line noise, channel noise) and B = brain-labeled ICs. ``other`` ICs are
excluded from both sets (they mix brain and motion content). Labels come from
the offline-ICA + ICLabel evaluation basis, not from the online pipeline.

The per-class variant replaces A with a single artifact class::

    delta_c = mean_{k in c} R_k - mean_{k in B} R_k

Cross-subject aggregation is mean +/- SD (population SD) of the per-subject
values; ``n_subjects`` counts the subjects contributing to each row (a subject
contributes to class *c* only if it has at least one IC of that class and at
least one brain IC). Classes with small n (heart beat, line noise in the
NCTU-LKT set) are reported but should not be over-interpreted.

Known caveat (discussed 2026-08-29): delta is a difference, so a config that
removes indiscriminately can still score high; report delta alongside brain
energy retention rather than alone.

Usage
-----
    python benchmarks/selectivity_analysis.py --run-dir benchmarks/results/run_YYYYMMDD_HHMMSS

Writes ``selectivity_summary.csv`` to the run directory and prints the table.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

ARTIFACT_LABELS = ["muscle artifact", "eye blink", "heart beat",
                   "line noise", "channel noise"]
BRAIN_LABEL = "brain"


def _log_reduction(ms_iir: float, ms_stage: float) -> float:
    return 10.0 * math.log10(ms_iir / ms_stage)


def _subject_reductions(rows: list[dict], stage_col: str) -> dict[str, list[float]]:
    """Per-label lists of R_k for one subject and one stage column."""
    out: dict[str, list[float]] = {}
    for r in rows:
        ms_iir, ms_stage = float(r["ms_iir"]), float(r[stage_col])
        if ms_iir <= 0 or ms_stage <= 0:
            continue
        out.setdefault(r["label"], []).append(_log_reduction(ms_iir, ms_stage))
    return out


def compute_selectivity(run_dir: Path) -> list[dict]:
    """Overall and per-class selectivity across all subjects in run_dir."""
    csv_files = sorted(run_dir.glob("*_ic_source_energy.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_ic_source_energy.csv files found in {run_dir}")

    scopes = ["all artifacts"] + ARTIFACT_LABELS
    per_subject: dict[str, dict[str, list[float]]] = {
        s: {"asr": [], "orica": []} for s in scopes
    }

    for csv_path in csv_files:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for stage_key, stage_col in (("asr", "ms_asr"), ("orica", "ms_orica")):
            by_label = _subject_reductions(rows, stage_col)
            brain = by_label.get(BRAIN_LABEL)
            if not brain:
                continue
            brain_mean = float(np.mean(brain))
            pooled = [r for lbl in ARTIFACT_LABELS for r in by_label.get(lbl, [])]
            if pooled:
                per_subject["all artifacts"][stage_key].append(
                    float(np.mean(pooled)) - brain_mean)
            for lbl in ARTIFACT_LABELS:
                if by_label.get(lbl):
                    per_subject[lbl][stage_key].append(
                        float(np.mean(by_label[lbl])) - brain_mean)

    summary = []
    for scope in scopes:
        vals_asr = np.array(per_subject[scope]["asr"], dtype=float)
        vals_orica = np.array(per_subject[scope]["orica"], dtype=float)
        n = len(vals_orica)
        summary.append({
            "scope": scope,
            "delta_asr_mean": float(np.mean(vals_asr)) if n else float("nan"),
            "delta_asr_sd": float(np.std(vals_asr)) if n else float("nan"),
            "delta_orica_mean": float(np.mean(vals_orica)) if n else float("nan"),
            "delta_orica_sd": float(np.std(vals_orica)) if n else float("nan"),
            "n_subjects": n,
        })
    return summary


def save_summary_csv(summary: list[dict], out_path: Path) -> None:
    fieldnames = ["scope", "delta_asr_mean", "delta_asr_sd",
                  "delta_orica_mean", "delta_orica_sd", "n_subjects"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the artifact-brain selectivity index (overall and per-class)."
    )
    parser.add_argument(
        "--run-dir", required=True, metavar="DIR",
        help="Directory containing *_ic_source_energy.csv files.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        import sys
        print(f"ERROR: {run_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    summary = compute_selectivity(run_dir)

    print(f"{'scope':16} | {'d_ASR (dB)':>16} | {'d_ORICA (dB)':>16} | n")
    for row in summary:
        print(f"{row['scope']:16} | "
              f"{row['delta_asr_mean']:6.2f} +/- {row['delta_asr_sd']:5.2f} | "
              f"{row['delta_orica_mean']:6.2f} +/- {row['delta_orica_sd']:5.2f} | "
              f"{row['n_subjects']}")

    out_path = run_dir / "selectivity_summary.csv"
    save_summary_csv(summary, out_path)
    print(f"\nSaved summary -> {out_path}")


if __name__ == "__main__":
    main()
