"""Paired comparison of two benchmark runs with rank-based statistics.

Compares two run directories produced by ``run_all_subjects.py`` /
``run_validation.py`` on their shared subjects, using paired per-subject
measures and the Wilcoxon signed-rank test (rank-based, as agreed on
2026-09-05: retention values are not normally distributed at the IC level,
and between-subject variance is large, so paired rank tests are preferred
over unpaired t-tests).

Per-subject measures
--------------------
delta   : artifact-brain selectivity index (see selectivity_analysis.py) --
          mean log power reduction of artifact-labeled ICs minus mean log
          power reduction of brain-labeled ICs, in dB, vs the IIR stage.
brain / eye / muscle / other :
          within-subject MEDIAN of per-IC energy retained (pct_orica).
          Median rather than mean because per-IC retention is strongly
          non-normal (left-skewed for brain; Shapiro-Wilk p < 0.05 for all
          classes on the reference run).

Statistics
----------
For each measure, subjects present in both runs (finite in both) are paired
and tested with ``scipy.stats.wilcoxon`` (two-sided). The reported effect
size is the median of the per-subject differences (B - A). p-values are
reported raw; when many comparisons are made across configs, apply Holm (or
Bonferroni) across the full set of tests.

Usage
-----
    python benchmarks/compare_configs.py --run-a benchmarks/results/run_A \\
                                         --run-b benchmarks/results/run_B \\
                                         [--label-a refconfig] [--label-b variant]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

ARTIFACT_LABELS = {"muscle artifact", "eye blink", "heart beat",
                   "line noise", "channel noise"}
CLASS_MEASURES = [("brain", "brain"), ("eye", "eye blink"),
                  ("muscle", "muscle artifact"), ("other", "other")]


def subject_measures(rows: list[dict]) -> dict | None:
    """delta + per-class median retention for one subject's CSV rows."""
    reductions_artifact, reductions_brain = [], []
    pct_by_label: dict[str, list[float]] = {}
    for r in rows:
        ms_iir, ms_orica = float(r["ms_iir"]), float(r["ms_orica"])
        if ms_iir <= 0 or ms_orica <= 0:
            continue
        reduction = 10.0 * math.log10(ms_iir / ms_orica)
        if r["label"] in ARTIFACT_LABELS:
            reductions_artifact.append(reduction)
        elif r["label"] == "brain":
            reductions_brain.append(reduction)
        pct_by_label.setdefault(r["label"], []).append(float(r["pct_orica"]))
    if not (reductions_artifact and reductions_brain):
        return None
    out = {"delta": float(np.mean(reductions_artifact) - np.mean(reductions_brain))}
    for key, label in CLASS_MEASURES:
        vals = pct_by_label.get(label)
        out[key] = float(np.median(vals)) if vals else float("nan")
    return out


def load_run(run_dir: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(run_dir.glob("*_ic_source_energy.csv")):
        with open(f, newline="") as fh:
            m = subject_measures(list(csv.DictReader(fh)))
        if m:
            out[f.name.split("_")[0]] = m
    return out


def compare(run_a: Path, run_b: Path) -> list[dict]:
    a, b = load_run(run_a), load_run(run_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("No shared subjects between the two runs.")
    results = []
    for measure in ["delta"] + [k for k, _ in CLASS_MEASURES]:
        x = np.array([a[s][measure] for s in shared])
        y = np.array([b[s][measure] for s in shared])
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        row = {"measure": measure, "n": int(len(x))}
        if len(x) < 5 or np.allclose(x, y):
            row.update({"median_diff": float("nan"), "p": float("nan")})
        else:
            _, p = stats.wilcoxon(x, y)
            row.update({"median_diff": float(np.median(y - x)), "p": float(p)})
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon comparison of two benchmark runs.")
    parser.add_argument("--run-a", required=True, metavar="DIR")
    parser.add_argument("--run-b", required=True, metavar="DIR")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    results = compare(Path(args.run_a), Path(args.run_b))
    print(f"Paired Wilcoxon: {args.label_a} vs {args.label_b} "
          f"(median diff = {args.label_b} - {args.label_a})")
    print(f"{'measure':8} {'n':>3} {'median_diff':>12} {'p':>10}")
    for r in results:
        print(f"{r['measure']:8} {r['n']:3d} {r['median_diff']:+12.3f} {r['p']:10.2e}")


if __name__ == "__main__":
    main()
