"""Aggregate latency runs into an Overleaf-ready computation-time table.

Scans ``Latency_test/results/<subject>_chunk<N>_full/latency_table_row.csv``
(and similar ``_sec*`` folders), then writes:

1. ``latency_all_runs.csv`` — one row per subject × chunk size
2. ``latency_table_by_chunk.csv`` / ``.tex`` — mean ± SD across subjects
   for ASR / ORICA / ICLabel / Total at each chunk size
3. Figures showing cross-session variability

Usage
-----
    python Latency_test/aggregate_latency_results.py
    python Latency_test/aggregate_latency_results.py --results-dir Latency_test/results
    python Latency_test/aggregate_latency_results.py --output-dir Latency_test/results/aggregate
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

_LATENCY_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _LATENCY_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

STAGES = ("asr", "orica", "icalabel", "total")
STAGE_LABELS = {
    "asr": "ASR",
    "orica": "ORICA",
    "icalabel": "ICLabel",
    "total": "Total",
}
STAGE_COLORS = {
    "asr": "#ff7f0e",
    "orica": "#2ca02c",
    "icalabel": "#d62728",
    "total": "#1f77b4",
}

_DIR_RE = re.compile(
    r"^(?P<subject>s\d+)_chunk(?P<chunk>\d+)_(?:full|sec[\dp]+)$",
    re.IGNORECASE,
)


def _mean_pm_std(mean: float, std: float, nd: int = 1, *, latex: bool = False) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "—" if not latex else "---"
    pm = r" $\pm$ " if latex else " ± "
    return f"{mean:.{nd}f}{pm}{std:.{nd}f}"


def _load_table_row(csv_path: Path) -> dict | None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]
    out = {
        "subject": str(row.get("subject", "")).strip(),
        "chunk_size": int(float(row["chunk_size"])),
        "chunk_dur_s": float(row.get("chunk_dur_s", 0.0) or 0.0),
        "sfreq_hz": float(row.get("sfreq_hz", 250.0) or 250.0),
        "n_chunks": int(float(row.get("n_chunks", 0) or 0)),
        "stream_seconds": float(row.get("stream_seconds", 0.0) or 0.0),
        "force_iclabel": str(row.get("force_iclabel", "")).lower() in ("1", "true"),
        "source_dir": csv_path.parent.name,
        "source_path": str(csv_path),
    }
    for stage in STAGES:
        out[f"{stage}_mean_ms"] = float(row[f"{stage}_mean_ms"])
        out[f"{stage}_std_ms"] = float(row[f"{stage}_std_ms"])
    if not out["subject"]:
        m = _DIR_RE.match(csv_path.parent.name)
        if m:
            out["subject"] = m.group("subject")
    if out["chunk_dur_s"] <= 0 and out["sfreq_hz"] > 0:
        out["chunk_dur_s"] = out["chunk_size"] / out["sfreq_hz"]
    return out


def collect_runs(results_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for csv_path in sorted(results_dir.glob("*/latency_table_row.csv")):
        m = _DIR_RE.match(csv_path.parent.name)
        if m is None:
            # Still accept if CSV itself is valid
            pass
        row = _load_table_row(csv_path)
        if row is None:
            continue
        if m is not None:
            row["subject"] = m.group("subject")
            row["chunk_size"] = int(m.group("chunk"))
            if row["chunk_dur_s"] <= 0 and row["sfreq_hz"] > 0:
                row["chunk_dur_s"] = row["chunk_size"] / row["sfreq_hz"]
        runs.append(row)
    return runs


def aggregate_by_chunk(runs: list[dict]) -> list[dict]:
    """Cross-subject mean ± SD of per-run stage means, for each chunk size."""
    by_chunk: dict[int, list[dict]] = {}
    for r in runs:
        by_chunk.setdefault(int(r["chunk_size"]), []).append(r)

    summary: list[dict] = []
    for chunk_size in sorted(by_chunk):
        group = by_chunk[chunk_size]
        subjects = sorted({r["subject"] for r in group})
        sfreq = float(np.median([r["sfreq_hz"] for r in group]))
        chunk_dur = float(np.median([r["chunk_dur_s"] for r in group]))
        row: dict = {
            "chunk_size": chunk_size,
            "chunk_dur_s": chunk_dur,
            "sfreq_hz": sfreq,
            "n_subjects": len(subjects),
            "subjects": ",".join(subjects),
        }
        for stage in STAGES:
            vals = np.asarray([r[f"{stage}_mean_ms"] for r in group], dtype=np.float64)
            mean = float(np.mean(vals))
            # ddof=1 when n>1 so SD reflects cross-session variability
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            row[f"{stage}_mean_ms"] = mean
            row[f"{stage}_std_ms"] = std
            row[f"{stage}_ms"] = _mean_pm_std(mean, std)
            row[f"{stage}_ms_tex"] = _mean_pm_std(mean, std, latex=True)
            row[f"{stage}_min_ms"] = float(np.min(vals))
            row[f"{stage}_max_ms"] = float(np.max(vals))
        summary.append(row)
    return summary


def write_all_runs_csv(runs: list[dict], path: Path) -> None:
    fieldnames = [
        "subject",
        "chunk_size",
        "chunk_dur_s",
        "sfreq_hz",
        "n_chunks",
        "stream_seconds",
        "force_iclabel",
        "asr_mean_ms",
        "asr_std_ms",
        "orica_mean_ms",
        "orica_std_ms",
        "icalabel_mean_ms",
        "icalabel_std_ms",
        "total_mean_ms",
        "total_std_ms",
        "source_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(runs, key=lambda x: (x["subject"], x["chunk_size"])):
            writer.writerow(r)


def write_summary_csv(summary: list[dict], path: Path) -> None:
    fieldnames = [
        "chunk_size",
        "chunk_dur_s",
        "sfreq_hz",
        "n_subjects",
        "subjects",
        "asr_mean_ms",
        "asr_std_ms",
        "asr_ms",
        "orica_mean_ms",
        "orica_std_ms",
        "orica_ms",
        "icalabel_mean_ms",
        "icalabel_std_ms",
        "icalabel_ms",
        "total_mean_ms",
        "total_std_ms",
        "total_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)


def write_overleaf_tex(summary: list[dict], path: Path) -> None:
    """Compact LaTeX table: chunk × stage mean ± SD (ms)."""
    lines = [
        "% Auto-generated by Latency_test/aggregate_latency_results.py",
        "% Computation time (ms) per stage; mean ± SD across sessions.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-chunk computation time (ms) by pipeline stage and chunk size. "
        r"Values are mean $\pm$ SD across sessions.}",
        r"\label{tab:latency-chunk-stages}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Chunk & Duration & ASR (ms) & ORICA (ms) & ICLabel (ms) & Total (ms) \\",
        r"\midrule",
    ]
    for row in summary:
        chunk = int(row["chunk_size"])
        dur = float(row["chunk_dur_s"])
        lines.append(
            f"{chunk} & {dur:.2f}\\,s & "
            f"{row['asr_ms_tex']} & {row['orica_ms_tex']} & "
            f"{row['icalabel_ms_tex']} & {row['total_ms_tex']} \\\\"
        )
    n = int(summary[0]["n_subjects"]) if summary else 0
    # n_subjects may differ by chunk; report range
    ns = [int(r["n_subjects"]) for r in summary]
    n_txt = str(ns[0]) if len(set(ns)) == 1 else f"{min(ns)}--{max(ns)}"
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"% n_subjects per chunk size: {n_txt}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_table(summary: list[dict], path: Path) -> None:
    lines = [
        "# Computation time by chunk size",
        "",
        "Mean ± SD across sessions (ms).",
        "",
        "| Chunk | Duration | n | ASR (ms) | ORICA (ms) | ICLabel (ms) | Total (ms) |",
        "|------:|---------:|--:|---------:|-----------:|-------------:|-----------:|",
    ]
    for row in summary:
        lines.append(
            f"| {int(row['chunk_size'])} | {row['chunk_dur_s']:.2f}s | "
            f"{int(row['n_subjects'])} | {row['asr_ms']} | {row['orica_ms']} | "
            f"{row['icalabel_ms']} | {row['total_ms']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_grouped_bars(summary: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    chunks = [int(r["chunk_size"]) for r in summary]
    x = np.arange(len(chunks), dtype=float)
    width = 0.18
    stages_plot = list(STAGES)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, stage in enumerate(stages_plot):
        means = [r[f"{stage}_mean_ms"] for r in summary]
        stds = [r[f"{stage}_std_ms"] for r in summary]
        offset = (i - (len(stages_plot) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=STAGE_COLORS[stage],
            alpha=0.9,
            label=STAGE_LABELS[stage],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n({summary[i]['chunk_dur_s']:.1f}s)" for i, c in enumerate(chunks)]
    )
    ax.set_xlabel("Chunk size (samples)")
    ax.set_ylabel("Computation time (ms)")
    ax.set_title("Per-chunk stage cost across sessions (mean ± SD)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_subject_variability(runs: list[dict], path: Path) -> None:
    """One panel per stage: subject means vs chunk size (variability)."""
    import matplotlib.pyplot as plt

    chunk_sizes = sorted({int(r["chunk_size"]) for r in runs})
    subjects = sorted({r["subject"] for r in runs})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, stage in zip(axes, STAGES):
        for subj in subjects:
            xs, ys = [], []
            for cs in chunk_sizes:
                match = [
                    r
                    for r in runs
                    if r["subject"] == subj and int(r["chunk_size"]) == cs
                ]
                if not match:
                    continue
                xs.append(cs)
                ys.append(match[0][f"{stage}_mean_ms"])
            if xs:
                ax.plot(xs, ys, marker="o", lw=1.0, ms=4, alpha=0.55)
        # overlay cross-subject mean ± SD
        means, stds = [], []
        for cs in chunk_sizes:
            vals = [
                r[f"{stage}_mean_ms"]
                for r in runs
                if int(r["chunk_size"]) == cs
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.errorbar(
            chunk_sizes,
            means,
            yerr=stds,
            fmt="k-o",
            lw=2.0,
            ms=6,
            capsize=4,
            label="mean ± SD",
            zorder=5,
        )
        ax.set_title(STAGE_LABELS[stage])
        ax.set_xlabel("Chunk size (samples)")
        ax.set_ylabel("ms / chunk")
        ax.grid(True, alpha=0.3)
        if stage == "asr":
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Cross-session variability of per-chunk computation time", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_boxplot(runs: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    chunk_sizes = sorted({int(r["chunk_size"]) for r in runs})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, stage in zip(axes, STAGES):
        data = []
        labels = []
        for cs in chunk_sizes:
            vals = [
                r[f"{stage}_mean_ms"]
                for r in runs
                if int(r["chunk_size"]) == cs
            ]
            if vals:
                data.append(vals)
                labels.append(str(cs))
        if not data:
            continue
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
        for box in bp["boxes"]:
            box.set_facecolor(STAGE_COLORS[stage])
            box.set_alpha(0.55)
        ax.set_title(STAGE_LABELS[stage])
        ax.set_xlabel("Chunk size (samples)")
        ax.set_ylabel("ms / chunk")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Distribution of subject means by chunk size", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate latency_table_row.csv into Overleaf table + figures."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_LATENCY_ROOT / "results",
        help="Parent folder with <subject>_chunk<N>_full/ runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write aggregate outputs (default: <results-dir>/aggregate).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (results_dir / "aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(results_dir)
    if not runs:
        print(f"ERROR: no latency_table_row.csv under {results_dir}", file=sys.stderr)
        sys.exit(1)

    summary = aggregate_by_chunk(runs)
    subjects = sorted({r["subject"] for r in runs})
    chunks = sorted({int(r["chunk_size"]) for r in runs})

    all_csv = out_dir / "latency_all_runs.csv"
    sum_csv = out_dir / "latency_table_by_chunk.csv"
    tex_path = out_dir / "latency_table_by_chunk.tex"
    md_path = out_dir / "latency_table_by_chunk.md"
    fig_bars = out_dir / "latency_by_chunk_bars.png"
    fig_var = out_dir / "latency_cross_session_variability.png"
    fig_box = out_dir / "latency_by_chunk_boxplot.png"

    write_all_runs_csv(runs, all_csv)
    write_summary_csv(summary, sum_csv)
    write_overleaf_tex(summary, tex_path)
    write_markdown_table(summary, md_path)
    plot_grouped_bars(summary, fig_bars)
    plot_subject_variability(runs, fig_var)
    plot_boxplot(runs, fig_box)

    print(f"Runs loaded: {len(runs)}")
    print(f"Subjects: {len(subjects)}  ({', '.join(subjects)})")
    print(f"Chunk sizes: {chunks}")
    print()
    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote → {out_dir.resolve()}")
    for p in (all_csv, sum_csv, tex_path, md_path, fig_bars, fig_var, fig_box):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
