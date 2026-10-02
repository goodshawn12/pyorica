"""Aggregate latency runs into an Overleaf-ready computation-time table.

Scans ``benchmarks/latency_test/results/<subject>_chunk<N>_full/latency_summary.csv``
(and similar ``_sec*`` folders) for **all** online stages, then writes two
subfolders under the output directory:

* ``per_chunk/`` — ms **per chunk** (raw measurement window)
* ``per_sec/`` — same stages normalized to **1 s of data**
  (250 samples @ 250 Hz): ``ms_per_sec = ms_per_chunk / chunk_dur_s``

Each folder contains::

    latency_all_runs.csv
    latency_table_by_chunk.csv / .tex / .md
    latency_by_chunk_bars.png
    latency_cross_session_variability.png
    latency_by_chunk_boxplot.png

Online stages (same as ``run_latency_test.ONLINE_STAGES``)::

    iir, asr, orica_update, orica_transform, unmixing_pinv,
    icalabel, reconstruct, chunk_total

Usage
-----
    python benchmarks/latency_test/aggregate_latency_results.py
    python benchmarks/latency_test/aggregate_latency_results.py --results-dir benchmarks/latency_test/results
    python benchmarks/latency_test/aggregate_latency_results.py --output-dir benchmarks/latency_test/results/aggregate
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

_LATENCY_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _LATENCY_ROOT.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Match run_latency_test.ONLINE_STAGES (order used in tables / plots).
STAGES = (
    "iir",
    "asr",
    "orica_update",
    "orica_transform",
    "unmixing_pinv",
    "icalabel",
    "reconstruct",
    "chunk_total",
)
STAGE_LABELS = {
    "iir": "IIR",
    "asr": "ASR",
    "orica_update": "ORICA update",
    "orica_transform": "ORICA transform",
    "unmixing_pinv": "Unmixing pinv",
    "icalabel": "ICLabel",
    "reconstruct": "Reconstruct",
    "chunk_total": "Total",
}
STAGE_COLORS = {
    "iir": "#1f77b4",
    "asr": "#ff7f0e",
    "orica_update": "#2ca02c",
    "orica_transform": "#98df8a",
    "unmixing_pinv": "#c5b0d5",
    "icalabel": "#d62728",
    "reconstruct": "#17becf",
    "chunk_total": "#7f7f7f",
}
# Stages shown in the grouped bar chart (exclude total so bars are comparable).
BAR_STAGES = tuple(s for s in STAGES if s != "chunk_total")

_DIR_RE = re.compile(
    r"^(?P<subject>s\d+)_chunk(?P<chunk>\d+)_(?:full|sec[\dp]+)$",
    re.IGNORECASE,
)


def _mean_pm_std(mean: float, std: float, nd: int = 1, *, latex: bool = False) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "—" if not latex else "---"
    pm = r" $\pm$ " if latex else " ± "
    return f"{mean:.{nd}f}{pm}{std:.{nd}f}"


def _load_summary_online(csv_path: Path) -> dict[str, dict[str, float]] | None:
    """Parse latency_summary.csv → {stage: {mean_ms, std_ms, ...}} for online rows."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if str(row.get("phase", "")).strip().lower() != "online":
            continue
        stage = str(row.get("stage", "")).strip()
        if not stage:
            continue
        out[stage] = {
            "mean_ms": float(row["mean_ms"]),
            "std_ms": float(row["std_ms"]),
            "count": float(row.get("count", 0) or 0),
        }
    return out or None


def _load_table_row_meta(csv_path: Path) -> dict:
    """Optional metadata from latency_table_row.csv (subject, force_iclabel, …)."""
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    row = rows[0]
    return {
        "subject": str(row.get("subject", "")).strip(),
        "chunk_size": int(float(row["chunk_size"])) if row.get("chunk_size") else None,
        "chunk_dur_s": float(row.get("chunk_dur_s", 0.0) or 0.0),
        "sfreq_hz": float(row.get("sfreq_hz", 250.0) or 250.0),
        "n_chunks": int(float(row.get("n_chunks", 0) or 0)),
        "stream_seconds": float(row.get("stream_seconds", 0.0) or 0.0),
        "force_iclabel": str(row.get("force_iclabel", "")).lower() in ("1", "true"),
    }


def _load_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "latency_summary.csv"
    if not summary_path.exists():
        return None
    stage_stats = _load_summary_online(summary_path)
    if stage_stats is None:
        return None

    meta = _load_table_row_meta(run_dir / "latency_table_row.csv")
    m = _DIR_RE.match(run_dir.name)

    subject = meta.get("subject") or (m.group("subject") if m else run_dir.name)
    chunk_size = meta.get("chunk_size")
    if chunk_size is None and m is not None:
        chunk_size = int(m.group("chunk"))
    if chunk_size is None:
        return None

    sfreq = float(meta.get("sfreq_hz") or 250.0)
    chunk_dur = float(meta.get("chunk_dur_s") or 0.0)
    if chunk_dur <= 0 and sfreq > 0:
        chunk_dur = chunk_size / sfreq

    n_chunks = int(meta.get("n_chunks") or 0)
    if n_chunks <= 0:
        # Fall back to count from any online stage
        for st in STAGES:
            if st in stage_stats and stage_stats[st]["count"] > 0:
                n_chunks = int(stage_stats[st]["count"])
                break

    out: dict = {
        "subject": subject,
        "chunk_size": int(chunk_size),
        "chunk_dur_s": chunk_dur,
        "sfreq_hz": sfreq,
        "n_chunks": n_chunks,
        "stream_seconds": float(meta.get("stream_seconds") or 0.0),
        "force_iclabel": bool(meta.get("force_iclabel", False)),
        "source_dir": run_dir.name,
        "source_path": str(summary_path),
    }
    for stage in STAGES:
        st = stage_stats.get(stage)
        if st is None:
            out[f"{stage}_mean_ms"] = float("nan")
            out[f"{stage}_std_ms"] = float("nan")
        else:
            out[f"{stage}_mean_ms"] = st["mean_ms"]
            out[f"{stage}_std_ms"] = st["std_ms"]
    return out


def collect_runs(results_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for summary_path in sorted(results_dir.glob("*/latency_summary.csv")):
        row = _load_run(summary_path.parent)
        if row is None:
            continue
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
            vals = np.asarray(
                [r[f"{stage}_mean_ms"] for r in group], dtype=np.float64
            )
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                mean = float("nan")
                std = float("nan")
                vmin = float("nan")
                vmax = float("nan")
            else:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
            row[f"{stage}_mean_ms"] = mean
            row[f"{stage}_std_ms"] = std
            row[f"{stage}_ms"] = _mean_pm_std(mean, std)
            row[f"{stage}_ms_tex"] = _mean_pm_std(mean, std, latex=True)
            row[f"{stage}_min_ms"] = vmin
            row[f"{stage}_max_ms"] = vmax
        summary.append(row)
    return summary


def _stage_fieldnames(kind: str) -> list[str]:
    """Build CSV column names for all stages."""
    cols: list[str] = []
    for stage in STAGES:
        if kind == "runs":
            cols.extend([f"{stage}_mean_ms", f"{stage}_std_ms"])
        else:  # summary
            cols.extend(
                [f"{stage}_mean_ms", f"{stage}_std_ms", f"{stage}_ms"]
            )
    return cols


def write_all_runs_csv(runs: list[dict], path: Path) -> None:
    fieldnames = [
        "subject",
        "chunk_size",
        "chunk_dur_s",
        "sfreq_hz",
        "n_chunks",
        "stream_seconds",
        "force_iclabel",
        *_stage_fieldnames("runs"),
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
        *_stage_fieldnames("summary"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)


def write_overleaf_tex(summary: list[dict], path: Path) -> None:
    """Wide LaTeX table: chunk × all stages (mean ± SD, ms)."""
    # Abbreviated headers to keep the table printable.
    short = {
        "iir": "IIR",
        "asr": "ASR",
        "orica_update": "ORICA upd.",
        "orica_transform": "ORICA xf.",
        "unmixing_pinv": "pinv",
        "icalabel": "ICLabel",
        "reconstruct": "Recon.",
        "chunk_total": "Total",
    }
    n_cols = 2 + len(STAGES)  # Chunk + Duration + stages
    col_spec = "l" + "r" * (n_cols - 1)
    header = (
        r"Chunk & Duration & "
        + " & ".join(f"{short[s]} (ms)" for s in STAGES)
        + r" \\"
    )
    lines = [
        "% Auto-generated by benchmarks/latency_test/aggregate_latency_results.py",
        "% Computation time (ms) per stage; mean ± SD across sessions.",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Per-chunk computation time (ms) by pipeline stage and chunk size. "
        r"Values are mean $\pm$ SD across sessions.}",
        r"\label{tab:latency-chunk-stages}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for row in summary:
        chunk = int(row["chunk_size"])
        dur = float(row["chunk_dur_s"])
        cells = " & ".join(row[f"{s}_ms_tex"] for s in STAGES)
        lines.append(f"{chunk} & {dur:.2f}\\,s & {cells} \\\\")
    ns = [int(r["n_subjects"]) for r in summary]
    n_txt = str(ns[0]) if ns and len(set(ns)) == 1 else (
        f"{min(ns)}--{max(ns)}" if ns else "0"
    )
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


def write_markdown_table(
    summary: list[dict],
    path: Path,
    *,
    title: str = "Computation time by chunk size",
    intro: str = (
        "Mean ± SD across sessions (ms). All online stages from "
        "``latency_summary.csv``."
    ),
    unit_label: str = "ms",
) -> None:
    headers = ["Chunk", "Duration", "n"] + [
        f"{STAGE_LABELS[s]} ({unit_label})" for s in STAGES
    ]
    sep = ["------:", "---------:", "--:"] + ["---------:"] * len(STAGES)
    lines = [
        f"# {title}",
        "",
        intro,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in summary:
        cells = [
            str(int(row["chunk_size"])),
            f"{row['chunk_dur_s']:.2f}s",
            str(int(row["n_subjects"])),
        ]
        cells.extend(row[f"{s}_ms"] for s in STAGES)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# Reference window for unit-time stats: 1 s of data (250 samples @ 250 Hz).
UNIT_DUR_S = 1.0


def normalize_runs_per_second(runs: list[dict]) -> list[dict]:
    """Scale each run's stage means to cost per ``UNIT_DUR_S`` of EEG data.

    ``ms_per_sec = mean_ms_per_chunk / chunk_dur_s``.
    At 250 Hz this equals cost for a 250-sample (1 s) worth of data.
    """
    out: list[dict] = []
    for r in runs:
        dur = float(r["chunk_dur_s"])
        if not np.isfinite(dur) or dur <= 0:
            continue
        scale = UNIT_DUR_S / dur
        nr = dict(r)
        for stage in STAGES:
            mean = r.get(f"{stage}_mean_ms", float("nan"))
            std = r.get(f"{stage}_std_ms", float("nan"))
            nr[f"{stage}_mean_ms"] = (
                float(mean) * scale if np.isfinite(mean) else float("nan")
            )
            nr[f"{stage}_std_ms"] = (
                float(std) * scale if np.isfinite(std) else float("nan")
            )
        out.append(nr)
    return out


def plot_grouped_bars(
    summary: list[dict],
    path: Path,
    *,
    ylabel: str = "Computation time (ms / chunk)",
    title: str = "Per-chunk stage cost across sessions (mean ± SD)",
) -> None:
    """Single grouped-bar chart (small stages may be hard to see vs ICLabel)."""
    import matplotlib.pyplot as plt

    chunks = [int(r["chunk_size"]) for r in summary]
    x = np.arange(len(chunks), dtype=float)
    n = len(BAR_STAGES)
    width = min(0.12, 0.8 / max(n, 1))
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for i, stage in enumerate(BAR_STAGES):
        means = [r[f"{stage}_mean_ms"] for r in summary]
        stds = [r[f"{stage}_std_ms"] for r in summary]
        offset = (i - (n - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=2,
            color=STAGE_COLORS[stage],
            alpha=0.9,
            label=STAGE_LABELS[stage],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n({summary[i]['chunk_dur_s']:.1f}s)" for i, c in enumerate(chunks)]
    )
    ax.set_xlabel("Chunk size (samples)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _subplot_grid(n: int):
    """Choose a reasonable subplot layout for n stages."""
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 8:
        return 2, 4
    if n <= 9:
        return 3, 3
    nrows = int(np.ceil(n / 4))
    return nrows, 4


def plot_subject_variability(
    runs: list[dict],
    path: Path,
    *,
    ylabel: str = "ms / chunk",
    suptitle: str = "Cross-session variability of per-chunk computation time",
) -> None:
    """One panel per stage: subject means vs chunk size (variability)."""
    import matplotlib.pyplot as plt

    chunk_sizes = sorted({int(r["chunk_size"]) for r in runs})
    subjects = sorted({r["subject"] for r in runs})
    nrows, ncols = _subplot_grid(len(STAGES))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

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
                val = match[0][f"{stage}_mean_ms"]
                if not np.isfinite(val):
                    continue
                xs.append(cs)
                ys.append(val)
            if xs:
                ax.plot(xs, ys, marker="o", lw=1.0, ms=4, alpha=0.55)
        means, stds = [], []
        for cs in chunk_sizes:
            vals = [
                r[f"{stage}_mean_ms"]
                for r in runs
                if int(r["chunk_size"]) == cs and np.isfinite(r[f"{stage}_mean_ms"])
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
        ax.set_title(STAGE_LABELS[stage], fontsize=10)
        ax.set_xlabel("Chunk size (samples)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if stage == STAGES[0]:
            ax.legend(loc="upper left", fontsize=7)

    for ax in axes[len(STAGES) :]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(
    runs: list[dict],
    path: Path,
    *,
    ylabel: str = "ms / chunk",
    suptitle: str = "Distribution of subject means by chunk size",
) -> None:
    import matplotlib.pyplot as plt

    chunk_sizes = sorted({int(r["chunk_size"]) for r in runs})
    nrows, ncols = _subplot_grid(len(STAGES))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, stage in zip(axes, STAGES):
        data = []
        labels = []
        for cs in chunk_sizes:
            vals = [
                r[f"{stage}_mean_ms"]
                for r in runs
                if int(r["chunk_size"]) == cs and np.isfinite(r[f"{stage}_mean_ms"])
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
        ax.set_title(STAGE_LABELS[stage], fontsize=10)
        ax.set_xlabel("Chunk size (samples)")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes[len(STAGES) :]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate latency_summary.csv (all online stages) into "
            "Overleaf table + figures."
        )
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
    per_chunk_dir = out_dir / "per_chunk"
    per_sec_dir = out_dir / "per_sec"
    per_chunk_dir.mkdir(parents=True, exist_ok=True)
    per_sec_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(results_dir)
    if not runs:
        print(
            f"ERROR: no latency_summary.csv under {results_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    summary = aggregate_by_chunk(runs)
    runs_per_sec = normalize_runs_per_second(runs)
    summary_per_sec = aggregate_by_chunk(runs_per_sec)
    subjects = sorted({r["subject"] for r in runs})
    chunks = sorted({int(r["chunk_size"]) for r in runs})

    # Shared filenames inside each folder (folder name carries the unit).
    names = {
        "all_csv": "latency_all_runs.csv",
        "sum_csv": "latency_table_by_chunk.csv",
        "tex": "latency_table_by_chunk.tex",
        "md": "latency_table_by_chunk.md",
        "bars": "latency_by_chunk_bars.png",
        "var": "latency_cross_session_variability.png",
        "box": "latency_by_chunk_boxplot.png",
    }

    # --- per chunk (ms / chunk) ---
    write_all_runs_csv(runs, per_chunk_dir / names["all_csv"])
    write_summary_csv(summary, per_chunk_dir / names["sum_csv"])
    write_overleaf_tex(summary, per_chunk_dir / names["tex"])
    write_markdown_table(summary, per_chunk_dir / names["md"])
    plot_grouped_bars(summary, per_chunk_dir / names["bars"])
    plot_subject_variability(runs, per_chunk_dir / names["var"])
    plot_boxplot(runs, per_chunk_dir / names["box"])

    # --- per second of data (ms / s) ---
    write_all_runs_csv(runs_per_sec, per_sec_dir / names["all_csv"])
    write_summary_csv(summary_per_sec, per_sec_dir / names["sum_csv"])
    write_overleaf_tex(summary_per_sec, per_sec_dir / names["tex"])
    write_markdown_table(
        summary_per_sec,
        per_sec_dir / names["md"],
        title="Computation time per second of data",
        intro=(
            f"Mean ± SD across sessions, normalized to **{UNIT_DUR_S:.0f} s of EEG** "
            f"(= 250 samples @ 250 Hz). "
            f"Formula: ``ms_per_sec = ms_per_chunk / chunk_dur_s``. "
            "X-axis still shows the chunk size used when measuring."
        ),
        unit_label="ms/s",
    )
    plot_grouped_bars(
        summary_per_sec,
        per_sec_dir / names["bars"],
        ylabel="Computation time (ms / s of data)",
        title=(
            f"Stage cost per {UNIT_DUR_S:.0f} s of data "
            f"(normalized; 250 samples @ 250 Hz)"
        ),
    )
    plot_subject_variability(
        runs_per_sec,
        per_sec_dir / names["var"],
        ylabel="ms / s of data",
        suptitle=(
            f"Cross-session variability per {UNIT_DUR_S:.0f} s of data "
            f"(normalized; 250 samples @ 250 Hz)"
        ),
    )
    plot_boxplot(
        runs_per_sec,
        per_sec_dir / names["box"],
        ylabel="ms / s of data",
        suptitle=(
            f"Subject-mean distribution per {UNIT_DUR_S:.0f} s of data "
            f"(normalized; 250 samples @ 250 Hz)"
        ),
    )

    print(f"Runs loaded: {len(runs)}")
    print(f"Subjects: {len(subjects)}  ({', '.join(subjects)})")
    print(f"Chunk sizes: {chunks}")
    print(f"Stages: {', '.join(STAGES)}")
    print()
    print((per_chunk_dir / names["md"]).read_text(encoding="utf-8"))
    print()
    print((per_sec_dir / names["md"]).read_text(encoding="utf-8"))
    print(f"Wrote → {out_dir.resolve()}")
    print(f"  per_chunk/  ({', '.join(names.values())})")
    print(f"  per_sec/    ({', '.join(names.values())})")


if __name__ == "__main__":
    main()
