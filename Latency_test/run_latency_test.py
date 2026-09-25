"""Latency benchmark matching ``benchmarks/run_validation.py`` flow.

Flow
----
1. Load one NCTU subject (same ``.set`` loader as run_validation)
2. **Calibration once** on the first ``asr_calibration_seconds`` (120 s in
   ``reference.yaml``): IIR filtfilt → ASR.fit → ASR.transform → ORICA.fit
3. **Online stream** for ``--stream-seconds`` (default **0 = full recording**):
   each chunk times IIR / ASR / ORICA / ICLabel / reconstruct
4. Writes detailed CSVs/plots plus a one-row ``latency_table_row.csv``
   (ASR / ORICA / ICLabel / Total as mean±std) for later chunk/subject merge.

Real-time budget
----------------
At ``sfreq`` Hz, ``chunk_size`` samples represent
``chunk_dur_s = chunk_size / sfreq`` seconds of EEG.
Processing that chunk must finish in ``< chunk_dur_s`` wall time
(e.g. 1000 samples @ 250 Hz → **4.0 s budget**; equivalently 250 samples → 1 s).

This script does **not** auto-raise ``chunk_size`` to the ICLabel ~3.5 s
minimum (unlike ``run_validation``). Short chunks are allowed so you can
profile smaller delays. Below that floor, ``ICLabelClassifier`` normally
no-ops unless you pass ``--force-iclabel`` (bypasses the guard in-process).

Usage
-----
    $env:PYORICA_NCTU_DATA = \"D:\\work\\Python_Project\\pyorica\\data\\input_data\"
    python Latency_test/run_latency_test.py --subject s01 --stream-seconds 600
    python Latency_test/run_latency_test.py --subject s01 --chunk-size 500 --stream-seconds 600
    python Latency_test/run_latency_test.py --subject s01 --chunk-size 250 --force-iclabel

``--output-dir`` is only the parent folder. The run folder name is built
from the other flags, e.g. ``Latency_test/results/s01_chunk250_full``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_LATENCY_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _LATENCY_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from benchmarks.run_validation import (  # noqa: E402
    CHUNK_SIZE,
    _find_sessions,
    _load_set,
    _make_mne_info,
)

ONLINE_STAGES = (
    "iir",
    "asr",
    "orica_update",
    "orica_transform",
    "unmixing_pinv",
    "icalabel",
    "reconstruct",
    "chunk_total",
)

CALIB_STAGES = (
    "calib_iir_filtfilt",
    "calib_asr_fit",
    "calib_asr_transform",
    "calib_orica_fit",
    "calib_total",
)

STAGE_COLORS = {
    "iir": "#1f77b4",
    "asr": "#ff7f0e",
    "orica_update": "#2ca02c",
    "orica_transform": "#98df8a",
    "unmixing_pinv": "#c5b0d5",
    "icalabel": "#d62728",
    "reconstruct": "#17becf",
}


def _ms(dt: float) -> float:
    return float(dt) * 1000.0


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "sum_ms": 0.0,
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "sum_ms": float(np.sum(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _mean_pm_std(mean: float, std: float) -> str:
    """Format like the paper table: ``3.2 ± 0.5``."""
    if not np.isfinite(mean) or not np.isfinite(std):
        return "nan"
    return f"{mean:.1f} ± {std:.1f}"


def _table_arrays(per_chunk: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """ASR / ORICA / ICLabel / Total arrays for the subject table row.

    ORICA = orica_update + orica_transform.
    Total = ASR + ORICA + ICLabel (same columns as the summary table).
    """
    asr = np.asarray(per_chunk["asr"], dtype=np.float64)
    orica = (
        np.asarray(per_chunk["orica_update"], dtype=np.float64)
        + np.asarray(per_chunk["orica_transform"], dtype=np.float64)
    )
    icalabel = np.asarray(per_chunk["icalabel"], dtype=np.float64)
    total = asr + orica + icalabel
    return {"asr": asr, "orica": orica, "icalabel": icalabel, "total": total}


def _write_table_row(
    out_dir: Path,
    *,
    subject: str,
    chunk_size: int,
    chunk_dur_s: float,
    sfreq: float,
    n_chunks: int,
    stream_seconds_actual: float,
    force_iclabel: bool,
    per_chunk: dict[str, list[float]],
) -> tuple[Path, Path]:
    """One-row CSV + bar plot for later multi-chunk / multi-subject merge."""
    import matplotlib.pyplot as plt

    cols = _table_arrays(per_chunk)
    stats = {k: _stats(v.tolist()) for k, v in cols.items()}

    row = {
        "subject": subject,
        "chunk_size": int(chunk_size),
        "chunk_dur_s": float(chunk_dur_s),
        "sfreq_hz": float(sfreq),
        "n_chunks": int(n_chunks),
        "stream_seconds": float(stream_seconds_actual),
        "force_iclabel": bool(force_iclabel),
        "asr_mean_ms": stats["asr"]["mean_ms"],
        "asr_std_ms": stats["asr"]["std_ms"],
        "asr_ms": _mean_pm_std(stats["asr"]["mean_ms"], stats["asr"]["std_ms"]),
        "orica_mean_ms": stats["orica"]["mean_ms"],
        "orica_std_ms": stats["orica"]["std_ms"],
        "orica_ms": _mean_pm_std(stats["orica"]["mean_ms"], stats["orica"]["std_ms"]),
        "icalabel_mean_ms": stats["icalabel"]["mean_ms"],
        "icalabel_std_ms": stats["icalabel"]["std_ms"],
        "icalabel_ms": _mean_pm_std(
            stats["icalabel"]["mean_ms"], stats["icalabel"]["std_ms"]
        ),
        "total_mean_ms": stats["total"]["mean_ms"],
        "total_std_ms": stats["total"]["std_ms"],
        "total_ms": _mean_pm_std(stats["total"]["mean_ms"], stats["total"]["std_ms"]),
    }

    csv_path = out_dir / "latency_table_row.csv"
    fieldnames = list(row.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    # Compact bar figure (mean ± std) for this single chunk size
    labels = ["ASR", "ORICA", "ICLabel", "Total"]
    keys = ["asr", "orica", "icalabel", "total"]
    means = [stats[k]["mean_ms"] for k in keys]
    stds = [stats[k]["std_ms"] for k in keys]
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (ms)")
    ax.set_title(
        f"{subject}  |  chunk={chunk_size} ({chunk_dur_s:.2f}s)  |  "
        f"n={n_chunks}  |  mean ± std"
    )
    ax.grid(True, axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s, _mean_pm_std(m, s), ha="center", va="bottom", fontsize=8)
    fig_path = out_dir / "latency_table_row.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return csv_path, fig_path


def _auto_run_dirname(subject: str, chunk_size: int, stream_seconds: float) -> str:
    """Folder name from CLI params, e.g. ``s01_chunk250_full`` or ``s01_chunk1000_sec60``."""
    if float(stream_seconds) <= 0:
        stream_tag = "full"
    else:
        stream_tag = "sec" + f"{float(stream_seconds):g}".replace(".", "p")
    return f"{subject}_chunk{int(chunk_size)}_{stream_tag}"


def _resolve_subject_set(data_root: Path, subject: str) -> Path:
    sessions = _find_sessions(data_root)
    for p in sessions:
        if p.parent.name == subject:
            return p
    raise FileNotFoundError(
        f"Subject '{subject}' not found under {data_root} "
        f"(looked for s*/s*_resampled.set)"
    )


def _plot_latency(
    out_dir: Path,
    *,
    subject: str,
    sfreq: float,
    chunk_size: int,
    stream_seconds: float,
    per_chunk: dict[str, list[float]],
    calib_times: dict[str, float],
) -> Path:
    import matplotlib.pyplot as plt

    chunk_dur_s = chunk_size / sfreq
    budget_ms = chunk_dur_s * 1000.0  # real-time budget for one chunk
    # Per 1 s of EEG: processing must be < 1000 ms
    budget_per_sec_ms = 1000.0

    totals = np.asarray(per_chunk["chunk_total"], dtype=np.float64)
    n = len(totals)
    t_axis = (np.arange(n) * chunk_size) / sfreq  # stream time at chunk start
    overrun = totals > budget_ms
    n_over = int(np.sum(overrun))

    # ms to process 1 s of data
    ms_per_sec = totals / chunk_dur_s
    overrun_1s = ms_per_sec > budget_per_sec_ms

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    # (1) Per-chunk total vs budget
    ax = axes[0]
    ax.plot(t_axis, totals, color="#1f77b4", lw=0.9, label="chunk_total")
    if n_over:
        ax.scatter(
            t_axis[overrun],
            totals[overrun],
            c="#d62728",
            s=18,
            zorder=3,
            label=f"overrun ({n_over}/{n})",
        )
    ax.axhline(
        budget_ms,
        color="#d62728",
        ls="--",
        lw=1.5,
        label=f"real-time budget = {budget_ms:.0f} ms "
        f"({chunk_size} samp / {sfreq:g} Hz = {chunk_dur_s:.2f} s)",
    )
    ax.set_ylabel("Processing time (ms)")
    ax.set_xlabel("Stream time (s)")
    ax.set_title(
        f"{subject}: per-chunk latency vs real-time budget "
        f"(overruns {n_over}/{n} = {100.0 * n_over / max(n, 1):.1f}%)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) Equivalent ms per 1 s of EEG
    ax = axes[1]
    ax.plot(t_axis, ms_per_sec, color="#ff7f0e", lw=0.9, label="ms to process 1 s EEG")
    ax.axhline(
        budget_per_sec_ms,
        color="#d62728",
        ls="--",
        lw=1.5,
        label="budget = 1000 ms / 1 s EEG (250 samples @ 250 Hz)",
    )
    if np.any(overrun_1s):
        ax.scatter(
            t_axis[overrun_1s],
            ms_per_sec[overrun_1s],
            c="#d62728",
            s=18,
            zorder=3,
            label=f"over 1s-budget ({int(np.sum(overrun_1s))}/{n})",
        )
    ax.set_ylabel("ms / second of EEG")
    ax.set_xlabel("Stream time (s)")
    ax.set_title(
        f"Normalized load (mean {np.mean(ms_per_sec):.1f} ms/s EEG, "
        f"RTF={np.mean(totals) / budget_ms:.3f})"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) Mean stage breakdown vs chunk budget
    ax = axes[2]
    stages_bar = [
        "iir",
        "asr",
        "orica_update",
        "orica_transform",
        "unmixing_pinv",
        "icalabel",
        "reconstruct",
    ]
    means = [float(np.mean(per_chunk[s])) for s in stages_bar]
    colors = [STAGE_COLORS.get(s, "C0") for s in stages_bar]
    x = np.arange(len(stages_bar))
    ax.bar(x, means, color=colors, alpha=0.9)
    ax.axhline(budget_ms, color="#d62728", ls="--", lw=1.2, label=f"chunk budget {budget_ms:.0f} ms")
    ax.set_xticks(x)
    ax.set_xticklabels(stages_bar, rotation=25, ha="right")
    ax.set_ylabel("Mean time (ms)")
    ax.set_title(
        "Mean per-chunk stage cost "
        f"(calib once: ASR.fit={calib_times['calib_asr_fit']:.0f} ms, "
        f"total={calib_times['calib_total']:.0f} ms)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Latency vs real-time  |  stream={stream_seconds:g}s  "
        f"| chunk={chunk_size} @ {sfreq:g} Hz  "
        f"| budget {chunk_dur_s:.2f}s/chunk",
        fontsize=12,
    )
    out_path = out_dir / "latency_realtime.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_latency(
    set_path: Path,
    config,
    out_dir: Path,
    *,
    stream_seconds: float = 600.0,
    chunk_size: Optional[int] = None,
    force_iclabel: bool = False,
) -> Path:
    """Calibrate like run_validation, then time online stages for stream_seconds."""
    from scipy.signal import sosfiltfilt

    from pyorica.pipeline.classify import ICLabelClassifier
    from pyorica.pipeline.pipeline import EEGPipeline
    from pyorica.streaming.array import ArrayStream

    subject = set_path.parent.name
    print(f"[{subject}] loading {set_path}...")
    data_full, sfreq, ch_names = _load_set(set_path)
    n_ch, n_full = data_full.shape
    full_dur = n_full / sfreq

    calib_seconds = float(config.asr_calibration_seconds)
    calib_samples = int(calib_seconds * sfreq)
    stream_seconds_requested = float(stream_seconds)
    # stream_seconds <= 0 → use the full recording
    if stream_seconds_requested <= 0:
        stream_samples = n_full
        stream_seconds = full_dur
    else:
        stream_samples = int(stream_seconds_requested * sfreq)

    if n_full < calib_samples:
        raise ValueError(
            f"Recording too short for full calibration: have {full_dur:.1f} s, "
            f"need asr_calibration_seconds={calib_seconds:g} s "
            f"(same as run_validation / reference.yaml)."
        )
    if n_full < stream_samples:
        print(
            f"[{subject}] WARNING: requested stream {stream_seconds_requested:g} s but "
            f"recording is only {full_dur:.1f} s — using full recording.",
            file=sys.stderr,
        )
        stream_samples = n_full
        stream_seconds = full_dur

    # Like run_validation --max-seconds stream_seconds:
    # truncate session to stream length, calib = first calib_seconds of that.
    data = data_full[:, :stream_samples]
    n_samples = data.shape[1]
    calibration = data_full[:, :calib_samples]  # always full 120 s from original

    cfg_chunk = int(getattr(config, "chunk_size", CHUNK_SIZE) or CHUNK_SIZE)
    chunk_size = int(chunk_size) if chunk_size is not None else cfg_chunk
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    # ICLabelClassifier guard (classify.py): < ~3.5 s → no-op unless forced here.
    min_iclabel = int(np.ceil(sfreq * 3.5))
    # ICLabel PSD uses FFT length min(n_times, sfreq) but always writes `nfreqs`
    # bins (min(nyquist, 100)). A short tail (e.g. 50 samples) cannot fill that
    # and crashes inside mne-icalabel. Full chunks at/above this length are fine.
    nyquist = int(np.floor(sfreq / 2.0))
    nfreqs = nyquist if nyquist < 100 else 100
    iclabel_psd_min = int(nfreqs + 1)
    if chunk_size < min_iclabel:
        if force_iclabel:
            print(
                f"[{subject}] NOTE: chunk_size {chunk_size} < ICLabel floor "
                f"{min_iclabel} (~3.5 s); --force-iclabel will bypass the "
                f"classifier length guard (may warn/fail inside MNE FIR).",
                file=sys.stderr,
            )
        else:
            print(
                f"[{subject}] NOTE: chunk_size {chunk_size} < ICLabel floor "
                f"{min_iclabel} (~3.5 s). Keeping short chunks (no auto-raise). "
                f"ICLabel will no-op each call; pass --force-iclabel to bypass.",
                file=sys.stderr,
            )

    chunk_dur_s = chunk_size / sfreq
    budget_ms = chunk_dur_s * 1000.0

    print(
        f"[{subject}] {n_ch} ch @ {sfreq} Hz | full recording {full_dur:.1f} s\n"
        f"  calib:  {calib_samples / sfreq:.1f} s  (run_validation style)\n"
        f"  stream: {n_samples / sfreq:.1f} s  ({n_samples} samples)\n"
        f"  chunk:  {chunk_size} samples = {chunk_dur_s:.3f} s EEG  "
        f"→ real-time budget {budget_ms:.1f} ms/chunk"
        f"{'  [force_iclabel]' if force_iclabel else ''}"
    )

    info = _make_mne_info(ch_names, sfreq)
    classifier = ICLabelClassifier(
        info,
        threshold=config.icalabel_threshold,
        apply_car_bandpass=config.icalabel_apply_car_bandpass,
        record_snapshots=False,
    )

    # NOTE: assigning instance.__call__ does NOT affect classifier(...) —
    # Python looks up __call__ on the class. Use an explicit callable instead.
    if force_iclabel:
        def classify_fn(data, sources, unmixing, mixing, sfreq_):
            n_components = sources.shape[0]
            label_strings, prob_top1 = classifier._run_icalabel(
                data, unmixing, mixing, sfreq_, n_components
            )
            return classifier._artifact_mask(label_strings, prob_top1)
    else:
        def classify_fn(data, sources, unmixing, mixing, sfreq_):
            return classifier(data, sources, unmixing, mixing, sfreq_)

    pipeline = EEGPipeline(
        n_channels=n_ch, sfreq=sfreq, classifier=classifier, verbose=False, config=config
    )

    # ── Calibration (once) — same steps as EEGPipeline.fit ──────────────
    calib_times: dict[str, float] = {}
    t_calib0 = time.perf_counter()

    t0 = time.perf_counter()
    iir_filtered = sosfiltfilt(
        pipeline._iir._sos, np.asarray(calibration, dtype=np.float64), axis=1
    )
    calib_times["calib_iir_filtfilt"] = _ms(time.perf_counter() - t0)

    t0 = time.perf_counter()
    pipeline._asr.fit(iir_filtered)
    pipeline._asr_fitted = True
    calib_times["calib_asr_fit"] = _ms(time.perf_counter() - t0)

    t0 = time.perf_counter()
    orica_input = pipeline._asr.transform(iir_filtered)
    calib_times["calib_asr_transform"] = _ms(time.perf_counter() - t0)

    t0 = time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api="blas"):
            pipeline.orica.fit(orica_input)
    except ImportError:
        pipeline.orica.fit(orica_input)
    calib_times["calib_orica_fit"] = _ms(time.perf_counter() - t0)
    calib_times["calib_total"] = _ms(time.perf_counter() - t_calib0)

    print(f"[{subject}] calibration done  ({calib_times['calib_total']:.1f} ms)")
    for k in CALIB_STAGES:
        print(f"  {k:24s}  {calib_times[k]:10.2f} ms")

    # ── Online streaming ────────────────────────────────────────────────
    per_chunk: dict[str, list[float]] = {k: [] for k in ONLINE_STAGES}
    n_classify_calls = 0
    n_classify_skipped = 0

    classify_interval_samples = int(sfreq * float(config.classify_interval_s))
    samples_since_classify = 0
    cached_mask = None

    stream = ArrayStream(data, chunk_size=chunk_size)
    n_chunks = int(np.ceil(n_samples / chunk_size))
    print(f"[{subject}] streaming {n_chunks} chunks...")

    t_stream0 = time.perf_counter()
    for chunk_idx, chunk in enumerate(stream):
        t_chunk0 = time.perf_counter()

        t0 = time.perf_counter()
        out = pipeline._iir.process(chunk)
        per_chunk["iir"].append(_ms(time.perf_counter() - t0))

        t0 = time.perf_counter()
        if pipeline._asr_fitted:
            out = pipeline._asr.transform(out)
        per_chunk["asr"].append(_ms(time.perf_counter() - t0))

        t0 = time.perf_counter()
        pipeline.orica.update(out)
        per_chunk["orica_update"].append(_ms(time.perf_counter() - t0))

        t0 = time.perf_counter()
        sources = pipeline.orica.transform(out)
        per_chunk["orica_transform"].append(_ms(time.perf_counter() - t0))

        t0 = time.perf_counter()
        unmixing = pipeline.orica.weights_ @ pipeline.orica.sphere_
        mixing = np.linalg.pinv(unmixing)
        per_chunk["unmixing_pinv"].append(_ms(time.perf_counter() - t0))

        t0 = time.perf_counter()
        n_this = int(out.shape[1])
        # Tail shorter than one full chunk, or shorter than the PSD FFT, cannot
        # run real ICLabel. Skip it so the rest of the recording is still saved.
        skip_real_iclabel = n_this < chunk_size or n_this < iclabel_psd_min
        if classify_interval_samples == 0:
            if force_iclabel and skip_real_iclabel:
                mask = np.zeros(sources.shape[0], dtype=bool)
                n_classify_calls += 1
                n_classify_skipped += 1
                print(
                    f"[{subject}] skip ICLabel on chunk {chunk_idx} "
                    f"({n_this} samples < chunk {chunk_size} or PSD min {iclabel_psd_min})",
                    file=sys.stderr,
                )
            else:
                try:
                    mask = classify_fn(out, sources, unmixing, mixing, sfreq)
                except Exception as exc:
                    print(
                        f"[{subject}] ICLabel failed on chunk {chunk_idx} "
                        f"({n_this} samples): {exc}; skipping this chunk",
                        file=sys.stderr,
                    )
                    mask = np.zeros(sources.shape[0], dtype=bool)
                    n_classify_skipped += 1
                sources[mask] = 0.0
                n_classify_calls += 1
                if (not force_iclabel) and n_this < min_iclabel:
                    n_classify_skipped += 1
        else:
            if cached_mask is not None:
                sources[cached_mask] = 0.0
            samples_since_classify += out.shape[1]
            if samples_since_classify >= classify_interval_samples:
                cached_mask = classify_fn(out, sources, unmixing, mixing, sfreq)
                samples_since_classify %= classify_interval_samples
                n_classify_calls += 1
                if (not force_iclabel) and out.shape[1] < min_iclabel:
                    n_classify_skipped += 1
        per_chunk["icalabel"].append(_ms(time.perf_counter() - t0))

        t0 = time.perf_counter()
        _ = pipeline.orica.inverse_transform(sources)
        per_chunk["reconstruct"].append(_ms(time.perf_counter() - t0))

        per_chunk["chunk_total"].append(_ms(time.perf_counter() - t_chunk0))

        if (chunk_idx + 1) % 25 == 0 or (chunk_idx + 1) == n_chunks:
            elapsed = time.perf_counter() - t_stream0
            last = per_chunk["chunk_total"][-1]
            flag = " OVER" if last > budget_ms else ""
            print(
                f"[{subject}] chunk {chunk_idx + 1}/{n_chunks}  "
                f"({100.0 * (chunk_idx + 1) / n_chunks:.0f}%)  "
                f"last={last:.1f}ms / budget={budget_ms:.0f}ms{flag}  "
                f"elapsed {elapsed:.1f}s"
            )

    stream_total_ms = _ms(time.perf_counter() - t_stream0)
    print(f"[{subject}] streaming done  ({stream_total_ms:.1f} ms)")

    totals = np.asarray(per_chunk["chunk_total"], dtype=np.float64)
    n_over = int(np.sum(totals > budget_ms))
    rtf = float(np.mean(totals) / budget_ms) if budget_ms > 0 else float("nan")

    # ── Write outputs ───────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(out_dir / "config.yaml")

    # CLI / run parameters (stream length is not in PipelineConfig)
    data_dur_s = n_samples / sfreq
    run_meta = {
        "subject": subject,
        "set_path": str(set_path),
        "sfreq_hz": float(sfreq),
        "n_channels": int(n_ch),
        "stream_seconds_requested": float(stream_seconds_requested),
        "stream_seconds_actual": float(data_dur_s),
        "calib_seconds": float(calib_samples / sfreq),
        "chunk_size": int(chunk_size),
        "chunk_dur_s": float(chunk_dur_s),
        "iclabel_min_samples": int(min_iclabel),
        "force_iclabel": bool(force_iclabel),
        "real_time_budget_ms_per_chunk": float(budget_ms),
        "n_chunks": int(len(totals)),
        "classify_interval_s": float(config.classify_interval_s),
    }
    run_meta_path = out_dir / "run_meta.yaml"
    try:
        import yaml

        with open(run_meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(run_meta, f, sort_keys=False, allow_unicode=True)
    except Exception:
        # Fallback if PyYAML unavailable
        run_meta_path.write_text(
            "\n".join(f"{k}: {v}" for k, v in run_meta.items()) + "\n",
            encoding="utf-8",
        )

    summary_rows = []
    for stage in CALIB_STAGES:
        summary_rows.append(
            {
                "phase": "calibration",
                "stage": stage,
                "count": 1,
                "mean_ms": calib_times[stage],
                "std_ms": 0.0,
                "min_ms": calib_times[stage],
                "max_ms": calib_times[stage],
                "sum_ms": calib_times[stage],
                "p50_ms": calib_times[stage],
                "p95_ms": calib_times[stage],
            }
        )
    for stage in ONLINE_STAGES:
        st = _stats(per_chunk[stage])
        summary_rows.append({"phase": "online", "stage": stage, **st})

    summary_path = out_dir / "latency_summary.csv"
    fieldnames = [
        "phase",
        "stage",
        "count",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "sum_ms",
        "p50_ms",
        "p95_ms",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    per_chunk_path = out_dir / "latency_per_chunk.csv"
    with open(per_chunk_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_idx",
                "stream_time_s",
                "budget_ms",
                "over_budget",
                *ONLINE_STAGES,
            ],
        )
        writer.writeheader()
        for i in range(len(totals)):
            row = {
                "chunk_idx": i,
                "stream_time_s": i * chunk_size / sfreq,
                "budget_ms": budget_ms,
                "over_budget": int(totals[i] > budget_ms),
            }
            for stage in ONLINE_STAGES:
                row[stage] = per_chunk[stage][i]
            writer.writerow(row)

    plot_path = _plot_latency(
        out_dir,
        subject=subject,
        sfreq=sfreq,
        chunk_size=chunk_size,
        stream_seconds=data_dur_s,
        per_chunk=per_chunk,
        calib_times=calib_times,
    )

    table_csv_path, table_fig_path = _write_table_row(
        out_dir,
        subject=subject,
        chunk_size=chunk_size,
        chunk_dur_s=chunk_dur_s,
        sfreq=sfreq,
        n_chunks=len(totals),
        stream_seconds_actual=data_dur_s,
        force_iclabel=force_iclabel,
        per_chunk=per_chunk,
    )

    report_path = out_dir / "latency_report.txt"
    lines = [
        f"subject: {subject}",
        f"set: {set_path}",
        f"sfreq: {sfreq} Hz",
        f"n_channels: {n_ch}",
        f"calib_seconds: {calib_samples / sfreq:.2f}  (full reference.yaml calib)",
        f"stream_seconds_requested: {stream_seconds_requested:.2f}"
        f"{'  (0 = full recording)' if stream_seconds_requested <= 0 else ''}",
        f"stream_seconds_actual: {data_dur_s:.2f}",
        f"chunk_size: {chunk_size}  (= {chunk_dur_s:.3f} s of EEG)",
        f"iclabel_min_samples: {min_iclabel}  (~3.5 s; classify.py guard)",
        f"force_iclabel: {force_iclabel}",
        f"real_time_budget_ms_per_chunk: {budget_ms:.2f}",
        f"n_chunks: {len(totals)}",
        f"chunks_over_budget: {n_over} / {len(totals)} "
        f"({100.0 * n_over / max(len(totals), 1):.1f}%)",
        f"mean_chunk_ms: {float(np.mean(totals)):.2f}",
        f"real_time_factor: {rtf:.4f}  (<1 OK; >1 cannot keep up)",
        f"classify_interval_s: {config.classify_interval_s}",
        f"icalabel_calls: {n_classify_calls} "
        f"(short-chunk classifier no-ops≈{n_classify_skipped})",
        f"stream_wall_ms: {stream_total_ms:.2f}",
        "",
        "=== Table row (ASR / ORICA / ICLabel / Total), mean ± std ms ===",
    ]
    # Append human-readable table row from CSV content
    with open(table_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        trow = next(reader)
    lines.append(
        f"  chunk={trow['chunk_size']}  "
        f"ASR {trow['asr_ms']}  ORICA {trow['orica_ms']}  "
        f"ICLabel {trow['icalabel_ms']}  Total {trow['total_ms']}"
    )
    lines.append("")
    lines.append("=== Calibration (once, ms) — same as run_validation / EEGPipeline.fit ===")
    for stage in CALIB_STAGES:
        lines.append(f"  {stage:24s}  {calib_times[stage]:10.2f}")
    lines.append("")
    lines.append("=== Online per-chunk (ms) ===")
    lines.append(
        f"  {'stage':20s}  {'n':>6s}  {'mean':>10s}  {'std':>10s}  "
        f"{'p50':>10s}  {'p95':>10s}  {'sum':>12s}"
    )
    for stage in ONLINE_STAGES:
        st = _stats(per_chunk[stage])
        lines.append(
            f"  {stage:20s}  {st['count']:6d}  {st['mean_ms']:10.3f}  "
            f"{st['std_ms']:10.3f}  {st['p50_ms']:10.3f}  {st['p95_ms']:10.3f}  "
            f"{st['sum_ms']:12.1f}"
        )
    lines.append("")
    lines.append(f"Plot: {plot_path.name}, {table_fig_path.name}")
    lines.append(
        f"Wrote: {run_meta_path.name}, {summary_path.name}, {per_chunk_path.name}, "
        f"{table_csv_path.name}"
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(report_path.read_text(encoding="utf-8"))
    print(f"Results → {out_dir.resolve()}")
    return summary_path


def main() -> None:
    from pyorica.config import PipelineConfig

    parser = argparse.ArgumentParser(
        description=(
            "Time IIR/ASR/ORICA/ICLabel like run_validation: "
            "full calib then stream N seconds; plot vs real-time budget"
        )
    )
    parser.add_argument("--subject", default="s01")
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "benchmarks" / "config" / "reference.yaml"),
    )
    parser.add_argument(
        "--stream-seconds",
        type=float,
        default=0.0,
        metavar="S",
        help=(
            "Online stream length in seconds after calib. "
            "Use 0 (default) for the FULL recording."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Samples per online chunk (overrides config chunk_size). "
            "Does not auto-raise to the ICLabel ~3.5 s floor."
        ),
    )
    parser.add_argument(
        "--force-iclabel",
        action="store_true",
        help=(
            "Bypass ICLabelClassifier's ~3.5 s length guard in this script only "
            "(real ICLabel path; may fail/warn on very short chunks)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Parent folder only. A subfolder is created from the other flags "
            "(default parent: Latency_test/results). "
            "Example: --output-dir Latency_test/results "
            "→ Latency_test/results/s01_chunk250_full"
        ),
    )
    args = parser.parse_args()

    data_root_env = os.environ.get("PYORICA_NCTU_DATA", "")
    if not data_root_env:
        fallback = _REPO_ROOT / "data" / "input_data"
        if fallback.is_dir():
            data_root_env = str(fallback)
            print(f"PYORICA_NCTU_DATA not set; using {fallback}")
        else:
            print("ERROR: set PYORICA_NCTU_DATA", file=sys.stderr)
            sys.exit(1)

    data_root = Path(data_root_env)
    set_path = _resolve_subject_set(data_root, args.subject)
    config = PipelineConfig.from_yaml(args.config)

    stream_seconds = float(args.stream_seconds)
    cfg_chunk = int(getattr(config, "chunk_size", CHUNK_SIZE) or CHUNK_SIZE)
    chunk_size = int(args.chunk_size) if args.chunk_size is not None else cfg_chunk
    parent = Path(args.output_dir) if args.output_dir else (_LATENCY_ROOT / "results")
    out_dir = parent / _auto_run_dirname(args.subject, chunk_size, stream_seconds)
    print(f"Output → {out_dir}")

    run_latency(
        set_path,
        config,
        out_dir,
        stream_seconds=stream_seconds,
        chunk_size=args.chunk_size,
        force_iclabel=bool(args.force_iclabel),
    )


if __name__ == "__main__":
    main()
