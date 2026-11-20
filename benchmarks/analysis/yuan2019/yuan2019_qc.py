# -*- coding: utf-8 -*-
"""Yuan 2019 cue-artifact dataset (Cognionics Quick-30 dry EEG) — loader + QC.

The dataset is unpublished and is NOT part of this repository. Point
PYORICA_YUAN2019_DATA at a folder that holds the raw recordings as
``sXX_eye.xdf`` / ``sXX_motion.xdf`` (s01..s10). Results go to
``benchmarks/results/`` (git-ignored) or to PYORICA_YUAN2019_RESULTS.

This module is also imported by the other yuan2019_* scripts for
``load`` (xdf -> arrays + markers) and ``blocks`` (marker pairs -> blocks).

No pipeline run. For every subject / file:
  * marker timeline relative to the EEG stream start (block start/end events,
    oddball span),
  * per-channel quality on the 29 scalp channels after a 1-50 Hz band-pass:
    robust SD, max |corr| with any other channel, median impedance.

Usage:  python yuan2019_qc.py [--timeline s03]
Outputs: <results>/yuan2019_qc/qc_channels.csv, qc_blocks.csv
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pyxdf
from scipy.signal import butter, sosfiltfilt

REPO = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PYORICA_YUAN2019_DATA", "data/yuan2019/raw"))
RESULTS = Path(os.environ.get("PYORICA_YUAN2019_RESULTS",
                              REPO / "benchmarks" / "results"))
OUT = RESULTS / "yuan2019_qc"
SCALP = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F8", "F3", "Fz", "F4", "FC5",
         "FC6", "T7", "T8", "C3", "Cz", "C4", "CP5", "CP6", "P7", "P8", "P3",
         "Pz", "P4", "PO7", "PO8", "PO3", "PO4", "O1", "O2"]
SUBJECTS = [f"s{i:02d}" for i in range(1, 11)]


def xdf_path(subj, part):
    p = RAW / f"{subj}_{part}.xdf"
    if not p.exists():
        p = RAW / f"pilot_{subj}_{part}.xdf"
    return p


def load(subj, part):
    """Return dict(eeg, sfreq, labels, t0, markers[(t_rel, label)], imp)."""
    streams, _ = pyxdf.load_xdf(str(xdf_path(subj, part)))
    out = {"imp": None}
    for s in streams:
        typ = s["info"]["type"][0]
        if typ == "EEG":
            chs = s["info"]["desc"][0]["channels"][0]["channel"]
            out["labels"] = [c["label"][0] for c in chs]
            out["eeg"] = np.asarray(s["time_series"], dtype=np.float64).T
            out["ts"] = np.asarray(s["time_stamps"])
            out["sfreq"] = float(s["info"]["nominal_srate"][0])
        elif typ == "Markers":
            out["_mk"] = (np.asarray(s["time_stamps"]),
                          [str(x[0]) for x in s["time_series"]])
        elif typ.lower().startswith("impe"):
            out["imp"] = np.asarray(s["time_series"], dtype=np.float64).T
    t0 = out["ts"][0]
    mt, ml = out.pop("_mk")
    out["markers"] = [(float(t - t0), lb) for t, lb in zip(mt, ml)]
    out["dur"] = float(out["ts"][-1] - t0)
    return out


def blocks(markers):
    """Pair X_start / X_end markers -> [(name, t_start, t_end)]."""
    open_, res = {}, []
    for t, lb in markers:
        if lb.endswith("_start"):
            open_[lb[:-6]] = t
        elif lb.endswith("_end"):
            nm = lb[:-4]
            if nm in open_:
                res.append((nm, open_.pop(nm), t))
    for nm, t in open_.items():
        res.append((nm, t, float("nan")))
    return sorted(res, key=lambda r: r[1])


def channel_qc(d):
    idx = [d["labels"].index(c) for c in SCALP]
    x = d["eeg"][idx]
    sos = butter(4, [1.0, 50.0], btype="band", fs=d["sfreq"], output="sos")
    xf = sosfiltfilt(sos, x, axis=1)
    med = np.median(xf, axis=1, keepdims=True)
    rsd = 1.4826 * np.median(np.abs(xf - med), axis=1)
    # correlation on 4x decimated data is plenty for QC
    c = np.corrcoef(xf[:, ::4])
    np.fill_diagonal(c, 0.0)
    maxcorr = np.max(np.abs(c), axis=1)
    imp = (np.median(d["imp"][idx], axis=1) if d["imp"] is not None
           else np.full(len(idx), np.nan))
    return rsd, maxcorr, imp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tl = sys.argv[sys.argv.index("--timeline") + 1] if "--timeline" in sys.argv else None
    ch_rows, blk_rows = [], []
    for subj in SUBJECTS:
        for part in ("eye", "motion"):
            d = load(subj, part)
            bl = blocks(d["markers"])
            for nm, a, b in bl:
                blk_rows.append({"subject": subj, "file": part, "block": nm,
                                 "t_start": round(a, 2), "t_end": round(b, 2)})
            rsd, mc, imp = channel_qc(d)
            ref = np.median(rsd)
            flags = []
            for c, r, m, z in zip(SCALP, rsd, mc, imp):
                bad = []
                if r > 3 * ref:
                    bad.append("noisy")
                if r < ref / 4:
                    bad.append("flat")
                if m < 0.4:
                    bad.append("uncorr")
                ch_rows.append({"subject": subj, "file": part, "ch": c,
                                "rsd_uV": round(float(r), 2),
                                "max_abs_corr": round(float(m), 3),
                                "imp_kohm": round(float(z), 1),
                                "flag": "+".join(bad)})
                if bad:
                    flags.append(f"{c}({'+'.join(bad)} sd={r:.0f} r={m:.2f})")
            odd = [t for t, lb in d["markers"] if lb in ("odd_0", "odd_1")]
            odd_s = (f" | oddball {odd[0]/60:.2f}-{odd[-1]/60:.2f} min, "
                     f"n={len(odd)}" if odd else "")
            print(f"{subj} {part:6} dur {d['dur']/60:5.2f} min, "
                  f"first marker {d['markers'][0][0]:.1f}s, median chan SD "
                  f"{ref:5.1f} uV, median imp {np.nanmedian(imp):6.0f}k"
                  f"{odd_s}\n      flagged: {', '.join(flags) or '-'}",
                  flush=True)
            if tl == subj:
                for nm, a, b in bl:
                    print(f"        {a/60:6.2f}-{b/60:6.2f} min  {nm}",
                          flush=True)
    for name, rows in (("qc_channels.csv", ch_rows), ("qc_blocks.csv", blk_rows)):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"saved qc_channels.csv, qc_blocks.csv -> {OUT}")


if __name__ == "__main__":
    main()
