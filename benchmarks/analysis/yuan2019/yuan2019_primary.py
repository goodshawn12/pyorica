# -*- coding: utf-8 -*-
"""Yuan 2019 cue-artifact dataset (Cognionics Quick-30 dry EEG, 10 subjects):
same three-stage evaluation as the driving / PhysioMotion primary results.

Per subject: the MOTION recording only (~21 min: eyes-closed rest, oddball,
then head shake / nod / walk blocks with concurrent oddball), 29 scalp
channels, 500 -> 250 Hz, reference config (benchmarks/config/reference.yaml,
NOT the tuned COMBO). ASR + ORICA calibration is selected with --calib (see
build_calib); the default "first120" is causal and follows the
driving-dataset rule. Evaluation decomposition: extended Infomax (seed 42,
max_iter 500) fit on the IIR stage, ICLabel-labeled; all stages projected
through the same unmixing. The whole-session CSV written here uses samples
after CONV_MIN minutes only; the per-condition analysis
(yuan2019_percond_ica.py) works from the saved stage streams instead.

Also saved per subject for later cue-locked / P300 analyses: stage streams
(float32), window-level ICLabel labels/probs/mask + ORICA mixings, markers.

Resumable (per-subject done markers). Statistics are deliberately NOT run
here: `--describe` prints class counts and distributions only.

Data: set PYORICA_YUAN2019_DATA (see yuan2019_qc.py). The dataset is
unpublished and not part of this repository.

Usage (from the repo root):
    python benchmarks/analysis/yuan2019/yuan2019_primary.py --only s03
    python benchmarks/analysis/yuan2019/yuan2019_primary.py          # all ten
    python benchmarks/analysis/yuan2019/yuan2019_primary.py --calib eye_rest
    python benchmarks/analysis/yuan2019/yuan2019_primary.py --part eye
    python benchmarks/analysis/yuan2019/yuan2019_primary.py --describe
Output: <results>/yuan2019_primary_<calib>/  (eye recording:
<results>/yuan2019_eye_first_eor/).
"""
import csv
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(Path(__file__).parent))

from yuan2019_qc import RESULTS, SCALP, SUBJECTS, blocks, load  # noqa: E402

REF_CFG = REPO / "benchmarks" / "config" / "reference.yaml"
VARIANT = "first120"
PART = "motion"
OUT = RESULTS / f"yuan2019_primary_{VARIANT}"
CSV = OUT / "yuan_retention_ics.csv"

TARGET_SFREQ = 250.0
CONV_MIN = 8.0
CALIB_BLOCK = "eor"
CALIB_TRIM_S = 2.0
CLASSES = ["brain", "muscle", "eye", "channel_noise", "heart", "line_noise",
           "other"]


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def load_motion(subj):
    """-> data (29, n) in uV at 250 Hz, sfreq, markers, block list."""
    import mne
    d = load(subj, PART)
    idx = [d["labels"].index(c) for c in SCALP]
    info = mne.create_info(SCALP, d["sfreq"], "eeg")
    raw = mne.io.RawArray(d["eeg"][idx] * 1e-6, info, verbose=False)
    raw.resample(TARGET_SFREQ, verbose=False)
    data = raw.get_data() * 1e6
    return data, float(raw.info["sfreq"]), d["markers"], blocks(d["markers"])


def build_calib(subj, variant, data, sfreq, blks):
    """Calibration data for ASR + ORICA warm start.

    eor_same : all eyes-open-rest blocks of the motion recording itself
               (NON-causal: uses data from later in the session).
    first120 : first 120 s of the motion recording's opening eyes-closed
               rest block (causal; same rule as the driving dataset).
    eye_rest : eyes-open-rest blocks of section 1 of the separate EYE
               recording (causal only if that recording came first).
    """
    if variant == "eor_same":
        segs = [(a + CALIB_TRIM_S, b - CALIB_TRIM_S) for nm, a, b in blks
                if nm == CALIB_BLOCK and np.isfinite(b)]
        return np.concatenate(
            [data[:, int(a * sfreq):int(b * sfreq)] for a, b in segs], axis=1)
    if variant == "first_eor":
        # EYE recording: its opening eyes-open-rest block (causal, ~56 s).
        nm, a, b = next(x for x in blks if x[0] == "eor")
        return data[:, int((a + CALIB_TRIM_S) * sfreq):
                    int((b - CALIB_TRIM_S) * sfreq)]
    if variant == "first120":
        a = next(a for nm, a, b in blks if nm == "ecr") + CALIB_TRIM_S
        return data[:, int(a * sfreq):int((a + 120.0) * sfreq)]
    if variant == "eye_rest":
        import mne
        d = load(subj, "eye")
        idx = [d["labels"].index(c) for c in SCALP]
        raw = mne.io.RawArray(d["eeg"][idx] * 1e-6,
                              mne.create_info(SCALP, d["sfreq"], "eeg"),
                              verbose=False)
        raw.resample(TARGET_SFREQ, verbose=False)
        x = raw.get_data() * 1e6
        eb = blocks(d["markers"])
        sec2 = next((a for nm, a, b in eb if nm == "section2"), np.inf)
        segs = [(a + CALIB_TRIM_S, b - CALIB_TRIM_S) for nm, a, b in eb
                if nm == "eor" and np.isfinite(b) and a < sec2]
        return np.concatenate(
            [x[:, int(a * sfreq):int(b * sfreq)] for a, b in segs], axis=1)
    raise ValueError(variant)


def make_classifier(info, cfg):
    """ICLabelClassifier that also keeps the ORICA mixing matrix of every
    window that produced a snapshot (the two lists stay index-aligned)."""
    from pyorica.pipeline.classify import ICLabelClassifier

    class TopoRecordingClassifier(ICLabelClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mixings = []

        def __call__(self, data, sources, unmixing, mixing, sfreq):
            n_before = len(self.snapshots)
            mask = super().__call__(data, sources, unmixing, mixing, sfreq)
            if len(self.snapshots) > n_before:
                self.mixings.append(
                    np.asarray(mixing, dtype=np.float64).copy())
            return mask

    return TopoRecordingClassifier(
        info, threshold=cfg.icalabel_threshold,
        apply_car_bandpass=cfg.icalabel_apply_car_bandpass,
        record_snapshots=True)


def process_subject(subj, cfg):
    done = OUT / f"{subj}_done.json"
    if done.exists():
        print(f"[{subj}] done marker exists - skip", flush=True)
        return
    from mne_icalabel import label_components
    from run_validation import _make_mne_info
    from pyorica.pipeline.pipeline import EEGPipeline
    from pyorica.pipeline.classify import _canonical_icalabel_label
    from pyorica.eval.runner import run
    from pyorica.eval.ica_analysis import _fit_ica, _make_raw, _project_sources

    t_subj = time.monotonic()
    data, sfreq, markers, blks = load_motion(subj)
    calib = build_calib(subj, VARIANT, data, sfreq, blks)
    print(f"[{subj}] {data.shape[0]} ch, {data.shape[1] / sfreq / 60:.1f} min,"
          f" calibration '{VARIANT}' = {calib.shape[1] / sfreq:.0f} s",
          flush=True)

    info = _make_mne_info(SCALP, sfreq)
    clf = make_classifier(info, cfg)
    pipe = EEGPipeline(n_channels=data.shape[0], sfreq=sfreq,
                       classifier=clf, verbose=True, config=cfg)
    chunk = int(getattr(cfg, "chunk_size", 1000) or 1000)
    t0 = time.monotonic()
    result = run(pipe, data, chunk_size=chunk, calibration_data=calib,
                 verbose=True, label=None)
    t_fwd = time.monotonic() - t0
    print(f"[{subj}] forward pipeline ({t_fwd / 60:.1f} min)", flush=True)
    del data, calib
    gc.collect()

    np.savez(OUT / f"{subj}_stages.npz",
             iir=result.iir.astype(np.float32),
             asr=result.asr.astype(np.float32),
             output=result.output.astype(np.float32),
             sfreq=sfreq, ch_names=np.array(SCALP, dtype="U8"),
             marker_t=np.array([m[0] for m in markers]),
             marker_label=np.array([m[1] for m in markers], dtype="U16"))
    np.savez_compressed(
        OUT / f"{subj}_windows.npz",
        labels=np.array([s[1] for s in clf.snapshots], dtype="U32"),
        probs=np.stack([s[2] for s in clf.snapshots]),
        mask=np.stack([s[3] for s in clf.snapshots]),
        mixings=np.stack(clf.mixings), chunk_size=chunk, sfreq=sfreq)

    t0 = time.monotonic()
    raw_iir = _make_raw(result.iir, SCALP, sfreq,
                        l_freq=cfg.iir_l_freq, h_freq=cfg.iir_h_freq)
    eval_ica = _fit_ica(raw_iir, random_state=42, max_iter=500)
    lab_out = label_components(raw_iir, eval_ica, method="iclabel")
    eval_labels = [_canonical_icalabel_label(lb) for lb in lab_out["labels"]]
    eval_probs = np.asarray(lab_out["y_pred_proba"], dtype=float)
    eval_ica.save(str(OUT / f"{subj}_eval_ica.fif"), overwrite=True)
    (OUT / f"{subj}_eval_ica_labels.json").write_text(json.dumps({
        "labels": eval_labels, "probs": eval_probs.tolist()}))
    t_ica = time.monotonic() - t0
    print(f"[{subj}] eval ICA fit+label ({t_ica / 60:.1f} min)", flush=True)
    del raw_iir
    gc.collect()

    i0 = int(CONV_MIN * 60.0 * sfreq)
    src_iir = _project_sources(eval_ica, result.iir)[:, i0:]
    src_asr = _project_sources(eval_ica, result.asr)[:, i0:]
    src_onl = _project_sources(eval_ica, result.output)[:, i0:]
    for i, (lb, pr) in enumerate(zip(eval_labels, eval_probs)):
        append_row(CSV, {
            "subject": subj, "ic": i, "label": lb,
            "prob": round(float(pr), 3),
            "ms_iir": float(np.mean(src_iir[i] ** 2)),
            "ms_asr": float(np.mean(src_asr[i] ** 2)),
            "ms_online": float(np.mean(src_onl[i] ** 2)),
        })
    del src_iir, src_asr, src_onl, result, eval_ica
    gc.collect()
    done.write_text(json.dumps({
        "fwd_min": round(t_fwd / 60, 2), "ica_min": round(t_ica / 60, 2),
        "minutes": round((time.monotonic() - t_subj) / 60, 1)}))
    print(f"[{subj}] ALL DONE in {(time.monotonic() - t_subj) / 60:.1f} min",
          flush=True)


def describe():
    """Counts and retention distributions only — no hypothesis tests."""
    rows = list(csv.DictReader(open(CSV, newline="")))
    subs = sorted({r["subject"] for r in rows})
    print(f"\nsubjects done: {len(subs)}  ({', '.join(subs)})")
    print("\nIC counts per subject (evaluation decomposition):")
    print("  subj  " + "  ".join(f"{c[:7]:>7}" for c in CLASSES))
    for s in subs:
        cnt = [sum(1 for r in rows if r["subject"] == s and r["label"] == c)
               for c in CLASSES]
        print(f"  {s}   " + "  ".join(f"{n:7d}" for n in cnt))
    print("\nretention % vs IIR (IC level, pooled) — median [IQR]:")
    for c in CLASSES:
        rr = [r for r in rows if r["label"] == c and float(r["ms_iir"]) > 0]
        if not rr:
            continue
        a = np.array([100 * float(r["ms_asr"]) / float(r["ms_iir"]) for r in rr])
        o = np.array([100 * float(r["ms_online"]) / float(r["ms_iir"]) for r in rr])
        ns = len({r["subject"] for r in rr})
        q = lambda v: (f"{np.median(v):6.1f} [{np.percentile(v, 25):5.1f}, "
                       f"{np.percentile(v, 75):5.1f}]")
        print(f"  {c:14} n_ic={len(rr):3d} n_subj={ns:2d}  ASR {q(a)}   "
              f"full {q(o)}")


def main():
    global VARIANT, OUT, CSV, PART
    if "--calib" in sys.argv:
        VARIANT = sys.argv[sys.argv.index("--calib") + 1]
    if "--part" in sys.argv:          # --part eye : the eye-artifact recording
        PART = sys.argv[sys.argv.index("--part") + 1]
    if PART == "eye":
        VARIANT = "first_eor"
        OUT = RESULTS / "yuan2019_eye_first_eor"
    else:
        OUT = RESULTS / f"yuan2019_primary_{VARIANT}"
    CSV = OUT / "yuan_retention_ics.csv"
    print(f"calibration variant: {VARIANT} -> {OUT}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    subjects = SUBJECTS
    if "--only" in sys.argv:
        subjects = [sys.argv[sys.argv.index("--only") + 1]]
    if "--describe" not in sys.argv:
        from pyorica.config import PipelineConfig
        cfg = PipelineConfig.from_yaml(REF_CFG)
        for subj in subjects:
            try:
                process_subject(subj, cfg)
            except Exception:
                print(f"[{subj}] FAILED:\n{traceback.format_exc()}", flush=True)
            gc.collect()
    if CSV.exists():
        describe()


if __name__ == "__main__":
    main()
