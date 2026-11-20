# -*- coding: utf-8 -*-
"""Yuan 2019 MOTION recording: evaluation ICA fit separately per condition.

Feedback (11/13): each condition is roughly stationary on its own, so the
evaluation decomposition should be fit per condition instead of once over the
whole 21-min session (where the large motion artifacts leak into the
components that get the 'brain' label).

No pipeline rerun: uses the saved stage streams of the causal-calibration run
(<results>/yuan2019_primary_first120/{subj}_stages.npz, written by
yuan2019_primary.py). For every subject x
condition (ecr, odd, eor, shh, shv, str, wlk): concatenate that condition's
IIR-stage blocks, fit extended Infomax (seed 42, max_iter 500), ICLabel, then
project the IIR / ASR / full-pipeline streams through that unmixing and take
the per-block mean square of every component.

Retention of a component = ms_asr / ms_iir (ASR only) or ms_online / ms_iir
(full pipeline), computed per block; repeated blocks of a condition are
combined by their median, then components of a class by their median within
subject (see subj_table).

Output: <results>/yuan2019_primary_first120/yuan_percond_ics.csv
    columns: subject, cond, ic, label, prob, rep, n, n_fit_s,
             ms_iir, ms_asr, ms_online
Usage: python benchmarks/analysis/yuan2019/yuan2019_percond_ica.py [--describe]
"""
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(Path(__file__).parent))
from yuan2019_qc import RESULTS, SCALP, SUBJECTS, blocks  # noqa: E402

REF_CFG = REPO / "benchmarks" / "config" / "reference.yaml"
OUT = RESULTS / "yuan2019_primary_first120"
CSV = OUT / "yuan_percond_ics.csv"
CONDS = ["ecr", "odd", "eor", "shh", "shv", "str", "wlk"]
FIELDS = ["subject", "cond", "ic", "label", "prob", "rep", "n", "n_fit_s",
          "ms_iir", "ms_asr", "ms_online"]


def main():
    from mne_icalabel import label_components
    from pyorica.config import PipelineConfig
    from pyorica.pipeline.classify import _canonical_icalabel_label
    from pyorica.eval.ica_analysis import _fit_ica, _make_raw, _project_sources
    cfg = PipelineConfig.from_yaml(REF_CFG)
    done = set()
    if CSV.exists():
        d = pd.read_csv(CSV)
        done = set(zip(d.subject, d.cond))
    else:
        with open(CSV, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()
    for subj in SUBJECTS:
        st = np.load(OUT / f"{subj}_stages.npz")
        sf = float(st["sfreq"])
        X = {k: st[k].astype(float) for k in ("iir", "asr", "output")}
        blks = blocks(list(zip(st["marker_t"].tolist(),
                               st["marker_label"].tolist())))
        for cond in CONDS:
            if (subj, cond) in done:
                continue
            if subj == "s08" and cond == "wlk":   # signal collapse
                continue
            sls = [slice(int(a * sf), int(b * sf)) for nm, a, b in blks
                   if nm == cond and np.isfinite(b)]
            if not sls:
                continue
            t0 = time.monotonic()
            seg = np.concatenate([X["iir"][:, s] for s in sls], axis=1)
            raw = _make_raw(seg, SCALP, sf, l_freq=cfg.iir_l_freq,
                            h_freq=cfg.iir_h_freq)
            ica = _fit_ica(raw, random_state=42, max_iter=500)
            lab = label_components(raw, ica, method="iclabel")
            labels = [_canonical_icalabel_label(x) for x in lab["labels"]]
            probs = np.asarray(lab["y_pred_proba"], dtype=float)
            rows = []
            for rep, s in enumerate(sls, 1):
                src = {k: _project_sources(ica, X[k][:, s]) for k in X}
                for i, (lb, pr) in enumerate(zip(labels, probs)):
                    rows.append({
                        "subject": subj, "cond": cond, "ic": i, "label": lb,
                        "prob": round(float(pr), 3), "rep": rep,
                        "n": s.stop - s.start,
                        "n_fit_s": round(seg.shape[1] / sf, 1),
                        "ms_iir": float(np.mean(src["iir"][i] ** 2)),
                        "ms_asr": float(np.mean(src["asr"][i] ** 2)),
                        "ms_online": float(np.mean(src["output"][i] ** 2))})
            with open(CSV, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writerows(rows)
            cnt = pd.Series(labels).value_counts().to_dict()
            print(f"[{subj}] {cond}: fit on {seg.shape[1] / sf:.0f} s, "
                  f"{len(labels)} ICs {cnt} "
                  f"({time.monotonic() - t0:.0f} s)", flush=True)
            del raw, ica, seg
            gc.collect()
    describe()


def subj_table(path):
    d = pd.read_csv(path)
    d = d[~((d.subject == "s08") & (d.cond == "wlk"))]
    d["r_asr"] = 100 * d.ms_asr / d.ms_iir
    d["r_full"] = 100 * d.ms_online / d.ms_iir
    g = d.groupby(["subject", "ic", "label", "cond"])[["r_asr", "r_full"]] \
         .median().reset_index()
    return g.groupby(["subject", "label", "cond"])[["r_asr", "r_full"]] \
            .median().reset_index()


def describe():
    d = pd.read_csv(CSV)
    ics = d.drop_duplicates(["subject", "cond", "ic"])
    print("\nfit length per condition (s), median over subjects:")
    print(ics.groupby("cond").n_fit_s.median().reindex(CONDS).to_string())
    print("\nIC counts per condition: median over subjects [min-max]")
    cnt = ics.groupby(["cond", "subject", "label"]).size().unstack(fill_value=0)
    for lab in ("brain", "eye", "muscle", "channel_noise", "other"):
        if lab not in cnt:
            continue
        s = cnt[lab].groupby("cond")
        print(f"  {lab:14}" + "  ".join(
            f"{c}:{s.median()[c]:.0f}[{s.min()[c]}-{s.max()[c]}]" for c in CONDS))
    new = subj_table(CSV)
    print("\nretention % vs IIR, subject-level median (n subj), "
          "per-condition fit")
    for lab in ("brain", "eye"):
        for c in CONDS:
            x = new[(new.label == lab) & (new.cond == c)]
            dd = x.r_asr - x.r_full
            print(f"  {lab:5} {c}: ASR {x.r_asr.median():5.1f} full "
                  f"{x.r_full.median():5.1f} extra {dd.median():5.2f} "
                  f"(n={len(x)}, full<ASR {(dd > 0).sum()})")


if __name__ == "__main__":
    describe() if "--describe" in sys.argv else main()
