#!/usr/bin/env python3
"""Aggregate OmniSET (MSCOCO cross-modal cmret) no-audio results vs paper Table 4.

Reads /rmeng_data/exps/vlm2vec/omni/<tag>/MSCOCO_<dir>_score.json for the 6 no-audio
directions and compares Hit@1 + MRR to paper Table 4. Writes per-model CSVs into
mmeb2gembench/docs/v3/paper_vs_repro/<model>/ and prints tables.
"""
import csv, json, os

SCORES = "/rmeng_data/exps/vlm2vec/omni"
OUT = "/rmeng_data/projects/embed/VLM2Vec/mmeb2gembench/docs/v3/paper_vs_repro"
DIRS = ["t2i", "t2v", "i2t", "i2v", "v2t", "v2i"]   # no-audio directions

# Paper Table 4 (Hit@1, MRR) for the 6 no-audio directions.
PAPER = {
    "nemotron": {  # Omni-Embed-Nemotron-3B
        "t2i": (0.0, 4.6), "t2v": (3.0, 19.1), "i2t": (0.0, 11.6),
        "i2v": (100.0, 100.0), "v2t": (0.0, 2.4), "v2i": (2.0, 15.1)},
    "q3vl8B": {   # Qwen3-VL-Embedding (paper col; Fig 3c labels it 8B)
        "t2i": (0.0, 6.66), "t2v": (0.0, 2.84), "i2t": (0.0, 6.48),
        "i2v": (100.0, 100.0), "v2t": (0.0, 4.1), "v2i": (2.0, 15.32)},
    # q3vl2B: no paper column -> reported only.
}
MODEL_DIR = {"nemotron": "nemotron", "q3vl2B": "qwen3vl-2B", "q3vl8B": "qwen3vl-8B"}


def load(tag, d):
    f = f"{SCORES}/{tag}/MSCOCO_{d}_score.json"
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return round(j.get("hit@1", 0) * 100, 1), round(j.get("mrr", 0) * 100, 1), j.get("num_data")


def dump(tag):
    sub = f"{OUT}/{MODEL_DIR[tag]}"
    os.makedirs(sub, exist_ok=True)
    path = f"{sub}/{MODEL_DIR[tag]}_omniset_hit@1_mrr_paper_vs_repro.csv"
    paper = PAPER.get(tag)
    rows = []
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["direction",
                    "Hit@1 (paper)", "Hit@1 (repro)", "Hit@1 diff",
                    "MRR (paper)", "MRR (repro)", "MRR diff", "n"])
        for d in DIRS:
            r = load(tag, d)
            if r is None:
                w.writerow([d.upper()] + ["(pending)"] * 7); continue
            h, m, n = r
            if paper:
                ph, pm = paper[d]
                w.writerow([d.upper(), ph, h, round(h - ph, 1), pm, m, round(m - pm, 1), n])
            else:
                w.writerow([d.upper(), "(no paper)", h, "", "(no paper)", m, "", n])
            rows.append((d, h, m))
    print(f"{tag}: wrote {path}")
    return rows


if __name__ == "__main__":
    for tag in ["nemotron", "q3vl2B", "q3vl8B"]:
        print(f"\n=== {tag} (OmniSET no-audio, Hit@1 / MRR@10) ===")
        paper = PAPER.get(tag)
        rows = dump(tag)
        print(f"  {'dir':<5}{'Hit@1':>8}{'paperH':>8}{'paperM':>8}")
        for d, h, m in rows:
            ph = paper[d][0] if paper else None
            pm = paper[d][1] if paper else None
            print(f"  {d.upper():<5}{h:>8}{str(ph):>8}{m:>8}{str(pm):>8}")
