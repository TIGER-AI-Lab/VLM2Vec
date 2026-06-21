#!/usr/bin/env python3
"""Dump paper (MMEB-V3 Table 8 audio Hit@1, Table 9 text NDCG@5) vs our reproduced
scores to per-modality CSVs in mmeb2gembench/docs/v3/.

Audio is PARTIAL (only classification datasets run in this env; see report). Paper
numbers are parsed from the pdftotext dump at /tmp/mmebv3.txt (last model column =
Omni-Embed-Nemotron-3B). Dataset names are matched by normalization (lower, strip /_- and space).
"""
import csv, glob, json, os, re

TXT = "/tmp/mmebv3.txt"
SCORES = "/rmeng_data/projects/embed/VLM2Vec-olm2vec/exps/nemotron"
OUT = "/rmeng_data/projects/embed/VLM2Vec/mmeb2gembench/docs/v3"
MODELS = ["Qwen3-VL-Embedding-2B","Qwen3-VL-Embedding-8B","VLM2Vec-Qwen2VL-7B",
          "VLM2Vec-V2.0-2B","GME-7B","WAVE-7B","Omni-Embed-Nemotron-3B"]


def norm(s):
    return re.sub(r"[/_\- ]", "", s).lower()


def parse_paper_global():
    """Parse all per-dataset paper rows in the pdf text -> {norm_name: (name, [7 floats])}."""
    rows = {}
    for ln in open(TXT).read().splitlines():
        toks = ln.split()
        if len(toks) < 8:
            continue
        try:
            vals = [float(x) for x in toks[-7:]]
        except ValueError:
            continue
        name = " ".join(toks[:-7])
        if name.startswith(("Avg", "T-", "A-", "I-", "V-")) or "(" in name:
            continue
        rows.setdefault(norm(name), (name, vals))
    return rows


PAPER = parse_paper_global()


def dump(modality, metric, paper_avg, note=""):
    subdir = f"{OUT}/paper_vs_repro/nemotron"
    os.makedirs(subdir, exist_ok=True)
    path = f"{subdir}/nemotron_{modality}_{metric.replace('_','')}_paper_vs_repro.csv"
    diffs = []
    rows_out = []
    for f in sorted(glob.glob(f"{SCORES}/{modality}/**/*_score.json", recursive=True)):
        key = os.path.relpath(f, f"{SCORES}/{modality}")[:-len("_score.json")]
        # strip a leading modality segment (e.g. text/BRIGHT/aops -> BRIGHT/aops)
        key_clean = key.split("/", 1)[1] if key.startswith(modality + "/") else key
        ours = round(json.load(open(f)).get(metric, 0) * 100, 1)
        nk = norm(key_clean)
        if nk in PAPER:
            pname, pv = PAPER[nk]
            d = round(ours - pv[-1], 1); diffs.append(d)
            rows_out.append([pname] + pv + [ours, d])
        else:
            rows_out.append([key_clean] + [""]*7 + [ours, "(no paper match)"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        if note:
            w.writerow([note])
        w.writerow(["dataset"] + [f"{m} (paper)" for m in MODELS]
                   + [f"Omni-Embed-Nemotron-3B (reproduced, {metric})", "diff"])
        w.writerows(rows_out)
        if diffs:
            ours_avg = round(sum(r[8] for r in rows_out if isinstance(r[9], float)) / len(diffs), 1)
            w.writerow([f"AVG (repro over {len(diffs)} datasets)"] + [""]*6
                       + [paper_avg, ours_avg, round(ours_avg - paper_avg, 1)])
    mae = round(sum(abs(d) for d in diffs)/len(diffs), 2) if diffs else None
    print(f"{modality}: wrote {path}  ({len(diffs)} matched, MAE {mae})")


dump("text", "ndcg_linear@5", 38.6)

# Audio paper Table 8 has only 2 model cols (WAVE, Omni-Embed-Nemotron-3B); hardcode
# nemotron Hit@1. Only classification datasets run in this env (retrieval/grounding blocked).
AUDIO_PAPER = {  # Omni-Embed-Nemotron-3B Hit@1 x100
    "NSynth": 28.2, "UrbanSound8K": 43.9, "ESC-50": 75.7, "SpeechCommands": 45.0, "CREMA-D": 27.2,
    "Clotho": 11.5, "SoundDescs": 14.5, "AVE": 12.7, "SpeechCOCO": 26.4, "TUTSound": 42.1, "TUTSound hard": 4.3,
}
def dump_audio():
    subdir = f"{OUT}/paper_vs_repro/nemotron"
    os.makedirs(subdir, exist_ok=True)
    path = f"{subdir}/nemotron_audio_hit@1_paper_vs_repro.csv"
    diffs = []
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# PARTIAL: only audio CLASSIFICATION datasets ran in this env; retrieval/grounding "
                    "blocked by a broken audio stack (no torchcodec/ffmpeg). CLS numbers also diverge "
                    "from paper because soundfile workarounds don't faithfully match the paper's decoding. "
                    "See reproduction_report.md."])
        w.writerow(["dataset", "Omni-Embed-Nemotron-3B (paper, hit@1)",
                    "Omni-Embed-Nemotron-3B (reproduced, hit@1)", "diff", "status"])
        for ds, paper in AUDIO_PAPER.items():
            f = f"{SCORES}/audio/{ds}_score.json"
            if os.path.exists(f):
                ours = round(json.load(open(f))["hit@1"] * 100, 1)
                d = round(ours - paper, 1); diffs.append(d)
                w.writerow([ds, paper, ours, d, "CLS"])
            else:
                w.writerow([ds, paper, "", "(not run)", "RET/TG blocked"])
        if diffs:
            cls_paper = round(sum(AUDIO_PAPER[d] for d in ["NSynth","UrbanSound8K","ESC-50","SpeechCommands","CREMA-D"])/5, 1)
            ours_avg = round(sum(round(json.load(open(f"{SCORES}/audio/{d}_score.json"))["hit@1"]*100,1)
                                 for d in AUDIO_PAPER if os.path.exists(f"{SCORES}/audio/{d}_score.json"))/len(diffs), 1)
            w.writerow([f"AVG-CLS (repro over {len(diffs)}/5)", cls_paper, ours_avg, round(ours_avg-cls_paper,1), ""])
    print(f"audio: wrote {path} ({len(diffs)}/5 CLS matched)")
dump_audio()
