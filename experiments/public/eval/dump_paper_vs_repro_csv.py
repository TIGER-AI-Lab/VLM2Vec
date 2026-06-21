#!/usr/bin/env python3
"""Dump paper (MMEB-V3 Table 10) vs our reproduced Hit@1 to per-modality CSVs.

Parses the agent detailed table (Table 10) text from the paper PDF dump and joins
our reproduced Omni-Embed-Nemotron-3B Hit@1 scores. One CSV per reproduced modality
(tool, memory) written to mmeb2gembench/docs/v3/.
"""
import csv, json, os

TXT = "/tmp/mmebv3.txt"
SCORES_ROOT = "/rmeng_data/projects/embed/VLM2Vec-olm2vec/exps/nemotron"
OUT_DIR = "/rmeng_data/projects/embed/VLM2Vec/mmeb2gembench/docs/v3"

MODELS = [
    "Qwen3-VL-Embedding-2B", "Qwen3-VL-Embedding-8B", "VLM2Vec-Qwen2VL-7B",
    "VLM2Vec-V2.0-2B", "GME-7B", "WAVE-7B", "Omni-Embed-Nemotron-3B",
]
NEMO = "Omni-Embed-Nemotron-3B"  # last paper column == the model we reproduced


def parse_table10():
    """Return {prefix: {dataset: [7 floats]}} for tool / memory rows."""
    rows = {"tool": {}, "memory": {}}
    mem_names = {"KnowMeBench", "REALTALK", "PeerQA", "DeepPlanning"}
    with open(TXT) as f:
        lines = f.readlines()
    # Table 10 block is after the agent header; scan whole file, rows are unambiguous by prefix.
    for ln in lines:
        toks = ln.split()
        if len(toks) < 8:
            continue
        tail = toks[-7:]
        try:
            vals = [float(x) for x in tail]
        except ValueError:
            continue
        name_tokens = toks[: len(toks) - 7]
        name = " ".join(name_tokens)
        if name.startswith("Tool-"):
            rows["tool"][name[len("Tool-"):]] = vals
        elif name in mem_names:
            rows["memory"][name] = vals
    return rows


def load_repro(modality):
    out = {}
    mdir = os.path.join(SCORES_ROOT, modality)
    for dp, _, fs in os.walk(mdir):
        for fn in fs:
            if fn.endswith("_score.json"):
                with open(os.path.join(dp, fn)) as f:
                    out[fn[: -len("_score.json")]] = round(json.load(f)["hit@1"] * 100, 1)
    return out


def write_csv(modality, paper_rows, repro):
    subdir = os.path.join(OUT_DIR, "paper_vs_repro", "nemotron")
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"nemotron_{modality}_hit@1_paper_vs_repro.csv")
    header = (["dataset"] + [f"{m} (paper)" for m in MODELS]
              + [f"{NEMO} (reproduced)", "diff (repro - paper)"])
    keys = sorted(paper_rows)
    diffs = []
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in keys:
            paper = paper_rows[k]
            r = repro.get(k, "")
            d = round(r - paper[-1], 1) if r != "" else ""
            if d != "":
                diffs.append(d)
            w.writerow([k] + paper + [r, d])
        # averages
        n = len(keys)
        avg_paper = [round(sum(paper_rows[k][i] for k in keys) / n, 1) for i in range(7)]
        avg_repro = round(sum(repro[k] for k in keys if k in repro) / sum(1 for k in keys if k in repro), 1)
        avg_diff = round(avg_repro - avg_paper[-1], 1)
        w.writerow(["AVG (n=%d)" % n] + avg_paper + [avg_repro, avg_diff])
    mae = round(sum(abs(x) for x in diffs) / len(diffs), 2) if diffs else None
    print(f"wrote {path}  ({n} datasets, AVG repro {avg_repro} vs paper {avg_paper[-1]}, MAE {mae})")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    table = parse_table10()
    for mod in ["tool", "memory"]:
        write_csv(mod, table[mod], load_repro(mod))
