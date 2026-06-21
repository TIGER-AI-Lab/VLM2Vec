#!/usr/bin/env python3
"""Aggregate Omni-Embed-Nemotron-3B eval scores for tool + memory.

Walks exps/nemotron/{tool,memory}/**/*_score.json, prints per-task metrics and
category / overall averages. Reports several candidate headline metrics since the
exact metric the MMEB-V3 paper headlines for agent tasks may differ.
"""
import json
import os
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "exps", "nemotron")
ROOT = os.path.abspath(ROOT)
METRICS = ["hit@1", "ndcg_linear@5", "ndcg_linear@10", "recall@5", "recall@10", "map@10"]


def find_scores(modality_dir):
    out = {}
    for dirpath, _, files in os.walk(modality_dir):
        for fn in files:
            if fn.endswith("_score.json"):
                full = os.path.join(dirpath, fn)
                key = os.path.relpath(full, modality_dir)[: -len("_score.json")]
                with open(full) as f:
                    out[key] = json.load(f)
    return out


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt_row(name, d, width=42):
    cells = "  ".join(f"{d.get(m, float('nan')):.4f}" for m in METRICS)
    return f"{name:<{width}} {cells}"


def report(modality):
    mdir = os.path.join(ROOT, modality)
    if not os.path.isdir(mdir):
        print(f"[{modality}] no dir at {mdir}")
        return
    scores = find_scores(mdir)
    print(f"\n{'='*100}\n{modality.upper()}  ({len(scores)} tasks)   metrics: {METRICS}\n{'='*100}")
    print(fmt_row("task", {m: float('nan') for m in METRICS}).replace("nan", "    "))
    # group by top-level category (web/code/customized for tool; flat for memory)
    by_cat = defaultdict(list)
    for k in sorted(scores):
        cat = k.split("/")[0] if "/" in k else "_"
        by_cat[cat].append(k)
        print(fmt_row("  " + k, scores[k]))
    print("-" * 100)
    for cat in sorted(by_cat):
        avg = {m: mean([scores[k].get(m) for k in by_cat[cat]]) for m in METRICS}
        print(fmt_row(f"[avg/{cat}] (n={len(by_cat[cat])})", avg))
    overall = {m: mean([scores[k].get(m) for k in scores]) for m in METRICS}
    print("=" * 100)
    print(fmt_row(f"[OVERALL {modality}] (n={len(scores)})", overall))


if __name__ == "__main__":
    print(f"Reading from: {ROOT}")
    print("Header metrics order:", METRICS)
    for mod in ["tool", "memory"]:
        report(mod)
