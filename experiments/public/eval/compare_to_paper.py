#!/usr/bin/env python3
"""Compare reproduced Hit@1 vs MMEB-V3 paper (Table 2 + appendix) for Omni-Embed-Nemotron-3B."""
import json, os
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "exps", "nemotron"))

# Paper Omni-Embed-Nemotron-3B, Hit@1 x100 (last column of appendix tables)
PAPER = {
    # tool
    "craft-math-algebra": 85.0, "craft-tabmwp": 29.9, "craft-vqa": 64.5,
    "gorilla-huggingface": 26.6, "gorilla-pytorch": 11.6, "gorilla-tensor": 20.0,
    "toolink": 66.8, "apibank": 45.5, "apigen": 64.1, "mnms": 39.4,
    "reversechain": 50.0, "rotbench": 7.1, "t-eval-dialog": 16.0, "t-eval-step": 10.0,
    "taskbench-daily": 67.5, "toolace": 67.1, "toolbench": 13.1, "toolemu": 34.2,
    "tooleyes": 10.5, "toollens": 1.0, "ultratool": 43.8, "autotools-food": 13.6,
    "autotools-music": 6.2, "autotools-weather": 0.0, "restgpt-spotify": 25.0,
    "restgpt-tmdb": 16.7, "appbench": 87.5, "gpt4tools": 75.0, "gta": 42.9,
    "taskbench-huggingface": 52.2, "taskbench-multimedia": 77.5, "metatool": 50.0,
    "tool-be-honest": 55.1, "toolalpaca": 48.9, "toolbench-sam": 8.6,
    # memory
    "KnowMeBench": 44.3, "REALTALK": 32.4, "PeerQA": 18.4, "DeepPlanning": 34.2,
}
PAPER_AVG = {"tool": 38.1, "memory": 32.3}


def load(modality):
    out = {}
    mdir = os.path.join(ROOT, modality)
    for dp, _, fs in os.walk(mdir):
        for fn in fs:
            if fn.endswith("_score.json"):
                name = fn[:-len("_score.json")]
                with open(os.path.join(dp, fn)) as f:
                    out[name] = json.load(f)["hit@1"] * 100
    return out


for modality in ["tool", "memory"]:
    scores = load(modality)
    print(f"\n{'='*64}\n{modality.upper()}   (Hit@1 x100)\n{'='*64}")
    print(f"{'task':<26}{'ours':>8}{'paper':>8}{'diff':>8}")
    diffs = []
    for name in sorted(scores):
        ours = scores[name]
        paper = PAPER.get(name)
        if paper is None:
            print(f"{name:<26}{ours:>8.1f}{'?':>8}{'':>8}")
            continue
        d = ours - paper
        diffs.append(d)
        flag = "  <-- check" if abs(d) >= 2.0 else ""
        print(f"{name:<26}{ours:>8.1f}{paper:>8.1f}{d:>+8.1f}{flag}")
    our_avg = sum(scores.values()) / len(scores)
    print("-" * 50)
    print(f"{'AVG':<26}{our_avg:>8.1f}{PAPER_AVG[modality]:>8.1f}{our_avg - PAPER_AVG[modality]:>+8.1f}")
    if diffs:
        mae = sum(abs(x) for x in diffs) / len(diffs)
        print(f"per-task MAE: {mae:.2f}  | max |diff|: {max(abs(x) for x in diffs):.1f}")
