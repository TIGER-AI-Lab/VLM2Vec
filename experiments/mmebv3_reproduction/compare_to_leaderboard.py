#!/usr/bin/env python
"""Compare local MMEB-V2 score dirs against a leaderboard submission file.

The leaderboard stores one JSON per model (scores/<model>.json in the TIGER-Lab/MMEB-Leaderboard
space) with metrics[suite][dataset] -> {hit@1, ndcg_linear@5, ..., num_data}. Image and video are
averaged on hit@1, visdoc on ndcg_linear@5, plain means over the datasets present in the file.

Usage:
  python experiments/mmebv3_reproduction/compare_to_leaderboard.py \
      --lb docs/mmebv2_reproduction/leaderboard/Qwen3-VL-Embedding-8B.json \
      --image <dir> --video <dir> --visdoc <dir> [--top 8]
"""
import argparse
import glob
import json
import os

METRIC = {"image": "hit@1", "video": "hit@1", "visdoc": "ndcg_linear@5"}


def load_scores(score_dir):
    out = {}
    for f in glob.glob(os.path.join(score_dir, "*_score.json")):
        name = os.path.basename(f)[: -len("_score.json")]
        try:
            out[name] = json.load(open(f))
        except (OSError, ValueError):
            continue
    return out


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def compare(suite, lb_suite, ours, top):
    metric = METRIC[suite]
    lb = {k: float(v[metric]) * 100 for k, v in lb_suite.items() if metric in v}
    us = {k: float(v[metric]) * 100 for k, v in ours.items() if metric in v}
    common = sorted(set(lb) & set(us))
    print(f"\n=== {suite.upper()} ({metric}) ===")
    print(f"  leaderboard: n={len(lb)} mean={mean(lb.values()):.2f}")
    print(f"  ours:        n={len(us)} mean={mean(us.values()):.2f}")
    if not common:
        print("  no datasets in common")
        return
    lb_c, us_c = mean([lb[k] for k in common]), mean([us[k] for k in common])
    print(f"  like-for-like on n={len(common)}: LB {lb_c:.2f}  ours {us_c:.2f}  gap {us_c - lb_c:+.2f}")
    only_lb, only_us = sorted(set(lb) - set(us)), sorted(set(us) - set(lb))
    if only_lb:
        print(f"  only on leaderboard: {only_lb}")
    if only_us:
        print(f"  only in ours:        {only_us}")
    size_mismatch = []
    for k in common:
        n_lb, n_us = lb_suite[k].get("num_data"), ours[k].get("num_data", ours[k].get("num_pred"))
        if n_lb is not None and n_us is not None and int(n_lb) != int(n_us):
            size_mismatch.append((k, int(n_lb), int(n_us)))
    if size_mismatch:
        print(f"  num_data mismatches (LB, ours): {size_mismatch}")
    ranked = sorted(common, key=lambda k: us[k] - lb[k])
    print(f"  largest gaps (ours - LB):")
    for k in ranked[:top]:
        print(f"    {k:44s} {us[k]:6.1f}  vs {lb[k]:6.1f}  {us[k] - lb[k]:+6.1f}")
    if top and len(ranked) > top:
        print(f"  largest gains:")
        for k in ranked[-3:][::-1]:
            print(f"    {k:44s} {us[k]:6.1f}  vs {lb[k]:6.1f}  {us[k] - lb[k]:+6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb", required=True, help="leaderboard scores/<model>.json")
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--visdoc")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    lb = json.load(open(args.lb))["metrics"]
    for suite in ("image", "video", "visdoc"):
        d = getattr(args, suite)
        if d and suite in lb:
            compare(suite, lb[suite], load_scores(d), args.top)


if __name__ == "__main__":
    main()
