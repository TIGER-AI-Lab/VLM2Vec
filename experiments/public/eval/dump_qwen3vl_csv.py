#!/usr/bin/env python3
"""Dump paper (MMEB-V3) vs reproduced Qwen3-VL-Embedding (2B + 8B) scores.

One CSV per (tag, modality) into mmeb2gembench/docs/v3/:
    qwen3vl-{2B,8B}_<modality>_<metric>_paper_vs_repro.csv

Metric: image/video/tool/memory/gui = hit@1 ; visdoc/text = ndcg_linear@5.
Paper per-dataset rows parsed from /tmp/mmebv3.txt. The 7 paper model columns are
(in order): Qwen3VL-2B, Qwen3VL-8B, VLM2Vec-Qwen2VL-7B, VLM2Vec-V2.0-2B, GME-7B,
WAVE-7B, Omni-Embed-Nemotron-3B. So for tag 2B the comparison column is idx 0; for 8B idx 1.
eval_time_s parsed from the run logs ([Timing] dataset=..., total=HH:MM:SS).
"""
import csv, glob, json, os, re

TXT = "/tmp/mmebv3.txt"
SCORES = "/rmeng_data/exps/vlm2vec/qwen3vl"
LOGS = "/rmeng_data/exps/vlm2vec/qwen3vl/logs"
OUT = "/rmeng_data/projects/embed/VLM2Vec/mmeb2gembench/docs/v3"
MODELS = ["Qwen3-VL-Embedding-2B", "Qwen3-VL-Embedding-8B", "VLM2Vec-Qwen2VL-7B",
          "VLM2Vec-V2.0-2B", "GME-7B", "WAVE-7B", "Omni-Embed-Nemotron-3B"]
TAG_COL = {"2B": 0, "8B": 1}
METRIC = {"image": "hit@1", "video": "hit@1", "tool": "hit@1", "memory": "hit@1",
          "gui": "hit@1", "visdoc": "ndcg_linear@5", "text": "ndcg_linear@5"}
# paper category averages (from driver prompt / paper tables)
PAPER_CAT = {  # modality: (2B, 8B)
    "image": (69.5, 72.1), "video": (55.9, 58.6), "visdoc": (70.6, 70.9),
    "text": (39.2, 42.5), "tool": (42.6, 41.3), "memory": (28.4, 22.8), "gui": (30.4, 33.5)}
PREFIXES = ("Tool-", "T-", "A-", "I-", "V-")


def norm(s):
    return re.sub(r"[/_\- ]", "", s).lower()


def parse_paper():
    """Return dict mapping normalized dataset name -> (display_name, [7 floats]).
    Indexed both by the full name and by the name with a leading category prefix stripped."""
    idx = {}
    for ln in open(TXT).read().splitlines():
        toks = ln.split()
        if len(toks) < 8:
            continue
        try:
            vals = [float(x) for x in toks[-7:]]
        except ValueError:
            continue
        name = " ".join(toks[:-7])
        if "(" in name or name.lower().startswith("avg") or name.startswith("#"):
            continue
        idx.setdefault(norm(name), (name, vals))
        for p in PREFIXES:
            if name.startswith(p):
                idx.setdefault(norm(name[len(p):]), (name, vals))
    return idx


PAPER = parse_paper()


def parse_times(tag, modality):
    """leaf dataset name -> eval seconds, from the newest run log."""
    logs = sorted(glob.glob(f"{LOGS}/q3vl{tag}-{modality}-*.log"), key=os.path.getmtime)
    times = {}
    for lg in logs:  # later logs (resumes) overwrite earlier
        for ln in open(lg, errors="ignore"):
            m = re.search(r"\[Timing\] dataset=(.+?), status=(\w+), total=(\d+):(\d+):(\d+)", ln)
            if m:
                name, status, h, mm, s = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
                if status == "success":
                    times[name] = h * 3600 + mm * 60 + s
    return times


def leaf(key):
    return key.rsplit("/", 1)[-1]


def strip_first(key):
    return key.split("/", 1)[1] if "/" in key else key


def match_paper(relpath):
    """Try several normalized forms of a reproduced dataset path against the paper index."""
    for cand in (strip_first(relpath), leaf(relpath), relpath):
        if norm(cand) in PAPER:
            return PAPER[norm(cand)]
    return None


def collect(tag, modality):
    """full_relpath -> (metric_val_x100, eval_s)  (relpath == log Timing dataset= key)"""
    metric = METRIC[modality]
    times = parse_times(tag, modality)
    out = {}
    for f in sorted(glob.glob(f"{SCORES}/{tag}/{modality}/**/*_score.json", recursive=True)):
        k = os.path.relpath(f, f"{SCORES}/{tag}/{modality}")[: -len("_score.json")]
        v = round(json.load(open(f)).get(metric, 0) * 100, 1)
        out[k] = (v, times.get(k, ""))
    return out


def write_csv(tag, modality):
    metric = METRIC[modality]
    col = TAG_COL[tag]
    repro = collect(tag, modality)
    subdir = f"{OUT}/paper_vs_repro/qwen3vl-{tag}"
    os.makedirs(subdir, exist_ok=True)
    path = f"{subdir}/qwen3vl-{tag}_{modality}_{metric.replace('_', '')}_paper_vs_repro.csv"
    header = (["dataset"] + [f"{m} (paper)" for m in MODELS]
              + [f"Qwen3-VL-Embedding-{tag} (reproduced, {metric})",
                 f"diff (repro - paper {tag})", "eval_time_s"])
    diffs, vals, tsum = [], [], 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for k in sorted(repro):
            v, t = repro[k]
            vals.append(v)
            if isinstance(t, int):
                tsum += t
            hit = match_paper(k)
            if hit:
                pname, pv = hit
                d = round(v - pv[col], 1)
                diffs.append(d)
                w.writerow([pname] + pv + [v, d, t])
            else:
                w.writerow([k] + [""] * 7 + [v, "(no paper match)", t])
        n = len(vals)
        avg = round(sum(vals) / n, 1) if n else ""
        paper_avg = PAPER_CAT[modality][col]
        w.writerow([f"AVG (n={n})"] + [""] * 7
                   + [avg, round(avg - paper_avg, 1), tsum])
        w.writerow([f"PAPER category avg ({MODELS[col]})"] + [""] * 7 + [paper_avg, "", ""])
    mae = round(sum(abs(d) for d in diffs) / len(diffs), 2) if diffs else None
    print(f"{tag}/{modality}: wrote {os.path.basename(path)}  n={n} matched={len(diffs)} "
          f"AVG repro={avg} vs paper={paper_avg} (diff {round(avg-paper_avg,1)}) MAE={mae} time={tsum}s")
    return {"tag": tag, "modality": modality, "metric": metric, "n": n,
            "repro": avg, "paper": paper_avg, "diff": round(avg - paper_avg, 1),
            "mae": mae, "time_s": tsum}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for tag in ["2B", "8B"]:
        for mod in ["image", "video", "visdoc", "text", "tool", "memory", "gui"]:
            summary.append(write_csv(tag, mod))
    print("\n===== CATEGORY SUMMARY (reproduced / paper / diff) =====")
    for tag in ["2B", "8B"]:
        print(f"\nQwen3-VL-Embedding-{tag}:")
        print(f"  {'modality':<10}{'metric':<16}{'repro':>7}{'paper':>7}{'diff':>7}{'MAE':>7}{'time':>9}")
        rows = [s for s in summary if s["tag"] == tag]
        for s in rows:
            print(f"  {s['modality']:<10}{s['metric']:<16}{s['repro']:>7}{s['paper']:>7}"
                  f"{s['diff']:>7}{str(s['mae']):>7}{str(s['time_s'])+'s':>9}")
        macro = round(sum(s["repro"] for s in rows) / len(rows), 1)
        pmac = round(sum(s["paper"] for s in rows) / len(rows), 1)
        print(f"  {'MACRO-AVG':<10}{'':<16}{macro:>7}{pmac:>7}{round(macro-pmac,1):>7}")
