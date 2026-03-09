#!/usr/bin/env python3
import os
import glob
import random
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import datasets

# ========= CONFIG =========
SRC_ROOT = "/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/nsynth/data"
OUT_DIR  = "/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/nsynth-1k"

N_EVAL = 1000
SEED = 17

# 类别控制：None=保留全部；否则保留出现频率最高的前 K 类
MAX_CLASSES: Optional[int] = None  # e.g., 50 or 100, default None

# 每类至少保留多少条在 eval（防止某类完全消失；如果类太多会导致达不到 N_EVAL）
MIN_PER_CLASS_EVAL = 1


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_all_parquets(root: str) -> datasets.Dataset:
    # 兼容：nsynth 可能是多 parquet 分片，可能在 root 或子目录
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {root}")
    return datasets.load_dataset("parquet", data_files={"data": files}, split="data")


def write_parquet(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        raise RuntimeError(f"Empty rows for {out_path}")
    keys = list(rows[0].keys())
    obj = {k: [r.get(k) for r in rows] for k in keys}
    datasets.Dataset.from_dict(obj).to_parquet(out_path)


def main():
    rng = random.Random(SEED)

    out_eval = os.path.join(OUT_DIR, "eval")
    ensure_dir(out_eval)

    ds = load_all_parquets(SRC_ROOT)

    # --------- 1) 识别 query-id：NSynth 没有 id，就用 WavPath ----------
    if "WavPath" not in ds.column_names:
        raise KeyError(f"[nsynth] missing WavPath. available={ds.column_names}")

    # --------- 2) 识别 label 列：优先 instrument_family_str，其次 instrument_str ----------
    label_col_candidates = ["instrument_family_str", "instrument_str", "instrument_family", "instrument"]
    label_col = None
    for c in label_col_candidates:
        if c in ds.column_names:
            label_col = c
            break
    if label_col is None:
        raise KeyError(f"[nsynth] cannot find label column in {label_col_candidates}. available={ds.column_names}")

    # --------- 3) 统计类频次 + 可选截断 MAX_CLASSES ----------
    label_counts = defaultdict(int)
    for r in ds:
        lab = str(r[label_col])
        label_counts[lab] += 1

    labels_sorted = sorted(label_counts.keys(), key=lambda x: (-label_counts[x], x))
    if MAX_CLASSES is not None:
        labels_sorted = labels_sorted[:MAX_CLASSES]

    labels_set = set(labels_sorted)

    # --------- 4) 为每类收集 index，做一个简单“分层抽样”拿到 N_EVAL ----------
    # 先把每类至少拿 MIN_PER_CLASS_EVAL 条
    per_label_indices = defaultdict(list)
    for i, r in enumerate(ds):
        lab = str(r[label_col])
        if lab in labels_set:
            per_label_indices[lab].append(i)

    # shuffle each label
    for lab in list(per_label_indices.keys()):
        rng.shuffle(per_label_indices[lab])

    eval_idx: List[int] = []
    # (a) min quota
    for lab in labels_sorted:
        pool = per_label_indices.get(lab, [])
        take = min(MIN_PER_CLASS_EVAL, len(pool))
        eval_idx.extend(pool[:take])
        per_label_indices[lab] = pool[take:]

        if len(eval_idx) >= N_EVAL:
            break

    if len(eval_idx) > N_EVAL:
        eval_idx = eval_idx[:N_EVAL]

    # (b) fill remaining
    if len(eval_idx) < N_EVAL:
        remaining = []
        for lab in labels_sorted:
            remaining.extend(per_label_indices.get(lab, []))
        rng.shuffle(remaining)
        need = N_EVAL - len(eval_idx)
        if len(remaining) < need:
            raise RuntimeError(f"Not enough data to fill eval to {N_EVAL}. remaining={len(remaining)}")
        eval_idx.extend(remaining[:need])

    rng.shuffle(eval_idx)
    eval_ds = ds.select(eval_idx)

    # --------- 5) 构造 corpus_labels（分类任务 candidates=labels） ----------
    # corpus-id 用 0..C-1
    labels_final = sorted({str(r[label_col]) for r in eval_ds})
    label2id = {lab: i for i, lab in enumerate(labels_final)}

    corpus_rows = []
    for lab, cid in label2id.items():
        corpus_rows.append({
            "corpus-id": str(cid),
            "label": lab,
            "text": lab,  # 兼容下游把 cand_text 当作 text
        })

    # --------- 6) 构造 query + qrels ----------
    # query-id 用 WavPath 的 basename（更短），避免特别长；同时保留原 WavPath 方便 debug
    query_rows = []
    qrels_rows = []

    for r in eval_ds:
        wav_path = str(r["WavPath"])
        qid = os.path.basename(wav_path)  # e.g., "acoustic_guitar_000-123.wav"
        lab = str(r[label_col])
        cid = str(label2id[lab])

        # query: 保留 audio 对象（你的 pipeline 通常需要）
        query_rows.append({
            "query-id": qid,
            "WavPath": wav_path,
            "audio": r.get("audio"),  # parquet 里一般是 Audio 特征结构
            "label": lab,
            "label_id": int(cid),
        })

        # qrels: 复用 query-id/corpus-id/score 格式
        qrels_rows.append({
            "query-id": qid,
            "corpus-id": cid,
            "score": 1,
        })

    # --------- 7) 写出 ----------
    write_parquet(query_rows, os.path.join(out_eval, "query.parquet"))
    write_parquet(qrels_rows, os.path.join(out_eval, "qrels.parquet"))
    write_parquet(corpus_rows, os.path.join(out_eval, "corpus_labels.parquet"))

    print("[DONE] NSynth-1k (eval-only) built at:", OUT_DIR)
    print(f" - eval queries: {len(query_rows)} (target={N_EVAL})")
    print(f" - labels(candidates): {len(corpus_rows)} (<=1w always for classification)")
    print(f" - label_col used: {label_col}")
    print(" - files:")
    print(f"   * {out_eval}/query.parquet")
    print(f"   * {out_eval}/qrels.parquet")
    print(f"   * {out_eval}/corpus_labels.parquet")


if __name__ == "__main__":
    main()