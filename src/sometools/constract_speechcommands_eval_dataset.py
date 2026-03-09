#!/usr/bin/env python3
import os
import glob
import random
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

import datasets

# ========= CONFIG =========
SRC_ROOT = "/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/speechcommand"
OUT_DIR  = "/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/speechcommand-1k"

N_TEST = 1_000
SEED = 17

# 候选(label)控制：None=保留全部；否则只保留出现频率最高的前 K 个类
MAX_CLASSES: Optional[int] = None  # e.g., 20

# 每类至少在 test 里保留的最少样本数（避免某类 test 消失）
MIN_PER_CLASS_TEST = 1

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_parquet(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        raise RuntimeError(f"Empty rows for {out_path}")
    keys = list(rows[0].keys())
    obj = {k: [r.get(k) for r in rows] for k in keys}
    datasets.Dataset.from_dict(obj).to_parquet(out_path)


def list_audio_files_by_label(root: str) -> Dict[str, List[str]]:
    """
    SpeechCommands 常见结构：root/<label>/*.wav
    排除 _background_noise_ 以及隐藏目录。
    """
    label2files = defaultdict(list)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"SRC_ROOT not found: {root}")

    for label in sorted(os.listdir(root)):
        if label.startswith("."):
            continue
        if label == "_background_noise_":
            continue
        lab_dir = os.path.join(root, label)
        if not os.path.isdir(lab_dir):
            continue

        files = []
        for ext in AUDIO_EXTS:
            files.extend(glob.glob(os.path.join(lab_dir, f"*{ext}")))
        if files:
            uniq = sorted(set(files))
            label2files[label].extend(uniq)

    return label2files


def pick_labels(label2files: Dict[str, List[str]], max_classes: Optional[int]) -> List[str]:
    counts = {lab: len(fs) for lab, fs in label2files.items()}
    labs = sorted(counts.keys(), key=lambda x: (-counts[x], x))
    if max_classes is not None:
        labs = labs[:max_classes]
    labs = [l for l in labs if counts.get(l, 0) > 0]
    if not labs:
        raise RuntimeError("No labels found after filtering")
    return labs


def stratified_sample(
    label2files: Dict[str, List[str]],
    labels: List[str],
    n_test: int,
    rng: random.Random,
    min_per_class: int,
) -> List[Tuple[str, str]]:
    """
    只抽 test 的 (filepath, label) pairs，做简单分层抽样：
      1) 每类先拿 min_per_class
      2) 再用剩余样本补齐到 n_test
    """
    pool = {lab: list(label2files[lab]) for lab in labels}
    for lab in labels:
        rng.shuffle(pool[lab])

    test: List[Tuple[str, str]] = []

    # 1) min quota
    for lab in labels:
        take = min(min_per_class, len(pool[lab]))
        for _ in range(take):
            test.append((pool[lab].pop(), lab))

    if len(test) > n_test:
        raise RuntimeError(
            f"Test min quota too large: test={len(test)} > target={n_test}. "
            f"Reduce MIN_PER_CLASS_TEST or MAX_CLASSES."
        )

    # 2) fill remaining
    remaining: List[Tuple[str, str]] = []
    for lab in labels:
        remaining.extend([(fp, lab) for fp in pool[lab]])
    rng.shuffle(remaining)

    while len(test) < n_test and remaining:
        test.append(remaining.pop())

    if len(test) < n_test:
        raise RuntimeError(
            f"Not enough data to fill test target={n_test}. current={len(test)}. "
            f"Try reduce N_TEST or MAX_CLASSES."
        )

    rng.shuffle(test)
    return test


def build_label_candidates(labels: List[str]) -> List[Dict[str, Any]]:
    # classification candidates = labels
    return [{"corpus-id": str(i), "label": lab, "text": lab} for i, lab in enumerate(labels)]


def build_queries(pairs: List[Tuple[str, str]], label2id: Dict[str, int]) -> List[Dict[str, Any]]:
    rows = []
    for idx, (fp, lab) in enumerate(pairs):
        qid = f"test-{idx:06d}"
        rows.append({
            "query-id": qid,
            "audio": {"path": fp, "bytes": None},
            "audio_path": fp,
            "label": lab,
            "label_id": int(label2id[lab]),
            "split": "test",
        })
    return rows


def build_qrels(pairs: List[Tuple[str, str]], label2id: Dict[str, int]) -> List[Dict[str, Any]]:
    rows = []
    for idx, (_, lab) in enumerate(pairs):
        qid = f"test-{idx:06d}"
        rows.append({
            "query-id": qid,
            "corpus-id": str(label2id[lab]),
            "score": 1,
        })
    return rows


def main():
    rng = random.Random(SEED)
    ensure_dir(OUT_DIR)

    # 1) 扫描音频文件
    label2files = list_audio_files_by_label(SRC_ROOT)
    if not label2files:
        raise RuntimeError(f"No label folders with audio found under: {SRC_ROOT}")

    # 2) label 候选控制（决定 candidates 的 label 集合）
    labels = pick_labels(label2files, MAX_CLASSES)

    # 3) stratified test sampling：test=1k
    test_pairs = stratified_sample(
        label2files=label2files,
        labels=labels,
        n_test=N_TEST,
        rng=rng,
        min_per_class=MIN_PER_CLASS_TEST,
    )

    # 4) candidates(labels)
    candidates = build_label_candidates(labels)
    label2id = {c["label"]: int(c["corpus-id"]) for c in candidates}

    # 5) queries / qrels
    test_queries = build_queries(test_pairs, label2id)
    test_qrels   = build_qrels(test_pairs, label2id)

    # 6) 写文件
    write_parquet(candidates,   os.path.join(OUT_DIR, "corpus_labels.parquet"))
    write_parquet(test_queries, os.path.join(OUT_DIR, "query_test.parquet"))
    write_parquet(test_qrels,   os.path.join(OUT_DIR, "qrels_test.parquet"))

    # 7) 统计
    cnt = Counter([lab for _, lab in test_pairs])
    top5 = cnt.most_common(5)

    print("[DONE] SpeechCommands (test-only) built at:", OUT_DIR)
    print(f" - labels(candidates): {len(labels)} (MAX_CLASSES={MAX_CLASSES})")
    print(f" - test: q={len(test_pairs)} | classes_present={len(cnt)} | top5={top5}")
    print(" - files:")
    print("    corpus_labels.parquet")
    print("    query_test.parquet")
    print("    qrels_test.parquet")


if __name__ == "__main__":
    main()
