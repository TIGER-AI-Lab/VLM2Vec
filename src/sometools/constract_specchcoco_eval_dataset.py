#!/usr/bin/env python3
import os
import glob
import random
from typing import List, Dict, Any

import datasets

# ========= CONFIG =========
SRC_DIR = "/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/speechcoco/data"
OUT_DIR = "/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/speechcoco-1k"

N_TEST = 1000          # ✅ 只需要测试集 query 数
CAND_MAX = 10000       # ✅ 测试候选池大小（unique image_id）
SEED = 17


def _find_files() -> List[str]:
    # 你原来读 validation-*.parquet，这里保留
    files = sorted(glob.glob(os.path.join(SRC_DIR, "validation-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet found under: {SRC_DIR}")
    return files


def _infer_image_col(ds: datasets.Dataset) -> str:
    candidates = ["image", "image_path", "image_file", "img", "img_path"]
    cols = set(ds.column_names)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"Cannot find image column in dataset columns={sorted(cols)[:50]}...")


def _require_cols(ds: datasets.Dataset, cols: List[str]):
    missing = [c for c in cols if c not in ds.column_names]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available={ds.column_names}")


def _write_parquet(obj: Dict[str, list], path: str):
    datasets.Dataset.from_dict(obj).to_parquet(path)


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    files = _find_files()
    ds = datasets.load_dataset("parquet", data_files={"data": files}, split="data")

    _require_cols(ds, ["id", "image_id", "audio"])
    image_col = _infer_image_col(ds)
    print(f"[INFO] total rows={len(ds)} | image_col={image_col}")

    # 1) 收集 unique image_id -> image object（取第一次出现）
    img_map: Dict[str, Any] = {}
    for row in ds:
        img_id = str(row["image_id"])
        if img_id not in img_map:
            img_map[img_id] = row[image_col]

    all_img_ids = list(img_map.keys())
    rng.shuffle(all_img_ids)
    print(f"[INFO] unique image_id={len(all_img_ids)}")

    # 2) 选 test corpus（最多 CAND_MAX 个 unique image）
    test_img_ids = all_img_ids[: min(CAND_MAX, len(all_img_ids))]
    test_cand_set = set(test_img_ids)

    if len(test_img_ids) == 0:
        raise RuntimeError("No images collected for corpus (empty dataset?)")

    print(f"[INFO] corpus_test={len(test_img_ids)} (target<= {CAND_MAX})")

    # 3) 从原始 rows 中筛出“正例 image_id 在 test 候选池内”的 rows，再抽 N_TEST
    test_rows = []
    for row in ds:
        img_id = str(row["image_id"])
        if img_id in test_cand_set:
            test_rows.append(row)

    rng.shuffle(test_rows)

    if len(test_rows) < N_TEST:
        raise RuntimeError(f"Not enough test rows after filtering: {len(test_rows)} < {N_TEST}")

    test_rows = test_rows[:N_TEST]

    # 4) 写 corpus（测试候选池）
    _write_parquet(
        {"corpus-id": test_img_ids, "image": [img_map[iid] for iid in test_img_ids]},
        os.path.join(OUT_DIR, "corpus_test_10k.parquet"),
    )
    print("[OK] wrote corpus_test_10k.parquet")

    # 5) 写 query + qrels（测试集）
    query = {
        "id": [str(r["id"]) for r in test_rows],
        "audio": [r["audio"] for r in test_rows],
        "image_id": [str(r["image_id"]) for r in test_rows],
    }
    _write_parquet(query, os.path.join(OUT_DIR, "query_test.parquet"))

    qrels = {
        "query-id": [str(r["id"]) for r in test_rows],
        "corpus-id": [str(r["image_id"]) for r in test_rows],
        "score": [1 for _ in test_rows],
    }
    _write_parquet(qrels, os.path.join(OUT_DIR, "qrels_test.parquet"))

    print("[OK] wrote query_test.parquet and qrels_test.parquet")

    print("\n[DONE] SpeechCOCO lite (test-only) built at:", OUT_DIR)
    print(f" - test: q={len(test_rows)}, c={len(test_img_ids)}")


if __name__ == "__main__":
    main()
