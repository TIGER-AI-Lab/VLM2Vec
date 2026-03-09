#!/usr/bin/env python3
import os
import glob
import random
from typing import Dict, Any, List, Tuple

import datasets

# ✅ pyarrow: 流式重写 corpus（展平 audio struct，避免 nested bug & 降低 CPU）
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ========= CONFIG =========
SRC_ROOT = "/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/sounddescs"
OUT_DIR  = "/data/mengrui/.cache/huggingface/datasets/MMEB-V3/audio-tasks/sounddescs-1k"

N_EVAL = 1000
SEED = 17

# ✅ 控制 CPU / 内存：batch 越小越省（但稍慢）
CORPUS_BATCH_SIZE = 256
PYARROW_USE_THREADS = False

# ✅ 只抽 eval，不再产生 train
WRITE_CORPUS = True  # 如果你 eval 只需要 query/qrels，可以设 False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_parquet_dir(dirpath: str) -> datasets.Dataset:
    files = sorted(glob.glob(os.path.join(dirpath, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet under: {dirpath}")
    return datasets.load_dataset("parquet", data_files={"data": files}, split="data")


def list_parquet_files(dirpath: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(dirpath, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet under: {dirpath}")
    return files


def pick_first_existing(cands: List[str], available: List[str], name: str) -> str:
    s = set(available)
    for c in cands:
        if c in s:
            return c
    raise KeyError(f"[{name}] cannot find any of {cands}. available={available}")


def dump_query(ds: datasets.Dataset, out_dir: str, qid_col: str):
    def _add(ex):
        ex["query-id"] = str(ex[qid_col])
        return ex
    ds2 = ds.map(_add, desc="add query-id", load_from_cache_file=False)
    ds2.to_parquet(os.path.join(out_dir, "query.parquet"))


def dump_qrels(ds: datasets.Dataset, out_dir: str, qid_col: str, qrels_by_qid: Dict[str, Tuple[str, float]]):
    rows = []
    for r in ds:
        qid = str(r[qid_col])
        pos_cid, _ = qrels_by_qid[qid]
        rows.append({"query-id": qid, "corpus-id": str(pos_cid), "score": 1})
    datasets.Dataset.from_list(rows).to_parquet(os.path.join(out_dir, "qrels.parquet"))


def rewrite_corpus_flat_pyarrow(corpus_files: List[str], out_path: str, cid_col: str):
    """
    流式重写 corpus：
    - 每次读取小批 record_batch
    - 如果有 audio(struct)，展平为 audio_path/audio_bytes/(audio_array/audio_sampling_rate)，并删除 audio
    - 确保有 corpus-id（string）
    """
    writer = None

    for fp in corpus_files:
        pf = pq.ParquetFile(fp)
        for rb in pf.iter_batches(batch_size=CORPUS_BATCH_SIZE, use_threads=PYARROW_USE_THREADS):
            t = pa.Table.from_batches([rb])

            # 1) corpus-id
            if "corpus-id" in t.column_names:
                corpus_id_arr = pc.cast(t["corpus-id"], pa.string())
            else:
                if cid_col not in t.column_names:
                    raise KeyError(f"cid_col={cid_col} not in columns: {t.column_names}")
                corpus_id_arr = pc.cast(t[cid_col], pa.string())
                t = t.append_column("corpus-id", corpus_id_arr)

            # 2) flatten audio struct
            if "audio" in t.column_names:
                audio_col = t["audio"]
                if pa.types.is_struct(audio_col.type):
                    fields = set(audio_col.type.names)
                    if "path" in fields and "audio_path" not in t.column_names:
                        t = t.append_column("audio_path", pc.struct_field(audio_col, "path"))
                    if "bytes" in fields and "audio_bytes" not in t.column_names:
                        t = t.append_column("audio_bytes", pc.struct_field(audio_col, "bytes"))
                    if "array" in fields and "audio_array" not in t.column_names:
                        t = t.append_column("audio_array", pc.struct_field(audio_col, "array"))
                    if "sampling_rate" in fields and "audio_sampling_rate" not in t.column_names:
                        t = t.append_column("audio_sampling_rate", pc.struct_field(audio_col, "sampling_rate"))
                # ✅ 关键：删除 nested audio，规避某些下游 nested 兼容问题
                t = t.drop(["audio"])

            if writer is None:
                writer = pq.ParquetWriter(out_path, t.schema, compression="snappy", use_dictionary=True)
            writer.write_table(t)

    if writer is None:
        raise RuntimeError("No data written for corpus (empty input?)")
    writer.close()


def main():
    rng = random.Random(SEED)

    out_eval = os.path.join(OUT_DIR, "eval")
    ensure_dir(out_eval)

    # 1) load query/qrels（相对小）
    query_ds = load_parquet_dir(os.path.join(SRC_ROOT, "query"))
    qrels_ds = load_parquet_dir(os.path.join(SRC_ROOT, "qrels"))

    # 2) corpus：不全量 load，只拿 parquet 文件列表 + schema probe
    corpus_dir = os.path.join(SRC_ROOT, "corpus")
    corpus_files = list_parquet_files(corpus_dir)

    # 仅用一个 shard probe schema 来找 cid_col
    corpus_schema_probe = datasets.load_dataset("parquet", data_files={"data": corpus_files[:1]}, split="data")
    cid_col = pick_first_existing(
        ["corpus_id", "corpus-id", "id", "audio_id", "docid"],
        corpus_schema_probe.column_names,
        "corpus",
    )

    qid_col = pick_first_existing(["query_id", "query-id", "id", "qid"], query_ds.column_names, "query")
    qrels_qid_col = pick_first_existing(["query-id", "query_id", "qid", "id"], qrels_ds.column_names, "qrels")
    qrels_cid_col = pick_first_existing(["corpus-id", "corpus_id", "cid", "docid", "id"], qrels_ds.column_names, "qrels")
    qrels_score_col = pick_first_existing(["score", "relevance", "rel"], qrels_ds.column_names, "qrels")

    # 3) 构建 qrels 映射：每个 qid 取最高分的 pos
    qrels_by_qid: Dict[str, Tuple[str, float]] = {}
    for r in qrels_ds:
        qid = str(r[qrels_qid_col])
        cid = str(r[qrels_cid_col])
        sc = float(r[qrels_score_col])
        if (qid not in qrels_by_qid) or (sc > qrels_by_qid[qid][1]):
            qrels_by_qid[qid] = (cid, sc)

    # 4) 用 pyarrow 扫一遍 corpus id（只读 cid_col 列）
    corpus_ids = set()
    for fp in corpus_files:
        pf = pq.ParquetFile(fp)
        for rb in pf.iter_batches(
            batch_size=2048,
            columns=[cid_col] if cid_col in pf.schema.names else None,
            use_threads=PYARROW_USE_THREADS,
        ):
            arr = rb.column(0)
            corpus_ids.update(str(x) for x in arr.to_pylist())

    # 5) 过滤可用 query（必须有 qrels 且 pos_cid 在 corpus）
    kept_idx = []
    for i, r in enumerate(query_ds):
        qid = str(r[qid_col])
        if qid not in qrels_by_qid:
            continue
        pos_cid, _ = qrels_by_qid[qid]
        if pos_cid not in corpus_ids:
            continue
        kept_idx.append(i)

    if len(kept_idx) < N_EVAL:
        raise RuntimeError(f"Not enough queries after filtering: kept={len(kept_idx)} < N_EVAL={N_EVAL}")

    rng.shuffle(kept_idx)
    eval_idx = kept_idx[:N_EVAL]
    eval_q = query_ds.select(eval_idx)

    # 6) 写 eval 的 query/qrels
    dump_query(eval_q, out_eval, qid_col)
    dump_qrels(eval_q, out_eval, qid_col, qrels_by_qid)

    # 7) 写 corpus（可选）
    if WRITE_CORPUS:
        rewrite_corpus_flat_pyarrow(
            corpus_files=corpus_files,
            out_path=os.path.join(out_eval, "corpus.parquet"),
            cid_col=cid_col,
        )

    print("[DONE] SoundDescs-1k (eval-only) built at:", OUT_DIR)
    print(" - eval queries:", len(eval_q), f"(target={N_EVAL})")
    print(" - corpus written:", os.path.join(out_eval, "corpus.parquet") if WRITE_CORPUS else "(skipped)")
    print(" - cid_col used:", cid_col)
    print(" - outputs:")
    print(f"   * {out_eval}/query.parquet")
    print(f"   * {out_eval}/qrels.parquet")
    if WRITE_CORPUS:
        print(f"   * {out_eval}/corpus.parquet")


if __name__ == "__main__":
    main()
