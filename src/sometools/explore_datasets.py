#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMEB-v2_1 音频任务数据集统计（metadata-only，不触发音频解码）
输出每个数据集：
- #Queries
- #Candidates
- positives per query（若能从 qrels 统计）
- total pairs（若为检索任务）

设计目标：稳定、不破坏环境、不依赖 torchcodec/librosa/soundfile/ffmpeg
"""

import os
import sys
import glob
from collections import defaultdict

try:
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("需要 pyarrow 才能读取 parquet metadata：pip/conda 安装 pyarrow") from e


ROOT = "/code/.cache/datasets/MMEB-v2_1/audio-tasks"


# -------------------------
# utils
# -------------------------
def human_int(x: int) -> str:
    return f"{x:,}"


def parquet_num_rows(files):
    total = 0
    for f in files:
        try:
            total += pq.ParquetFile(f).metadata.num_rows
        except Exception as e:
            print(f"⚠️ parquet metadata 读取失败: {f} ({e})")
    return total


def parquet_columns_one_file(f):
    pf = pq.ParquetFile(f)
    return pf.schema.names


def pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def qrels_stats_from_parquet(qrels_files):
    """
    统计 qrels 的 positives per query（按 qid 聚合），只读列，不解码音频。
    兼容 query_id/query-id 等命名。
    """
    if not qrels_files:
        return {"qrels_ok": 0, "qrels_reason": "no qrels files"}

    # 尝试从第一份 parquet 推断列名
    cols = parquet_columns_one_file(qrels_files[0])

    qid_cols = ["query_id", "query-id", "qid", "queryid", "query"]
    cid_cols = ["corpus_id", "corpus-id", "doc_id", "doc-id", "docid", "cid", "id"]
    rel_cols = ["relevance", "rel", "label", "score"]

    qid_col = pick_first_existing(cols, qid_cols)
    cid_col = pick_first_existing(cols, cid_cols)
    rel_col = pick_first_existing(cols, rel_cols)

    if not qid_col or not cid_col:
        return {
            "qrels_ok": 0,
            "qrels_reason": f"missing qid/cid columns. found={cols}"
        }

    # 逐文件扫描 qrels（只读需要的列）
    q_pos = defaultdict(int)
    q_any = set()

    for f in qrels_files:
        try:
            table = pq.read_table(f, columns=[qid_col, cid_col] + ([rel_col] if rel_col else []))
        except Exception as e:
            return {"qrels_ok": 0, "qrels_reason": f"pyarrow read_table failed: {e}"}

        qids = table[qid_col].to_pylist()
        if rel_col:
            rels = table[rel_col].to_pylist()
        else:
            rels = [1] * len(qids)

        for qid, rel in zip(qids, rels):
            q_any.add(qid)
            # 只要出现一条 qrels 就算一个 positive（通常每个 qid 就 1 条）
            # 如果你的 qrels 有多条正例，这里会统计为多条。
            if rel is None:
                continue
            try:
                if float(rel) > 0:
                    q_pos[qid] += 1
            except Exception:
                # rel 不是数也没关系，出现即算
                q_pos[qid] += 1

    if not q_any:
        return {"qrels_ok": 0, "qrels_reason": "empty qrels"}

    pos_counts = [q_pos.get(q, 0) for q in q_any]
    return {
        "qrels_ok": 1,
        "num_qrels_queries": len(q_any),
        "pos_min": float(min(pos_counts)),
        "pos_avg": float(sum(pos_counts) / len(pos_counts)),
        "pos_max": float(max(pos_counts)),
        "qid_col": qid_col,
        "cid_col": cid_col,
        "rel_col": rel_col or "N/A",
    }


def count_csv_rows(csv_path):
    # 粗略统计：减去表头（如果没有表头也不会太离谱）
    n = 0
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)


def find_parquets_under(path):
    return sorted(glob.glob(os.path.join(path, "**/*.parquet"), recursive=True))


# -------------------------
# dataset explorers (metadata-only)
# -------------------------
def explore_ave():
    print("\n=== AVE Dataset (Audio->Video Retrieval) [metadata-only] ===")
    data_path = os.path.join(ROOT, "AVE", "AVE_Dataset")

    # AVE 多数不是 parquet；你之前脚本已能得到 402/402
    # 这里给一个保守策略：如果能找到 split list，就统计行数
    split_files = glob.glob(os.path.join(data_path, "**/*Set*.txt"), recursive=True) + \
                  glob.glob(os.path.join(data_path, "**/*test*.txt"), recursive=True)
    # 你用的是 testSet.txt
    split_file = os.path.join(data_path, "testSet.txt")
    if os.path.exists(split_file):
        n = 0
        with open(split_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    n += 1
        num_queries = n
        num_candidates = n  # AVE 典型 1:1
        print(f"📊 Queries: {num_queries}")
        print(f"🎯 Candidates: {num_candidates} (AVE 1:1 pairing)")
        print(f"✅ Positives per Query: 1.00")
        print(f"📈 Total pairs: {human_int(num_queries * num_candidates)}")
    else:
        print(f"⚠️ 未找到 testSet.txt；候选 split_files: {split_files[:5]}")


def explore_clotho():
    print("\n=== Clotho Dataset (Text->Audio Retrieval) [metadata-only] ===")
    data_path = os.path.join(ROOT, "clotho")

    # queries: 由 clotho_captions_evaluation.csv 行数得到（你之前 output 就是这样）
    csv_path = os.path.join(data_path, "clotho_captions_evaluation.csv")
    if not os.path.exists(csv_path):
        # 兼容你日志里的路径名
        csv_path = os.path.join(data_path, "clotho_captions_evaluation.csv")
    if os.path.exists(csv_path):
        num_queries = count_csv_rows(csv_path)
        print(f"📊 Queries: {num_queries} (from {os.path.basename(csv_path)})")
    else:
        num_queries = None
        print("❌ 未找到 clotho_captions_evaluation.csv，无法用 metadata-only 统计 queries")

    # candidates: evaluation 目录下音频文件数（只数文件，不解码）
    audio_dir = os.path.join(data_path, "evaluation")
    auds = []
    for ext in (".wav", ".flac", ".ogg", ".mp3"):
        auds += glob.glob(os.path.join(audio_dir, f"**/*{ext}"), recursive=True)
    num_candidates = len(set(auds)) if auds else None
    if num_candidates is not None:
        print(f"🎯 Candidates: {num_candidates} (from audio files under evaluation)")
        print(f"✅ Positives per Query: 1.00 (Clotho eval 通常 1:1)")
        if num_queries is not None:
            print(f"📈 Total pairs: {human_int(num_queries * num_candidates)}")
    else:
        print("⚠️ 未扫描到 evaluation 下音频文件（可能音频不落盘/或目录名不同）")


def explore_sounddescs():
    print("\n=== SoundDescs Dataset (Text->Audio Retrieval) [metadata-only] ===")
    data_path = os.path.join(ROOT, "sounddescs")

    # 优先使用 query/corpus parquet（最准确且快）
    query_files = sorted(glob.glob(os.path.join(data_path, "**/query/*.parquet"), recursive=True))
    corpus_files = sorted(glob.glob(os.path.join(data_path, "**/corpus/*.parquet"), recursive=True))
    qrels_files = sorted(glob.glob(os.path.join(data_path, "**/qrels/*.parquet"), recursive=True))

    if query_files and corpus_files:
        num_queries = parquet_num_rows(query_files)
        num_candidates = parquet_num_rows(corpus_files)
        print(f"📊 Queries: {num_queries} (parquet metadata)")
        print(f"🎯 Candidates: {num_candidates} (parquet metadata)")
        qs = qrels_stats_from_parquet(qrels_files)
        if qs.get("qrels_ok", 0) == 1:
            print(f"✅ Positives per Query: min={qs['pos_min']:.2f}, avg={qs['pos_avg']:.2f}, max={qs['pos_max']:.2f}"
                  f" (qid={qs['qid_col']}, cid={qs['cid_col']}, rel={qs['rel_col']})")
        else:
            print(f"⚠️ qrels 统计失败: {qs.get('qrels_reason')}")
        print(f"📈 Total pairs: {human_int(num_queries * num_candidates)}")
        return

    # fallback: shard parquet
    shard_files = sorted(glob.glob(os.path.join(data_path, "**/*.parquet"), recursive=True))
    if shard_files:
        num_rows = parquet_num_rows(shard_files)
        print(f"📊 Queries: {num_rows} (parquet metadata; 未区分 query/corpus)")
        print(f"🎯 Candidates: {num_rows} (parquet metadata; 未区分 query/corpus)")
        print(f"📈 Total pairs: {human_int(num_rows * num_rows)}")
        return

    print("❌ sounddescs: 未找到 parquet 文件，无法统计")


def explore_speechcoco():
    print("\n=== SpeechCOCO Dataset (Audio->Image Retrieval) [metadata-only] ===")
    data_path = os.path.join(ROOT, "speechcoco")

    # 1) query/corpus/qrels 结构
    query_files = sorted(glob.glob(os.path.join(data_path, "**/query/*.parquet"), recursive=True))
    corpus_files = sorted(glob.glob(os.path.join(data_path, "**/corpus/*.parquet"), recursive=True))
    qrels_files = sorted(glob.glob(os.path.join(data_path, "**/qrels/*.parquet"), recursive=True))

    if query_files and corpus_files:
        num_queries = parquet_num_rows(query_files)
        num_candidates = parquet_num_rows(corpus_files)
        print(f"📊 Queries: {num_queries} (parquet metadata)")
        print(f"🎯 Candidates: {num_candidates} (parquet metadata)")
        qs = qrels_stats_from_parquet(qrels_files)
        if qs.get("qrels_ok", 0) == 1:
            print(f"✅ Positives per Query: min={qs['pos_min']:.2f}, avg={qs['pos_avg']:.2f}, max={qs['pos_max']:.2f}")
        else:
            print(f"⚠️ qrels 统计失败: {qs.get('qrels_reason')}")
        print(f"📈 Total pairs: {human_int(num_queries * num_candidates)}")
        return

    # 2) shard parquet（validation-000xx-of-xxxxx.parquet）
    shard_files = sorted(glob.glob(os.path.join(data_path, "**/validation-*.parquet"), recursive=True)) + \
                  sorted(glob.glob(os.path.join(data_path, "**/test-*.parquet"), recursive=True)) + \
                  sorted(glob.glob(os.path.join(data_path, "**/train-*.parquet"), recursive=True))
    if shard_files:
        num_rows = parquet_num_rows(shard_files)
        print(f"📊 Queries: {num_rows} (from shard parquets, metadata)")
        print("🎯 Candidates: (未在当前目录结构中找到 corpus/qrels；需要根据项目 speechcoco_dataset.py 的候选池定义统计)")
        return

    print("❌ speechcoco: 未找到可识别 parquet（query/corpus/qrels 或 shard）")


def explore_tutsound():
    print("\n=== TUT Sound Events Dataset (Audio Event Grounding) [metadata-only] ===")
    data_path = os.path.join(ROOT, "TUTSound")

    # TUT 同时使用四个fold的evaluate文件
    folds = ["1", "2", "3", "4"]
    total_events = 0
    total_unique_files = set()

    for fold in folds:
        eval_file = os.path.join(data_path, "TUT-sound-events-2017-development", "evaluation_setup", f"street_fold{fold}_evaluate.txt")
        if os.path.exists(eval_file):
            fold_events = 0
            fold_files = set()
            with open(eval_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        fold_events += 1
                        # 提取文件名 (第一列是audio_file)
                        parts = line.split('\t')
                        if len(parts) >= 1:
                            fold_files.add(parts[0])
            print(f"📄 Fold{fold}: {fold_events} events, {len(fold_files)} unique audio files")
            total_events += fold_events
            total_unique_files.update(fold_files)
        else:
            print(f"⚠️ Fold{fold} evaluate file not found: {eval_file}")

    if total_events > 0:
        print(f"📊 Total Events: {total_events}")
        print(f"🎵 Total Unique Audio Files: {len(total_unique_files)}")
        print("🎯 Candidates: N/A (grounding task)")
        print("✅ Note: TUT dataset combines all 4 folds for comprehensive evaluation")
    else:
        print("⚠️ 未找到任何 evaluation_setup/*evaluate*.txt 文件，无法统计")


def explore_cls():
    print("\n🏷️ 分类任务数据集 [metadata-only] ===")
    datasets_info = [
        ("CREMA-D", os.path.join(ROOT, "creamD")),
        ("ESC-50", os.path.join(ROOT, "esc50")),
        ("NSynth", os.path.join(ROOT, "nsynth")),
        ("UrbanSound8K", os.path.join(ROOT, "urbansound8k")),
        ("SpeechCommands", os.path.join(ROOT, "speechcommand")),
    ]

    for name, path in datasets_info:
        print(f"\n=== {name} (Audio Classification) [metadata-only] ===")
        if not os.path.exists(path):
            print(f"❌ 路径不存在: {path}")
            continue

        parquets = sorted(glob.glob(os.path.join(path, "**/*.parquet"), recursive=True))
        csvs = sorted(glob.glob(os.path.join(path, "**/*.csv"), recursive=True))

        if parquets:
            num_queries = parquet_num_rows(parquets)
            src = f"parquet({len(parquets)})"
        elif csvs:
            # UrbanSound8K 你之前就有 csv_files/test_6.csv
            num_queries = count_csv_rows(csvs[0])
            src = os.path.basename(csvs[0])
        else:
            auds = []
            for ext in (".wav", ".flac", ".ogg", ".mp3"):
                auds += glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True)
            num_queries = len(set(auds))
            src = "audio-file-scan"

        print(f"📊 Queries: {num_queries} (from {src})")
        print("🎯 Candidates(classes): (需要读取 label 映射或统计 label 列 unique；不同数据集格式不同，先不强行推断)")


def main():
    print("🔍 开始探查 MMEB-v2_1 音频任务数据集统计信息（metadata-only）...")
    print("=" * 90)
    print(f"ROOT = {ROOT}")

    print("\n🎯 检索任务数据集:")
    explore_ave()
    explore_clotho()
    explore_sounddescs()
    explore_speechcoco()

    print("\n🎯 Grounding 任务数据集:")
    explore_tutsound()

    print("\n🏷️ 分类任务数据集:")
    explore_cls()

    print("\n" + "=" * 90)
    print("✅ 完成！(metadata-only，不触发音频解码)")


if __name__ == "__main__":
        main()