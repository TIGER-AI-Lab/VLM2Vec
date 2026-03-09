"""
Clotho 文本->音频检索数据集处理（训练/评测共用）。
- 训练：按音频聚合，每条样本包含该音频的多条 caption。
- 评测：每条音频的 5 条 caption 展开为 5 条查询。
"""

import os
from typing import Any, Dict, List, Tuple

import datasets
import pandas as pd
import torchaudio

from src.data.dataset.base_pair_dataset import AutoPairDataset
from src.utils.basic_utils import print_master
from src.data.eval_dataset.audio_instruction_utils import build_query_text


def _is_valid_audio_path(path: str) -> bool:
    try:
        info = torchaudio.info(path)
        return info.num_frames > 0 and info.sample_rate > 0
    except Exception:
        return False


def _expand_clotho_rows(csv_path: str, audio_dir: str) -> List[Dict[str, str]]:
    """将 Clotho CSV 每行展开为 5 条 (text, audio_path)。"""
    df = pd.read_csv(csv_path)
    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        file_name = row["file_name"]
        audio_path = os.path.join(audio_dir, file_name)
        for i in range(1, 6):
            cap = row.get(f"caption_{i}")
            if not isinstance(cap, str):
                continue
            records.append(
                {
                    "text": cap.strip(),
                    "audio_path": audio_path,
                    "file_name": file_name,
                }
            )
    return records


def _group_clotho_rows(csv_path: str, audio_dir: str) -> List[Dict[str, Any]]:
    """按音频聚合 Clotho CSV，每条记录包含该音频所有可用 captions。"""
    df = pd.read_csv(csv_path)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        file_name = row["file_name"]
        audio_path = os.path.join(audio_dir, file_name)
        captions: List[str] = []
        for i in range(1, 6):
            cap = row.get(f"caption_{i}")
            if not isinstance(cap, str):
                continue
            cap = cap.strip()
            if cap:
                captions.append(cap)
        if not captions:
            continue
        records.append(
            {
                "captions": captions,
                "audio_path": audio_path,
                "file_name": file_name,
            }
        )
    return records


@AutoPairDataset.register("load_audio_clotho_dataset")
def load_audio_clotho_dataset(*args: Any, **kwargs: Any):
    """
    加载 Clotho 文本->音频检索数据集（训练/finetune 场景，适配 train_collator_omni）。

    参数（kwargs）：
        data_path: 数据根目录，默认 /code/.cache/datasets/MMEB-v2_1/audio-tasks/clotho
        captions_csv: caption 文件名，默认 clotho_captions_development.csv
        audio_subdir: 音频子目录，默认 train/development

    返回：
        datasets.Dataset，字段：
            - query_text/query_image/query_audio
            - pos_text/pos_image/pos_audio
            - dataset_infos
    """
    data_path: str = kwargs.get(
        "data_path", "/code/.cache/datasets/MMEB-v2_1/audio-tasks/clotho"
    )
    captions_csv: str = kwargs.get("captions_csv", "clotho_captions_development.csv")
    audio_subdir: str = kwargs.get("audio_subdir", os.path.join("train", "development"))

    csv_path = os.path.join(data_path, captions_csv)
    audio_dir = os.path.join(data_path, audio_subdir)
    assert os.path.isfile(csv_path), f"未找到 caption CSV: {csv_path}"
    assert os.path.isdir(audio_dir), f"未找到音频目录: {audio_dir}"

    print_master(f"[Clotho] loading csv={csv_path}, audio_dir={audio_dir}")
    records = _group_clotho_rows(csv_path, audio_dir)
    print_master(f"[Clotho] 共生成音频样本数: {len(records)}")

    query_texts, query_images, query_audios = [], [], []
    pos_texts, pos_images, pos_audios = [], [], []
    dataset_infos = []

    for r in records:
        audio_path = r["audio_path"]
        if not os.path.isfile(audio_path):
            continue
        if not _is_valid_audio_path(audio_path):
            continue

        # Store all processed captions; collator will randomly pick one each step.
        query_text_candidates: List[str] = []
        for text in r["captions"]:
            query_text = build_query_text("Clotho", text)
            assert isinstance(query_text, list) and len(query_text) == 1 and isinstance(query_text[0], str) and query_text[0].strip()
            query_text_candidates.append(query_text[0])
        if not query_text_candidates:
            continue

        query_texts.append(query_text_candidates)
        query_images.append(None)
        query_audios.append(None)

        pos_texts.append("[AUDIO]")
        pos_images.append(None)
        pos_audios.append({"path": audio_path, "bytes": None, "start": None, "end": None})

        dataset_infos.append({"file_name": r["file_name"], "num_captions": len(query_text_candidates)})

    dataset = datasets.Dataset.from_dict(
        {
            "query_text": query_texts,
            "query_image": query_images,
            "query_audio": query_audios,
            "pos_text": pos_texts,
            "pos_image": pos_images,
            "pos_audio": pos_audios,
            "dataset_infos": dataset_infos,
        }
    )
    try:
        setattr(dataset, "num_rows", len(dataset))
    except (AttributeError, TypeError):
        pass
    return dataset


# ---------- 评测构建（文本->音频检索） ----------
def build_clotho_text_audio_dataset(path_info: Tuple[str, str, str], **kwargs):
    """
    返回 (query_dataset, corpus_dataset)，其中：
    - query_dataset: 只包含查询信息和标签
    - corpus_dataset: 包含所有候选，每个条目一个候选
    """
    data_path = path_info[0]
    captions_csv: str = kwargs.get("captions_csv", "clotho_captions_evaluation.csv")
    audio_subdir: str = kwargs.get("audio_subdir", "evaluation")

    csv_path = os.path.join(data_path, captions_csv)
    audio_dir = os.path.join(data_path, audio_subdir)
    assert os.path.isfile(csv_path), f"未找到 caption CSV: {csv_path}"
    assert os.path.isdir(audio_dir), f"未找到音频目录: {audio_dir}"

    records = _expand_clotho_rows(csv_path, audio_dir)
    # 构造候选池与映射
    file_names = [r["file_name"] for r in records]
    unique_files = list(dict.fromkeys(file_names))  # 保持顺序去重
    file2idx = {fn: i for i, fn in enumerate(unique_files)}

    texts = [r["text"] for r in records]
    label_ids = [file2idx[r["file_name"]] for r in records]

    # 构建query_text并添加debug assert
    query_texts = []
    for t in texts:
        query_text = build_query_text("Clotho", t)
        assert isinstance(query_text, list) and len(query_text) == 1 and isinstance(query_text[0], str) and query_text[0].strip()
        query_texts.append(query_text)

    # 构建query_dataset - 只包含查询信息和标签
    query_dataset = datasets.Dataset.from_dict({
        "query_text": query_texts,
        "query_image": [None for _ in texts],
        "query_audio": [None for _ in texts],
        "dataset_infos": [{"label_cand_id": lid, "file_name": fn, "label_name": fn} for lid, fn in zip(label_ids, file_names)],
    })

    # 构建corpus_dataset - 包含所有候选，每个条目一个候选
    corpus_rows = []
    for fn in unique_files:
        audio_obj = {"path": os.path.join(audio_dir, fn), "bytes": None}
        corpus_rows.append({
            "cand_text": ["[AUDIO]"],  # 音频候选的文本占位符
            "cand_image": [None],
            "cand_audio": [audio_obj],
            "dataset_infos": {"cand_names": [fn]},  # 用于 cand_embed_dict 的key
        })

    corpus_dataset = datasets.Dataset.from_list(corpus_rows)

    return query_dataset, corpus_dataset
