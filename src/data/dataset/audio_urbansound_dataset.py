"""
UrbanSound8K 音频分类训练数据集（适配 train_collator_omni）。
仅保留 UrbanSound8K 相关逻辑。
"""

import os
from typing import Any, Dict, List, Tuple

import datasets
import torchaudio

from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.utils.dataset_utils import sample_dataset
from src.data.dataset.base_pair_dataset import AutoPairDataset


def _is_valid_audio_path(path: str) -> bool:
    try:
        info = torchaudio.info(path)
        return info.num_frames > 0 and info.sample_rate > 0
    except Exception:
        return False


def _resolve_split_csv(split: str) -> str:
    if not split:
        return "train.csv"
    split = split.lower()
    if split.endswith(".csv"):
        return split
    return f"{split}.csv"


def _load_urbansound8k_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    dataset_path, _, split = path_info
    csv_name = _resolve_split_csv(split or "train")
    csv_path = os.path.join(dataset_path, "csv_files", csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"UrbanSound8K csv not found: {csv_path}")

    import pandas as pd

    df = pd.read_csv(csv_path)
    if "path" in df.columns:
        df["path"] = df["path"].apply(lambda p: os.path.normpath(os.path.join(dataset_path, p)))
    elif "slice_file_name" in df.columns and "fold" in df.columns:
        df["path"] = df.apply(
            lambda x: os.path.normpath(
                os.path.join(dataset_path, "audio", f"fold{x['fold']}", x["slice_file_name"])
            ),
            axis=1,
        )
    else:
        raise ValueError(f"UrbanSound8K csv missing path columns, columns={df.columns}")

    return datasets.Dataset.from_pandas(df, preserve_index=False)


@AutoPairDataset.register("load_audio_urbansound8k_dataset")
def load_audio_urbansound8k_dataset(*args: Any, **kwargs: Any):
    """
    返回：
        datasets.Dataset，字段：
            - query_text/query_image/query_audio
            - pos_text/pos_image/pos_audio
            - dataset_infos
    """
    path_info = kwargs.get("path_info")
    if path_info is not None:
        dataset_path, subset, split = path_info
    else:
        dataset_path = kwargs.get("data_path")
        subset = kwargs.get("dataset_subset", "")
        split = kwargs.get("dataset_split") or kwargs.get("split") or ""
    if not dataset_path:
        raise ValueError("UrbanSound8K: data_path is required")
    dataset_name = kwargs.get("dataset_name", "UrbanSound8K")
    split_name = split or "train"

    dataset = _load_urbansound8k_dataset((dataset_path, subset, split_name))
    dataset = sample_dataset(dataset, **kwargs)

    label_names = None
    for key in ["class", "classname", "label_name", "category"]:
        if key in dataset.column_names:
            label_names = dataset[key]
            break
    if label_names is None:
        raise ValueError(f"UrbanSound8K: no label name field in {dataset.column_names}")

    all_label_names = sorted(list(set(str(x) for x in label_names)))
    label2id = {name: i for i, name in enumerate(all_label_names)}

    query_texts: List[List[str]] = []
    query_images: List[Any] = []
    query_audios: List[Dict[str, Any]] = []
    pos_texts: List[str] = []
    pos_images: List[Any] = []
    pos_audios: List[Any] = []
    dataset_infos: List[Dict[str, Any]] = []

    for row in dataset:
        audio_path = row.get("path") or row.get("audio_path")
        if not audio_path or not os.path.isfile(audio_path):
            continue
        if not _is_valid_audio_path(audio_path):
            continue
        label_name = (
            row.get("class")
            or row.get("classname")
            or row.get("label_name")
            or row.get("category")
        )
        if label_name is None:
            continue
        label_name = str(label_name)
        label_id = label2id.get(label_name, -1)

        query_text = build_query_text(dataset_name)
        assert (
            isinstance(query_text, list)
            and len(query_text) == 1
            and isinstance(query_text[0], str)
            and query_text[0].strip()
        )

        query_texts.append(query_text)
        query_images.append(None)
        query_audios.append({"path": audio_path, "bytes": None, "start": None, "end": None})

        pos_texts.append(label_name)
        pos_images.append(None)
        pos_audios.append(None)

        dataset_infos.append(
            {
                "label_id": label_id,
                "label_name": label_name,
                "audio_path": audio_path,
            }
        )

    out_dataset = datasets.Dataset.from_dict(
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
        setattr(out_dataset, "num_rows", len(out_dataset))
    except (AttributeError, TypeError):
        pass
    return out_dataset
