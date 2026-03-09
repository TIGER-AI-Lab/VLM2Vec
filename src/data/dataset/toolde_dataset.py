import json
import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import datasets

from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook
from src.utils.basic_utils import print_master
from src.utils.dataset_utils import sample_dataset


def _resolve_data_file(data_path: str) -> str:
    if os.path.isfile(data_path):
        return data_path
    if os.path.isdir(data_path):
        candidates = [
            "output_response_all_converted_processed_5w-no-example-usage.json",
            "output_response_all_converted_processed_5w.json",
        ]
        for name in candidates:
            p = os.path.join(data_path, name)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"ToolDe json not found under: {data_path}")
    raise FileNotFoundError(f"ToolDe path not found: {data_path}")


def _load_toolde_json(data_path: str) -> datasets.Dataset:
    json_path = _resolve_data_file(data_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Unexpected ToolDe json format: {type(data)}")
    return datasets.Dataset.from_list(data)


def _normalize_response_list(resp: Any) -> List[str]:
    if isinstance(resp, str):
        text = resp.strip()
        return [text] if text else []
    if isinstance(resp, list):
        out: List[str] = []
        for x in resp:
            text = str(x).strip()
            if text:
                out.append(text)
        return out
    text = str(resp).strip()
    return [text] if text else []


def _group_toolde_by_query(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    将 Tool-DE 多行 (query, response) 聚合为 query 粒度：
    - 每个 query 仅保留一条样本
    - response 为该 query 的去重正样本列表
    """
    q2resp: "OrderedDict[str, List[str]]" = OrderedDict()
    q2seen: Dict[str, set] = {}

    for row in dataset:
        q = str(row.get("query", "")).strip()
        if not q:
            continue
        resp_list = _normalize_response_list(row.get("response", ""))
        if not resp_list:
            continue

        if q not in q2resp:
            q2resp[q] = []
            q2seen[q] = set()

        for r in resp_list:
            if r in q2seen[q]:
                continue
            q2seen[q].add(r)
            q2resp[q].append(r)

    queries = list(q2resp.keys())
    responses = [q2resp[q] for q in queries]
    return datasets.Dataset.from_dict({"query": queries, "response": responses})


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    """
    纯文本检索（query_text -> pos_text），适配 train_collator_omni。
    期望字段：query, response
    """
    queries = batch_dict.get("query", [])
    responses = batch_dict.get("response", [])

    # 检查长度匹配性，避免静默截断
    if len(queries) != len(responses):
        print_master(f"Warning: queries ({len(queries)}) and responses ({len(responses)}) length mismatch, using zip (will truncate longer list)")

    query_texts: List[List[str]] = []
    query_images: List[Any] = []
    query_audios: List[Any] = []
    pos_texts: List[str] = []
    pos_images: List[Any] = []
    pos_audios: List[Any] = []
    dataset_infos: List[Dict[str, Any]] = []

    for q, r in zip(queries, responses):
        q = str(q).strip()
        positives = _normalize_response_list(r)
        if not q or not positives:
            continue
        # Align schema with other tasks: query_text uses List[str] format.
        query_texts.append([q])
        query_images.append(None)
        query_audios.append(None)
        # Keep training schema aligned: pos_text is a single string.
        # Randomly sample one positive from grouped candidates.
        pos_texts.append(random.choice(positives))
        pos_images.append(None)
        pos_audios.append(None)
        dataset_infos.append({"num_positives": len(positives)})

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "query_audio": query_audios,
        "pos_text": pos_texts,
        "pos_image": pos_images,
        "pos_audio": pos_audios,
        "dataset_infos": dataset_infos,
    }


DATASET_PARSER_NAME = "toolDe"


@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_toolde_dataset(model_args, data_args, training_args, *args, **kwargs):
    """
    加载 Tool-De 训练数据（文本->文本检索）。

    参数:
        data_path: JSON 文件路径或目录（必填/默认本地）
    """
    default_path = "/code/.cache/datasets/MMEB-v2_1/tool-tasks/Tool-De-train"
    data_path = kwargs.get("data_path", default_path)

    dataset = _load_toolde_json(data_path)
    dataset = _group_toolde_by_query(dataset)
    dataset = sample_dataset(dataset, **kwargs)

    kwargs["global_dataset_name"] = f"{DATASET_PARSER_NAME}/train"

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=128,
        remove_columns=dataset.column_names,
        drop_last_batch=False,
    )

    # 在 map 后重新统计实际行数（可能因过滤而减少）
    num_rows = dataset.num_rows
    print_master(f"Loaded {DATASET_PARSER_NAME} dataset with {num_rows} samples")
    return dataset
