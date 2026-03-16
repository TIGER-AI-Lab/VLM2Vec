"""
TUTSound 音频事件检测数据集处理（训练版本）
- 文本标签 -> 音频片段检索训练数据处理
- 参考 eval_dataset/tutsound_dataset.py 的处理方式
"""

import os
from typing import Dict, List, Tuple, Any

import datasets
import torchaudio
from src.utils.dataset_utils import sample_dataset
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.data.dataset.base_pair_dataset import AutoPairDataset


POS_TEXT_AUDIO = "[AUDIO]"


def _read_evaluate_file(eval_path: str) -> List[Tuple[str, str, float, float, str]]:
    rows = []
    with open(eval_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                parts = line.split()
            if len(parts) < 5:
                continue
            fp = parts[0].strip()
            scene = parts[1].strip()
            onset = float(parts[2])
            offset = float(parts[3])
            label = parts[4].strip()
            rows.append((fp, scene, onset, offset, label))
    return rows


def _segment_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return 0.0 if union <= 0 else inter / union


def _seg_id(fp: str, start: float, end: float) -> str:
    return f"{fp}|{start:.3f}-{end:.3f}"


def _normalize_segment(start: float, end: float, audio_dur: float, min_dur: float = 0.05) -> Tuple[float, float]:
    """
    Ensure a valid non-empty segment within [0, audio_dur].
    Some raw labels in TUTSound may have onset==offset.
    """
    s = max(0.0, float(start))
    e = float(end)
    if e <= s:
        e = s + float(min_dur)
    if e > audio_dur:
        e = audio_dur
    if e <= s:
        s = max(0.0, audio_dur - float(min_dur))
        e = audio_dur
    if e <= s:
        e = s + 1e-3
    return s, e


@AutoPairDataset.register("build_tutsound_audio_dataset")
def build_tutsound_audio_dataset(*args: Any, **kwargs: Any):
    """
    TUTSound 文本标签->音频片段检索训练数据（适配 train_collator_omni）。
    参考 eval_dataset/tutsound_dataset.py 的处理方式。

    返回：
        datasets.Dataset，字段：
            - query_text/query_image/query_audio
            - pos_text/pos_image/pos_audio
            - dataset_infos
    """
    path_info = kwargs.get("path_info")
    if path_info is not None:
        dataset_path, _, split = path_info
    else:
        dataset_path = kwargs.get("data_path")
        split = kwargs.get("dataset_split") or kwargs.get("split") or ""
    if not dataset_path:
        raise ValueError("TUTSound: data_path is required")
    base_dir = os.path.join(dataset_path, "TUT-sound-events-2017-development")

    # 负例生成参数（可选）
    neg_win_len = float(kwargs.get("neg_win_len", 1.0))
    neg_stride = float(kwargs.get("neg_stride", 0.5))
    iou_thresh = float(kwargs.get("neg_iou_thresh", 0.1))
    neg_max_per_query = int(kwargs.get("neg_max_per_query", 50))

    # 1) 读四个 fold 的 train 文件
    all_rows: List[Tuple[str, str, float, float, str]] = []
    for fold in ["1", "2", "3", "4"]:
        train_file_name = f"street_fold{fold}_train.txt"
        train_file = os.path.join(base_dir, "evaluation_setup", train_file_name)
        if os.path.isfile(train_file):
            fold_rows = _read_evaluate_file(train_file)
            all_rows.extend(fold_rows)
            print(f"TUTSound: loaded fold{fold} with {len(fold_rows)} events")
        else:
            print(f"Warning: TUTSound train file not found: {train_file}")

    if not all_rows:
        raise FileNotFoundError("No TUTSound train files found for any fold")

    # 2) 按 wav 聚合 GT 事件段
    gt_by_fp: Dict[str, Dict[str, Any]] = {}
    for fp, scene, onset, offset, label in all_rows:
        if fp not in gt_by_fp:
            gt_by_fp[fp] = {"scene": scene, "events": []}
        gt_by_fp[fp]["events"].append({"label": label, "onset": onset, "offset": offset})

    # 3) 构建训练样本：每个 GT event 一条，query=文本标签，pos=对应音频片段
    query_texts: List[List[str]] = []
    query_images: List[Any] = []
    query_audios: List[Any] = []
    pos_texts: List[str] = []
    pos_images: List[Any] = []
    pos_audios: List[Dict[str, Any]] = []
    dataset_infos: List[Dict[str, Any]] = []

    wav_list = sorted(gt_by_fp.keys())
    cand_text_item = build_query_text("TUTSound")
    assert (
        isinstance(cand_text_item, list)
        and len(cand_text_item) == 1
        and isinstance(cand_text_item[0], str)
        and cand_text_item[0].strip()
    )

    for fp in wav_list:
        abs_path = os.path.normpath(os.path.join(base_dir, fp))
        if not os.path.isfile(abs_path):
            continue
            
        info = gt_by_fp[fp]
        gt_events = info["events"]

        try:
            audio_info = torchaudio.info(abs_path)
            if audio_info.num_frames <= 0 or audio_info.sample_rate <= 0:
                print(f"Warning: Invalid audio info for {abs_path}: {audio_info}")
                continue
        except Exception as e:
            print(f"Warning: Failed to read audio info for {abs_path}: {e}")
            continue

        audio_dur = float(audio_info.num_frames) / float(audio_info.sample_rate)
        normalized_events = []
        for ev in gt_events:
            seg_s, seg_e = _normalize_segment(ev["onset"], ev["offset"], audio_dur=audio_dur)
            normalized_events.append({
                "label": ev["label"],
                "onset": seg_s,
                "offset": seg_e,
            })
        gt_events = normalized_events
        gt_segments = [(float(ev["onset"]), float(ev["offset"])) for ev in gt_events]
        # Relaxed positives: within the same file, any segment with the same label is considered correct.
        label_to_seg_ids: Dict[str, List[str]] = {}
        for ev in gt_events:
            seg_id = _seg_id(fp, float(ev["onset"]), float(ev["offset"]))
            label_to_seg_ids.setdefault(ev["label"], []).append(seg_id)
        for k in list(label_to_seg_ids.keys()):
            label_to_seg_ids[k] = list(dict.fromkeys(label_to_seg_ids[k]))

        # 背景滑窗候选（可选，用于负例采样）
        neg_segments = []
        if neg_win_len > 0 and neg_stride > 0 and audio_dur >= neg_win_len:
            t = 0.0
            while t + neg_win_len <= audio_dur:
                seg_start = t
                seg_end = t + neg_win_len
                max_iou = 0.0
                for gs, ge in gt_segments:
                    max_iou = max(max_iou, _segment_iou(seg_start, seg_end, gs, ge))
                    if max_iou >= iou_thresh:
                        break
                if max_iou < iou_thresh:
                    neg_segments.append((seg_start, seg_end))
                t += neg_stride

        # 负例采样上限（每个 query）
        if neg_max_per_query > 0 and len(neg_segments) > neg_max_per_query:
            neg_segments = neg_segments[:neg_max_per_query]

        # 为每个 GT 事件创建训练样本
        for ev in gt_events:
            # Query: 文本标签
            query_text = build_query_text("TUTSound", ev["label"])
            assert (
                isinstance(query_text, list)
                and len(query_text) == 1
                and isinstance(query_text[0], str)
                and query_text[0].strip()
            )
            
            query_texts.append(query_text)
            query_images.append(None)
            # Align training query modality with eval grounding:
            # query is full audio + event text, target is a grounded segment.
            query_audios.append({
                "path": abs_path,
                "bytes": None,
                "start": 0.0,
                "end": float(audio_dur),
            })

            # Positive: 对应的 GT 音频片段
            pos_texts.append(POS_TEXT_AUDIO)
            pos_images.append(None)
            pos_audios.append({
                "path": abs_path,
                "bytes": None,
                "start": float(ev["onset"]),
                "end": float(ev["offset"])
            })

            primary_seg_id = _seg_id(fp, float(ev["onset"]), float(ev["offset"]))
            positive_seg_ids = label_to_seg_ids.get(ev["label"], [primary_seg_id])
            dataset_infos.append(
                {
                    "file_path": fp,
                    "scene": info["scene"],
                    "gt_events": gt_events,
                    "query_event": ev,
                    "query_label": ev["label"],
                    # Relaxed criterion metadata (same file + same label).
                    "label_name": list(positive_seg_ids),
                    "primary_label_name": primary_seg_id,
                    "seg_id": primary_seg_id,
                }
            )

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

    # 可选：分层采样
    stratify_by = kwargs.get("stratify_by", None)
    max_per_label = kwargs.get("max_per_label", None)
    stratify_ratio = kwargs.get("stratify_ratio", None)
    min_per_label = kwargs.get("min_per_label", 1)
    seed = kwargs.get("seed", 17)
    num_sample_per_subset = kwargs.get("num_sample_per_subset", None)

    if (max_per_label is not None or stratify_ratio is not None) and stratify_by is None:
        stratify_by = "query_label"

    if stratify_by in {"label_name", "query_label"}:
        if stratify_ratio is None and max_per_label is None and isinstance(num_sample_per_subset, int):
            stratify_ratio = num_sample_per_subset / max(1, len(dataset))

        if stratify_ratio is not None or max_per_label is not None:
            import math
            import random

            label2indices: Dict[str, List[int]] = {}
            for idx, info in enumerate(dataset_infos):
                lab_value = info.get(stratify_by, "")
                if isinstance(lab_value, list):
                    # Backward compatibility: when stratify_by=label_name (now list), fall back to query_label.
                    lab_value = info.get("query_label", "")
                lab = str(lab_value)
                label2indices.setdefault(lab, []).append(idx)

            rng = random.Random(seed)
            selected_indices: List[int] = []
            for lab, idxs in label2indices.items():
                if max_per_label is not None:
                    k = min(int(max_per_label), len(idxs))
                else:
                    k = max(min_per_label, int(math.ceil(float(stratify_ratio) * len(idxs))))
                    k = min(k, len(idxs))
                if k <= 0:
                    continue
                if k == len(idxs):
                    selected_indices.extend(idxs)
                else:
                    selected_indices.extend(rng.sample(idxs, k))

            selected_indices = sorted(set(selected_indices))
            if selected_indices:
                dataset = dataset.select(selected_indices)

    if not stratify_by and isinstance(num_sample_per_subset, int):
        dataset = sample_dataset(dataset, **kwargs)
    
    try:
        setattr(dataset, "num_rows", len(dataset))
    except (AttributeError, TypeError):
        pass
    
    return dataset
