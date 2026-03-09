"""
AVE 音频->视频检索数据集工具（训练用，适配 train_collator_omni）。
- 解析 split 文件 (train/val/test)
- query: audio + instruction text
- pos: video represented by N frames (paths only)
"""

import os
from typing import List, Dict, Any, Tuple

import datasets
import torchaudio

from src.model.processor import process_input_text
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames
from src.data.dataset.base_pair_dataset import RESOLUTION_MAPPING, AutoPairDataset


TASK_INST_TGT = "Understand the content of the provided video."


def parse_ave_split(split_file: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Category"):
                continue
            parts = line.split("&")
            if len(parts) < 5:
                continue
            category, video_id, quality, start, end = parts[:5]
            clip_id = f"{video_id}_{int(float(start))}_{int(float(end))}"
            samples.append(
                {
                    "category": category,
                    "video_id": video_id,
                    "quality": quality,
                    "start": float(start),
                    "end": float(end),
                    "clip_id": clip_id,
                }
            )
    return samples


def _resolve_split_file(split: str, default_name: str) -> str:
    if not split:
        return default_name
    split = split.lower()
    if split.endswith(".txt"):
        return split
    mapping = {
        "train": "trainSet.txt",
        "training": "trainSet.txt",
        "val": "valSet.txt",
        "valid": "valSet.txt",
        "dev": "valSet.txt",
        "test": "testSet.txt",
    }
    return mapping.get(split, default_name)


@AutoPairDataset.register("load_audio_ave_dataset")
def load_audio_ave_dataset(*args: Any, **kwargs: Any):
    """
    返回 (dataset, corpus=None)，用于训练：
      - query_text: List[str]（只放一个指令）
      - query_audio: {"path": ..., "bytes": None}
      - pos_text: 带 video token 的指令
      - pos_image: 视频帧路径 (dict: paths/bytes/resolutions)
    """
    path_info = kwargs.get("path_info")
    if path_info is not None:
        data_path, _, split = path_info
    else:
        data_path = kwargs.get("data_path")
        split = kwargs.get("dataset_split") or kwargs.get("split") or ""
    if not data_path:
        raise ValueError("[AVE] data_path is required")
    split_file: str = kwargs.get("split_file", _resolve_split_file(split, "trainSet.txt"))
    audio_dir: str = kwargs.get("audio_dir", "audios")
    video_dir: str = kwargs.get("video_dir", "AVE")

    frame_root: str = kwargs.get("frame_root", os.path.join(data_path, "frames"))
    num_frames: int = kwargs.get("num_frames", 8)
    max_frames_saved: int = kwargs.get("max_frames_saved", 100)
    image_resolution: str = kwargs.get("image_resolution", "low")

    model_backbone = kwargs.get("model_backbone", None)
    if model_backbone is None:
        model_args = kwargs.get("model_args")
        if model_args is not None:
            model_backbone = getattr(model_args, "model_backbone", None)
    if model_backbone is None:
        raise ValueError("[AVE] model_backbone is required")

    split_path = os.path.join(data_path, split_file)
    audio_root = os.path.join(data_path, audio_dir)
    video_root = os.path.join(data_path, video_dir)

    assert os.path.isfile(split_path), f"未找到 split 文件: {split_path}"
    assert os.path.isdir(audio_root), f"未找到音频目录: {audio_root}"
    assert os.path.isdir(video_root), f"未找到视频目录: {video_root}"

    samples = parse_ave_split(split_path)

    # 1) 过滤掉缺失音频 / 缺失视频的样本（音频文件名是 video_id + '_' + clip_id）
    audio_records: List[Dict[str, Any]] = []
    for s in samples:
        audio_abs = os.path.join(audio_root, f"{s['video_id']}_{s['clip_id']}.wav")
        video_abs = os.path.join(video_root, f"{s['video_id']}.mp4")
        if not os.path.isfile(audio_abs):
            continue
        if not os.path.isfile(video_abs):
            continue
        try:
            info = torchaudio.info(audio_abs)
            if info.num_frames <= 0 or info.sample_rate <= 0:
                continue
        except Exception:
            continue
        audio_records.append(
            {
                "category": s["category"],
                "video_id": s["video_id"],
                "clip_id": s["clip_id"],
                "audio_path": audio_abs,
                "video_path": video_abs,
            }
        )

    if not audio_records:
        raise FileNotFoundError("[AVE] no valid audio/video pairs found after filtering")

    # 2) 构造训练样本（query: audio, pos: video frames）
    query_texts: List[List[str]] = []
    query_images: List[Any] = []
    query_audios: List[Dict[str, Any]] = []
    pos_texts: List[str] = []
    pos_images: List[Dict[str, Any]] = []
    pos_audios: List[Any] = []
    dataset_infos: List[Dict[str, Any]] = []

    pos_text = process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)
    resolution = RESOLUTION_MAPPING.get(image_resolution, None)

    for r in audio_records:
        qt = build_query_text("AVE")
        assert isinstance(qt, list) and len(qt) == 1 and isinstance(qt[0], str) and qt[0].strip()
        query_texts.append(qt)
        query_images.append(None)
        query_audios.append({"path": r["audio_path"], "bytes": None, "start": None, "end": None})

        frame_dir = os.path.join(frame_root, r["video_id"])
        save_frames(video_path=r["video_path"], frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        if not frame_paths:
            frame_paths = [None]

        pos_texts.append(pos_text)
        pos_images.append(
            {
                "paths": frame_paths,
                "bytes": [None] * len(frame_paths),
                "resolutions": [resolution] * len(frame_paths),
            }
        )
        pos_audios.append(None)

        dataset_infos.append(
            {
                "video_id": r["video_id"],
                "clip_id": r["clip_id"],
                "category": r["category"],
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

    return dataset
