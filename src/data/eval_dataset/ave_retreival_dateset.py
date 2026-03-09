import os
import subprocess
from typing import List, Dict, Any, Tuple

import datasets
from datasets import Audio
from PIL import Image

from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    add_metainfo_hook,
    ImageVideoInstance,
    RESOLUTION_MAPPING,
)
from src.utils.dataset_utils import sample_dataset, load_hf_dataset
from src.constant.dataset_hflocal_path import BASE_RAW_DATA_DIR
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text


TASK_INST_TGT = "Understand the content of the provided video."


def _resolve_existing_dir(candidates: List[str]) -> str:
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return ""


def _case_variants(path: str) -> List[str]:
    if not path:
        return []
    variants = [path]
    variants.append(path.replace("/audio-tasks/AVE/", "/audio-tasks/ave/"))
    variants.append(path.replace("/audio-tasks/ave/", "/audio-tasks/AVE/"))
    variants.append(path.replace("/audio-tasks/AVE", "/audio-tasks/ave"))
    variants.append(path.replace("/audio-tasks/ave", "/audio-tasks/AVE"))
    dedup = []
    seen = set()
    for p in variants:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def _audio_filename(video_id: str, clip_id: str) -> str:
    return f"{video_id}_{clip_id}.wav"


def _extract_clip_audio_ffmpeg(video_path: str, start: float, end: float, out_wav: str) -> bool:
    if os.path.isfile(out_wav):
        return True
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    duration = max(float(end) - float(start), 0.05)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(float(start)),
        "-t",
        str(duration),
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        out_wav,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    return os.path.isfile(out_wav)


def _parse_split_file(split_file: str) -> List[Dict[str, Any]]:
    """
    解析 AVE 的 split 文件（如 testSet.txt），格式：
    Category&VideoID&Quality&StartTime&EndTime
    """
    samples = []
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


@add_metainfo_hook
def data_prepare(batch_dict, all_video_ids=None, **kwargs):
    """
    为 AVE audio->video 检索生成：
    - query_audio: 音频对象（已由 HF datasets 自动解码）
    - query_text: 查询指令文本
    - query_image: None 占位
    - cand_video: 候选视频池（全局固定，每个视频包含8帧）
    - cand_text: 候选视频描述（全局固定）
    - dataset_infos: 包含正确的候选ID和标签信息
    """
    assert all_video_ids is not None and len(all_video_ids) > 0

    # 3. 在 data_prepare 里构造 local 需要的三个字段（扁平 + 可复用）
    video_root = kwargs.get("video_root", "")
    frame_root = kwargs.get("frame_root", "")
    num_frames = kwargs.get("num_frames", 8)
    max_frames_saved = kwargs.get("max_frames_saved", 100)
    image_resolution = kwargs.get("image_resolution", "low")
    model_backbone = kwargs.get("model_backbone")
    
    cand_names = all_video_ids
    cand_text = [process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True) for _ in cand_names]
    cand_image = []
    
    # 为每个候选视频提取帧 (使用 ImageVideoInstance 避免内存爆炸)
    for vid in cand_names:
        video_path = os.path.join(video_root, f"{vid}.mp4")
        frame_dir = os.path.join(frame_root, vid)
        
        # 提取视频帧
        save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        
        # 使用 ImageVideoInstance 包装路径，避免加载到内存
        cand_image.append(ImageVideoInstance(
            bytes=[None] * len(video_frame_paths),
            paths=video_frame_paths,
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
        ).to_dict())

    # 提前建 dict 加速查找
    vid2idx = {vid: i for i, vid in enumerate(cand_names)}

    query_texts, query_images, query_audios = [], [], []
    dataset_infos = []

    # 为每个音频查询创建数据
    for category, video_id, clip_id, audio_obj in zip(
        batch_dict["category"],
        batch_dict["video_id"],
        batch_dict["clip_id"],
        batch_dict["query_audio"],
    ):
        # 音频查询（已解码）
        query_audios.append(audio_obj)
        query_text = build_query_text("AVE")
        query_texts.append(query_text)
        assert isinstance(query_text, list) and len(query_text) == 1 and isinstance(query_text[0], str) and query_text[0].strip()
        query_images.append([None])

        # 找到正确答案在候选池中的索引
        label_idx = vid2idx[video_id]

        info = {
            "label_cand_id": label_idx,
            "label_name": video_id,
            "cand_names": cand_names,
            "query_id": clip_id,
            "corpus_id": video_id,
            "category": category,
        }
        dataset_infos.append(info)

    # 4. 在 data_prepare 最后返回的 batch 字段里，明确返回 cand_*（不要依赖旧字段）
    bs = len(query_texts)
    return {
        "query_text": query_texts,
        "query_image": query_images,
        "query_audio": query_audios,

        "cand_text": [cand_text] * bs,
        "cand_video": [cand_image] * bs,  # 改为 cand_video，因为候选项是视频帧
        "cand_audio": [[]] * bs,

        "dataset_infos": dataset_infos,
    }


DATASET_PARSER_NAME = "ave_retrieval"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_ave_retrieval_dataset(model_args, data_args, **kwargs):
    """
    AVE 音频->视频检索数据集加载器。
    每个音频片段查询对应的完整视频。

    参数:
        data_path: AVE 数据根目录
        split_file: 分割文件，默认 testSet.txt
        audio_root: 音频文件根目录，默认 {data_path}/audios
        video_root: 视频文件根目录，默认 {data_path}/AVE
    """
    data_path_input = kwargs.get(
        "data_path", os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "AVE", "AVE_Dataset")
    )
    data_path = _resolve_existing_dir(
        _case_variants(data_path_input)
        + [os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "ave", "AVE", "AVE_Dataset")]
    )
    if not data_path:
        raise AssertionError(
            f"AVE data_path not found. tried={_case_variants(data_path_input) + [os.path.join(BASE_RAW_DATA_DIR, 'audio-tasks', 'ave', 'AVE', 'AVE_Dataset')]}"
        )

    split_file = kwargs.get("split_file", "testSet.txt")
    video_root_input = kwargs.get("video_root", os.path.join(data_path, "AVE"))
    video_root = _resolve_existing_dir(_case_variants(video_root_input) + [os.path.join(data_path, "AVE")])

    # Prefer configured audio_root if valid. If missing/empty, we will extract clips from mp4.
    audio_root_input = kwargs.get("audio_root", os.path.join(data_path, "audios"))
    resolved_audio_root = _resolve_existing_dir(_case_variants(audio_root_input))
    use_precomputed_audio = bool(resolved_audio_root)
    audio_root = resolved_audio_root if resolved_audio_root else audio_root_input
    if not audio_root:
        audio_root = os.path.join(data_path, "audios_extracted_16k")
    os.makedirs(audio_root, exist_ok=True)

    split_path = os.path.join(data_path, split_file)
    assert os.path.isfile(split_path), f"AVE split file not found: {split_path}"
    assert os.path.isdir(video_root), f"Video directory not found: {video_root}"

    samples = _parse_split_file(split_path)
    # 先过滤缺失视频（音频缺失时可走抽取）
    video_filtered = []
    for s in samples:
        video_path = os.path.join(video_root, f"{s['video_id']}.mp4")
        if os.path.isfile(video_path):
            video_filtered.append(s)
    samples = video_filtered

    # 采样（放在音频处理前，可显著减少抽取开销）
    max_samples = kwargs.get("max_samples", None)
    seed = kwargs.get("seed", 17)
    if max_samples is not None and max_samples < len(samples):
        import random
        random.seed(seed)
        samples = random.sample(samples, max_samples)

    filtered = []
    missing_audio = 0
    extract_failed = 0
    for s in samples:
        video_path = os.path.join(video_root, f"{s['video_id']}.mp4")
        audio_path = os.path.join(audio_root, _audio_filename(s["video_id"], s["clip_id"]))
        if use_precomputed_audio:
            if not os.path.isfile(audio_path):
                missing_audio += 1
                continue
        else:
            ok = _extract_clip_audio_ffmpeg(video_path, s["start"], s["end"], audio_path)
            if not ok:
                extract_failed += 1
                continue
        s = dict(s)
        s["audio_path"] = audio_path
        filtered.append(s)
    samples = filtered
    assert len(samples) > 0, (
        f"AVE has no valid samples after filtering. "
        f"use_precomputed_audio={use_precomputed_audio}, missing_audio={missing_audio}, extract_failed={extract_failed}, "
        f"video_root={video_root}, audio_root={audio_root}"
    )

    # 构造 HF Dataset
    # 构造音频文件路径（对齐 samples）
    audio_files = [
        s["audio_path"]
        for s in samples
    ]

    # 一次性建 Dataset，把 audio 直接作为一列塞进去（最稳）
    dataset = datasets.Dataset.from_dict(
        {
            "category": [s["category"] for s in samples],
            "video_id": [s["video_id"] for s in samples],
            "clip_id": [s["clip_id"] for s in samples],
            # ✅ HF Audio column expects dicts: {"path":..., "bytes": None}
            "query_audio": [{"path": p, "bytes": None} for p in audio_files],
        }
    )

    # ✅ 让 datasets 自动解码音频，query_audio 会变成 {"array","sampling_rate"} 形式
    dataset = dataset.cast_column("query_audio", Audio())

    # 1A. 从 dataset 的 video_id 列去重得到候选（最稳、最少外部依赖）
    all_video_ids = sorted(list(set(dataset["video_id"])))

    # 传递路径参数和视频处理参数
    kwargs["audio_root"] = audio_root
    kwargs["video_root"] = video_root
    kwargs["frame_root"] = kwargs.get("frame_root", os.path.join(data_path, "frames"))
    kwargs["num_frames"] = kwargs.get("num_frames", 8)
    kwargs["max_frames_saved"] = kwargs.get("max_frames_saved", 100)
    kwargs["image_resolution"] = data_args.image_resolution
    kwargs["model_backbone"] = model_args.model_backbone

    # 2. 用闭包把 all_video_ids 传进 data_prepare
    def _prepare(x):
        return data_prepare(x, all_video_ids=all_video_ids, **kwargs)

    # 处理数据集
    dataset = dataset.map(
        _prepare,
        batched=True,
        batch_size=128,
        drop_last_batch=False,
        load_from_cache_file=False,
    )

    dataset = dataset.select_columns(
        ["query_text", "query_image", "query_audio", "cand_text", "cand_video", "dataset_infos"]
    )

    corpus = None
    return dataset, corpus
