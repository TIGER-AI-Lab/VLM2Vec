"""
SpeechCOCO 音频->图像检索训练数据集处理（适配 train_collator_omni）。
- 数据源：本地 parquet 分片（MMEB-V3-train speechcoco）
- 训练样本形式：
    query_text  = build_query_text("SpeechCOCO")
    query_audio = {"path": ..., "bytes": ..., "start": None, "end": None}
    pos_text    = process_input_text(..., add_image_token=True)
    pos_image   = {"paths": [...], "bytes": [...], "resolutions": [(w,h)]}
"""

import glob
import os
from typing import Any, Dict, List

import datasets
import pyarrow as pa

from src.data.dataset.base_pair_dataset import AutoPairDataset, RESOLUTION_MAPPING
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.model.processor import process_input_text
from src.utils.basic_utils import print_master
from src.utils.dataset_utils import sample_dataset


POS_TEXT_IMAGE_INST = "Understand the content of the provided image."
DEFAULT_DATA_PATH = (
    "/data/mengrui/.cache/huggingface/datasets/MMEB-V3-train/audio-tasks-train/speechcoco"
)


def _resolve_parquet_files(data_path: str, data_subdir: str, parquet_pattern: str) -> List[str]:
    parquet_dir = os.path.join(data_path, data_subdir)
    files = sorted(glob.glob(os.path.join(parquet_dir, parquet_pattern)))
    if not files:
        raise FileNotFoundError(
            f"[SpeechCOCO] no parquet files found under {parquet_dir} with pattern={parquet_pattern}"
        )
    return files


def _load_parquet_dataset(parquet_files: List[str], cache_dir: str = None) -> datasets.Dataset:
    kwargs = {
        "path": "parquet",
        "data_files": {"train": parquet_files},
        "split": "train",
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        return datasets.load_dataset(**kwargs)
    except PermissionError:
        fallback_cache = os.path.join(os.path.dirname(parquet_files[0]), ".hf_datasets_cache")
        os.makedirs(fallback_cache, exist_ok=True)
        print_master(f"[SpeechCOCO] fallback cache_dir={fallback_cache}")
        kwargs["cache_dir"] = fallback_cache
        return datasets.load_dataset(**kwargs)


def _is_valid_row(row: Dict[str, Any]) -> bool:
    audio = row.get("audio")
    if not isinstance(audio, dict):
        return False
    audio_path = audio.get("path")
    audio_bytes = audio.get("bytes")
    if audio_bytes is not None:
        if isinstance(audio_bytes, (bytes, bytearray)):
            audio_ok = len(audio_bytes) > 0
        else:
            audio_ok = True
    else:
        audio_ok = isinstance(audio_path, str) and len(audio_path.strip()) > 0

    image = row.get("image")
    if not isinstance(image, dict):
        return False
    image_path = image.get("path")
    image_bytes = image.get("bytes")
    if image_bytes is not None:
        if isinstance(image_bytes, (bytes, bytearray)):
            image_ok = len(image_bytes) > 0
        else:
            image_ok = True
    else:
        image_ok = isinstance(image_path, str) and len(image_path.strip()) > 0

    return audio_ok and image_ok


def _build_audio_with_start_end(dataset: datasets.Dataset, column: str) -> datasets.Dataset:
    """
    将 struct<bytes,path> 规范成 struct<path,bytes,start,end>，避免 mixed_dataset.cast_column 失败。
    这里直接在 Arrow 层重建 struct 字段，复用原有 bytes/path 缓冲区，不复制大音频 payload。
    """
    if column not in dataset.column_names:
        raise KeyError(f"[SpeechCOCO] missing column: {column}")

    ds_table = dataset.data
    pa_table = ds_table.table if hasattr(ds_table, "table") else ds_table
    col_idx = pa_table.column_names.index(column)
    src_col = pa_table.column(column)

    out_chunks = []
    for chunk in src_col.chunks:
        if not pa.types.is_struct(chunk.type):
            raise TypeError(f"[SpeechCOCO] {column} must be a struct column, got={chunk.type}")
        field_names = set(chunk.type.names)

        path_arr = chunk.field("path") if "path" in field_names else pa.nulls(len(chunk), type=pa.string())
        bytes_arr = (
            chunk.field("bytes") if "bytes" in field_names else pa.nulls(len(chunk), type=pa.binary())
        )
        start_arr = (
            chunk.field("start")
            if "start" in field_names
            else pa.nulls(len(chunk), type=pa.float32())
        )
        end_arr = (
            chunk.field("end")
            if "end" in field_names
            else pa.nulls(len(chunk), type=pa.float32())
        )

        out_chunks.append(
            pa.StructArray.from_arrays(
                [path_arr, bytes_arr, start_arr, end_arr],
                names=["path", "bytes", "start", "end"],
            )
        )

    out_col = pa.chunked_array(out_chunks)
    out_table = pa_table.set_column(col_idx, column, out_col)
    return datasets.Dataset(out_table)


def _build_image_with_resolution(
    dataset: datasets.Dataset, column: str, resolution: List[int]
) -> datasets.Dataset:
    """
    将 struct<bytes,path> 规范成 struct<paths,bytes,resolutions>：
      - paths: List[str]，每行 1 个元素
      - bytes: List[bytes]，每行 1 个元素
      - resolutions: List[List[int,int]]，每行 1 个元素
    使用 Arrow 构造避免把整列图像 bytes 拉入 Python。
    """
    if column not in dataset.column_names:
        raise KeyError(f"[SpeechCOCO] missing column: {column}")

    ds_table = dataset.data
    pa_table = ds_table.table if hasattr(ds_table, "table") else ds_table
    col_idx = pa_table.column_names.index(column)
    src_col = pa_table.column(column)

    pair_type = pa.list_(pa.int32(), 2)
    resolution_pair = [int(resolution[0]), int(resolution[1])]
    out_chunks = []

    for chunk in src_col.chunks:
        if not pa.types.is_struct(chunk.type):
            raise TypeError(f"[SpeechCOCO] {column} must be a struct column, got={chunk.type}")

        n = len(chunk)
        field_names = set(chunk.type.names)

        path_arr = chunk.field("path") if "path" in field_names else pa.nulls(n, type=pa.string())
        bytes_arr = chunk.field("bytes") if "bytes" in field_names else pa.nulls(n, type=pa.binary())

        offsets = pa.array(range(n + 1), type=pa.int32())
        paths_list = pa.ListArray.from_arrays(offsets, path_arr, type=pa.list_(pa.string()))
        bytes_list = pa.ListArray.from_arrays(offsets, bytes_arr, type=pa.list_(pa.binary()))

        pair_values = pa.array([resolution_pair] * n, type=pair_type)
        resolutions_list = pa.ListArray.from_arrays(offsets, pair_values, type=pa.list_(pair_type))

        out_chunks.append(
            pa.StructArray.from_arrays(
                [paths_list, bytes_list, resolutions_list],
                names=["paths", "bytes", "resolutions"],
            )
        )

    out_col = pa.chunked_array(out_chunks)
    out_table = pa_table.set_column(col_idx, column, out_col)
    return datasets.Dataset(out_table)


@AutoPairDataset.register("load_audio_speechcoco_dataset")
def load_audio_speechcoco_dataset(*args: Any, **kwargs: Any):
    """
    加载 SpeechCOCO 训练数据（音频->图像检索），适配 train_collator_omni。

    参数（kwargs）：
        data_path: 数据根目录（默认 MMEB-V3-train speechcoco 本地路径）
        data_subdir: parquet 子目录，默认 "data"
        parquet_pattern: 分片匹配模式，默认 "train-*.parquet"
        cache_dir: 可选，HF datasets cache 目录

    返回：
        datasets.Dataset，字段：
            - query_text/query_image/query_audio
            - pos_text/pos_image/pos_audio
            - dataset_infos
    """
    path_info = kwargs.get("path_info")
    if path_info is not None:
        data_path = path_info[0]
    else:
        data_path = kwargs.get("data_path", DEFAULT_DATA_PATH)

    data_subdir = kwargs.get("data_subdir", "data")
    parquet_pattern = kwargs.get("parquet_pattern", "train-*.parquet")
    cache_dir = kwargs.get("cache_dir", None)

    parquet_files = _resolve_parquet_files(data_path, data_subdir, parquet_pattern)
    print_master(f"[SpeechCOCO] loading parquet shards: {len(parquet_files)} files")
    dataset = _load_parquet_dataset(parquet_files, cache_dir=cache_dir)
    print_master(f"[SpeechCOCO] raw rows: {len(dataset)}")

    keep_cols = [
        c
        for c in [
            "id",
            "image_id",
            "audio",
            "image",
            "text",
            "duration",
            "timecode",
            "speaker",
            "gender",
            "nationality",
            "speed",
            "disfluency_pos",
            "disfluency_val",
        ]
        if c in dataset.column_names
    ]
    dataset = dataset.select_columns(keep_cols)
    dataset = sample_dataset(dataset, **kwargs)
    dataset = dataset.filter(_is_valid_row)

    if len(dataset) == 0:
        raise ValueError("[SpeechCOCO] no valid rows after filtering")

    # query: audio
    dataset = dataset.rename_column("audio", "query_audio")
    dataset = _build_audio_with_start_end(dataset, "query_audio")

    # pos: image
    dataset = dataset.rename_column("image", "pos_image")
    image_resolution = kwargs.get("image_resolution", None)
    if image_resolution is None:
        data_args = kwargs.get("data_args", None)
        if data_args is not None:
            image_resolution = getattr(data_args, "image_resolution", None)
    resolution = RESOLUTION_MAPPING.get(image_resolution, RESOLUTION_MAPPING["low"])
    dataset = _build_image_with_resolution(dataset, "pos_image", resolution)

    # align with eval by default: instruction-only query text
    use_raw_text_in_query = bool(kwargs.get("use_raw_text_in_query", False))
    query_texts: List[List[str]] = []
    if use_raw_text_in_query:
        raw_texts = dataset["text"]
        for t in raw_texts:
            q = build_query_text("SpeechCOCO", t)
            assert isinstance(q, list) and len(q) == 1 and isinstance(q[0], str) and q[0].strip()
            query_texts.append(q)
    else:
        q = build_query_text("SpeechCOCO")
        assert isinstance(q, list) and len(q) == 1 and isinstance(q[0], str) and q[0].strip()
        query_texts = [list(q) for _ in range(len(dataset))]

    model_backbone = kwargs.get("model_backbone", None)
    if model_backbone is None:
        model_args = kwargs.get("model_args")
        if model_args is not None:
            model_backbone = getattr(model_args, "model_backbone", None)
    if model_backbone is None:
        raise ValueError("[SpeechCOCO] model_backbone is required")

    pos_text = process_input_text(POS_TEXT_IMAGE_INST, model_backbone, add_image_token=True)

    num_rows = len(dataset)
    ids = dataset["id"] if "id" in dataset.column_names else [None] * num_rows
    image_ids = dataset["image_id"] if "image_id" in dataset.column_names else [None] * num_rows
    texts = dataset["text"] if "text" in dataset.column_names else [None] * num_rows
    durations = dataset["duration"] if "duration" in dataset.column_names else [None] * num_rows
    timecodes = dataset["timecode"] if "timecode" in dataset.column_names else [None] * num_rows
    speakers = dataset["speaker"] if "speaker" in dataset.column_names else [None] * num_rows
    genders = dataset["gender"] if "gender" in dataset.column_names else [None] * num_rows
    nationalities = dataset["nationality"] if "nationality" in dataset.column_names else [None] * num_rows
    speeds = dataset["speed"] if "speed" in dataset.column_names else [None] * num_rows
    disfluency_pos = (
        dataset["disfluency_pos"] if "disfluency_pos" in dataset.column_names else [None] * num_rows
    )
    disfluency_val = (
        dataset["disfluency_val"] if "disfluency_val" in dataset.column_names else [None] * num_rows
    )

    dataset_infos: List[Dict[str, Any]] = []
    for i in range(num_rows):
        dataset_infos.append(
            {
                "id": ids[i],
                "image_id": image_ids[i],
                "text": texts[i],
                "duration": durations[i],
                "timecode": timecodes[i],
                "speaker": speakers[i],
                "gender": genders[i],
                "nationality": nationalities[i],
                "speed": speeds[i],
                "disfluency_pos": disfluency_pos[i],
                "disfluency_val": disfluency_val[i],
            }
        )

    dataset = dataset.add_column("query_text", query_texts)
    dataset = dataset.add_column("query_image", [None] * num_rows)
    dataset = dataset.add_column("pos_text", [pos_text] * num_rows)
    dataset = dataset.add_column("pos_audio", [None] * num_rows)
    dataset = dataset.add_column("dataset_infos", dataset_infos)

    dataset = dataset.select_columns(
        [
            "query_text",
            "query_image",
            "query_audio",
            "pos_text",
            "pos_image",
            "pos_audio",
            "dataset_infos",
        ]
    )

    try:
        setattr(dataset, "num_rows", len(dataset))
    except (AttributeError, TypeError):
        pass

    print_master(f"[SpeechCOCO] final rows: {len(dataset)}")
    return dataset
