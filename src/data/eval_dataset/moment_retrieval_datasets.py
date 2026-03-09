import os

from datasets import load_dataset
from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames, VID_EXTENSIONS
from src.model.processor import process_input_text
from src.utils.basic_utils import print_master

TASK_INST_QRY = "Find the clip that corresponds to the described scene in the given video:"
TASK_INST_TGT = "Understand the content of the provided video."

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution = kwargs['image_resolution']
    max_video_frames_saved = kwargs["max_video_frames_saved"]
    max_clip_frames_saved = kwargs["max_clip_frames_saved"]
    num_video_frames = kwargs["num_video_frames"]
    num_clip_frames = kwargs["num_clip_frames"]
    model_backbone = kwargs["model_backbone"]
    video_root, clip_root, frame_root = kwargs["video_root"], kwargs["clip_root"], kwargs["frame_root"]

    query_texts, query_images, cand_texts, cand_clip_images, dataset_infos = [], [], [], [], []

    for query, query_video_path in zip(batch_dict['query'], batch_dict['video_path']):

        # 允许传入绝对路径；若提供 video_root 则使用拼接的路径，否则直接用原始字段
        query_video_path = os.path.join(video_root, os.path.basename(query_video_path)) if video_root else query_video_path
        if query_video_path is None:
            raise ValueError("moment_retrieval: query_video_path is None; 请检查数据或 video_root。")
        video_name = os.path.splitext(os.path.basename(query_video_path))[0]

        # 仅使用预提取帧，不再尝试从视频抽帧
        frames_dir = os.path.join(frame_root, video_name)
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames dir not found: {frames_dir}. 请确认已预提取帧。")

        query_frame_dir = os.path.join(frames_dir, "query")
        if not os.path.exists(query_frame_dir):
            raise FileNotFoundError(f"Query frames not found: {query_frame_dir}.")
        qry_frame_paths = process_video_frames(query_frame_dir, num_frames=num_video_frames)
        if len(qry_frame_paths) == 0:
            raise FileNotFoundError(f"Query frames empty: {query_frame_dir}.")

        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=query, add_video_token=True)])
        query_images.append([ImageVideoInstance(
            bytes=[None] * len(qry_frame_paths),
            paths=qry_frame_paths,
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_frame_paths),
        ).to_dict()])

        # Load pos and neg clip：只使用已有帧
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

        cand_clip_names, cand_frames = [], []
        positive_clip_names = []
        for clip_frame_dir_or_file in os.listdir(frames_dir):
            clip_frame_dir_abs = os.path.join(frames_dir, clip_frame_dir_or_file)
            if clip_frame_dir_or_file == 'query' or os.path.isfile(clip_frame_dir_abs):
                continue
            if clip_frame_dir_or_file.startswith("positive"):
                positive_clip_names.append(clip_frame_dir_abs)
            cand_frame_paths = process_video_frames(clip_frame_dir_abs, num_frames=num_clip_frames)
            if len(cand_frame_paths) == 0:
                raise FileNotFoundError(f"Clip frames empty: {clip_frame_dir_abs}")
            cand_frames.append(ImageVideoInstance(
                bytes=[None] * len(cand_frame_paths),
                paths=cand_frame_paths,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(cand_frame_paths),
            ).to_dict())
            cand_clip_names.append(clip_frame_dir_abs)  # use absolute path here instead of file name to keep it unique
        if len(cand_clip_names) == 0:
            raise FileNotFoundError(f"No candidate clips found under {frames_dir}")
        if len(positive_clip_names) == 0:
            raise FileNotFoundError(f"No positive clip found under {frames_dir}")
        if len(positive_clip_names) > 1:
            print_master(f"Warning: multiple positive clips found under {frames_dir}: {positive_clip_names}")
        assert all(p in cand_clip_names for p in positive_clip_names), "positive clips not included in candidates"
        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)] * len(cand_clip_names))
        cand_clip_images.append(cand_frames)
        dataset_infos.append({
            "cand_names": cand_clip_names,
            "label_name": positive_clip_names if len(positive_clip_names) > 1 else positive_clip_names[0],
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_clip_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "moment_retrieval"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_moment_retrieval_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs.get('dataset_name')
    
    if kwargs.get("data_path", None) != None:
        dataset = load_dataset("json", data_files=kwargs["data_path"])
        dataset = dataset["train"]
    else:
        # 优先使用本地路径
        if dataset_name in EVAL_DATASET_LOCAL_PATH:
            local_path_info = EVAL_DATASET_LOCAL_PATH[dataset_name]
            local_path = local_path_info[0]
            if os.path.exists(local_path):
                print(f"Loading {dataset_name} from local path: {local_path}")
                dataset = load_hf_dataset((local_path, local_path_info[1], local_path_info[2], "local"))
            else:
                print(f"Local path {local_path} not found, falling back to HuggingFace Hub")
                dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        else:
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
    
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=2048, num_proc=8,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
    corpus = None  # No additional corpus

    return dataset, corpus
