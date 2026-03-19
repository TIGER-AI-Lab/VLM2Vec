import os
import sys

from datasets import load_dataset
from src.utils.basic_utils import print_rank, print_master
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance
from src.model.processor import process_input_text
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    image_root, video_root, frame_root = kwargs['image_root'], kwargs['video_root'], kwargs['frame_root']
    dataset_name = kwargs['dataset_name']
    model_backbone = kwargs['model_backbone']

    QRY_INST = "Find the video that best matches the given image and text."
    TGT_INST = "Represent the given videos."
    tgt_text = process_input_text(TGT_INST, text="", model_backbone=model_backbone, add_video_token=True) # no individual different tgt texts for different cand videos

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    # tgt_text could be empty for some datasets
    for id, qry_text, qry_imgs_paths, neg_videos in (
            zip(batch_dict['id'], batch_dict['qry_text'], batch_dict['qry_image_path'], batch_dict['negatives'])):
        qry_text = process_input_text(QRY_INST, text=qry_text, model_backbone=model_backbone, add_image_token=True)
        query_texts.append([qry_text])
        qry_img_list = [os.path.join(image_root, os.path.basename(qry_img_path))
                        for qry_img_path in (qry_imgs_paths 
                                             if isinstance(qry_imgs_paths, list) # if is a list of paths
                                             else [qry_imgs_paths])] # if is a single path str, convert to list
        query_images.append([ImageVideoInstance(
            bytes=[None] * len(qry_img_list),
            paths=qry_img_list,
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_img_list)
        ).to_dict()])

        # loading candidate video frames
        cand_video_id_lst = [id]+neg_videos # this is the list of video file names
        cand_videos_lst = [] # this is the list of processed video frames' paths
        cand_txt_lst = [] # corresponding texts
        for video_name in cand_video_id_lst: # has to be a list
            neg_video_path = os.path.join(video_root, f'{video_name}.mp4')
            frame_dir = os.path.join(frame_root, video_name)
            try:
                save_frames(video_path=neg_video_path,
                            frame_dir=frame_dir,
                            max_frames_saved=max_frames_saved)
                video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
            except:
                print(f"Skipping {neg_video_path}")
                continue

            cand_videos_lst.append(ImageVideoInstance(
                bytes=[None] * len(video_frame_paths),
                paths=video_frame_paths,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
            ).to_dict())
            cand_txt_lst.append(tgt_text)
        cand_images.append(cand_videos_lst)
        cand_texts.append(cand_txt_lst)
        dataset_infos.append({
            "cand_names": cand_video_id_lst,
            "label_name": id,
        })

    return {
        "query_text": query_texts, "query_image": query_images, 
        "cand_text": cand_texts, "cand_image": cand_images,
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "ti2v"
DATASET_HF_PATH = "MINGYISU/ti2v"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_ti2v_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]

    dataset = load_dataset(DATASET_HF_PATH, split="test")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print_master(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, None
