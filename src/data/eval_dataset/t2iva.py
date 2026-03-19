import os
import sys

from src.utils.basic_utils import print_rank, print_master
from datasets import load_dataset, Dataset
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance, MODALITY_INST_MAPPING, coco_filename_with_ext
from src.model.processor import process_input_text
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames

def generate_omnidirectional_dataset(dataset, *args, **kwargs):
    """
    Generate an omnidirectional evaluation dataset. Supported modalities: image, video, audio
    """
    AVAILABLE_MODALITIES=['I', 'V', 'A']
    img_ids, pos_filenames, qry_insts, qry_texts, tgt_texts, tgt_images, tgt_audios = [], [], [], [], [], [], []
    for img_id, qry_text, neg_ids in zip(dataset['image_id'], dataset['qry_text'], dataset['hard_negatives']):
        # generate the candidate pool, which is the same for different query modalities
        tgt_txt_lst, tgt_visual_lst, tgt_audio_lst = [], [], []
        # due to some inconsistency issues in the dataset where the pos id is sometimes in neg id pool, we take robust step to ensure the pos exists and is unique in the pool
        candidate_ids_pool = list(set(neg_ids+[img_id])) 
        for tgt_id in candidate_ids_pool: # add the positive sample into the candidate pool as well, ideally should have (len(neg_ids)+1)*3 candidates in total
            for modality in AVAILABLE_MODALITIES:
                if modality in ['I', 'V']:
                    tgt_txt_lst.append("") # no text input for the visual candidate for now, serving as a placeholder here
                    tgt_visual_lst.append(coco_filename_with_ext(tgt_id, modality))
                    tgt_audio_lst.append(None) # no audio input for the candidate for now
                elif modality == 'A':
                    tgt_txt_lst.append("")
                    tgt_visual_lst.append(None) # no visual input for the audio candidate
                    tgt_audio_lst.append(coco_filename_with_ext(tgt_id, modality))

        assert len(tgt_txt_lst) == len(tgt_visual_lst) == len(tgt_audio_lst), \
            f"Error: Inconsistent candidate pool lengths: {len(tgt_txt_lst)} text candidates, {len(tgt_visual_lst)} visual candidates, {len(tgt_audio_lst)} audio candidates."
        # now generate the query instance (positive sample) for each modality
        for modality in AVAILABLE_MODALITIES:
            img_ids.append(img_id)
            pos_filename = coco_filename_with_ext(img_id, modality)
            pos_filenames.append(pos_filename)
            qry_insts.append(MODALITY_INST_MAPPING[modality]) # pos filename and qry insts are the only two that are different
            qry_texts.append(qry_text)
            tgt_texts.append(tgt_txt_lst)
            tgt_images.append(tgt_visual_lst)
            tgt_audios.append(tgt_audio_lst)

    return Dataset.from_dict({
        "image_id": img_ids,
        "pos_filename": pos_filenames,
        "qry_inst": qry_insts,
        "qry_text": qry_texts,
        "tgt_text": tgt_texts,
        "tgt_image": tgt_images,
        "tgt_audio": tgt_audios
    })

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    image_root, video_root, frame_root = kwargs['image_root'], kwargs['video_root'], kwargs['frame_root']
    dataset_name = kwargs['dataset_name']
    model_backbone = kwargs['model_backbone']

    TGT_INST = "Represent the given text, image or video."
    query_texts, query_images, cand_texts, cand_images, cand_videos, cand_audios, dataset_infos = [], [], [], [], [], [], []
    for pos_id, pos_filename, qry_inst, qry_text, tgt_text_lst, tgt_visual_lst, tgt_audio_lst in zip(batch_dict['image_id'], batch_dict['pos_filename'], batch_dict['qry_inst'], batch_dict['qry_text'], batch_dict['tgt_text'], batch_dict['tgt_image'], batch_dict['tgt_audio']):
        query_texts.append([process_input_text(qry_inst, 
                                               text=qry_text, 
                                               model_backbone=model_backbone)])
        query_images.append([None]) # no visual input for the query, or it will be too simple

        cand_name_insts, cand_txt_insts, cand_img_insts, cand_vid_insts, cand_aud_insts = [], [], [], [], [] # candidate pool for each single query sample in a batch
        for cand_text, cand_visual_filename, cand_audio_filename in zip(tgt_text_lst, tgt_visual_lst, tgt_audio_lst):
            if cand_audio_filename is not None: # if this is audio candidate
                cand_audio_path = os.path.join(video_root, cand_audio_filename) # audio files are stored in the same dir as videos
                assert os.path.exists(cand_audio_path), f"Audio {cand_audio_path} does not exist."
                cand_aud_insts.append({"path": cand_audio_path, "bytes": None})
                cand_txt_insts.append(f"<|audio_pad|> {TGT_INST}") # temp fix, will update process_input_text function to handle this
                cand_name_insts.append(cand_audio_filename)
                assert cand_visual_filename is None, f"Error: audio candidate {cand_audio_filename} should not have visual candidate, but got {cand_visual_filename}"
                cand_vid_insts.append(None) # no visual input for the audio candidate
                cand_img_insts.append(None) # no visual input for the audio candidate
            if cand_visual_filename is not None:
                if cand_visual_filename.endswith('.jpg'): # if this is image
                    cand_img_path = os.path.join(image_root, cand_visual_filename)
                    assert os.path.exists(cand_img_path), f"Image {cand_img_path} does not exist."
                    cand_img_insts.append(ImageVideoInstance(
                        bytes=[None],
                        paths=[cand_img_path],
                        resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
                    ).to_dict())
                    cand_txt_insts.append(process_input_text(TGT_INST, text=cand_text, model_backbone=model_backbone, add_image_token=True))
                    cand_name_insts.append(cand_visual_filename)
                    cand_vid_insts.append(None) # no video input for the image candidate
                    cand_aud_insts.append(None) # no audio input for the visual candidate
                elif cand_visual_filename.endswith('.mp4'): # if is video
                    cand_video_path = os.path.join(video_root, cand_visual_filename)
                    assert os.path.exists(cand_video_path), \
                        f"Video {cand_video_path} does not exist."
                    frame_dir = os.path.join(frame_root, cand_visual_filename.split('.')[0]) # use the filename without extension as the frame dir name
                    try:
                        save_frames(video_path=cand_video_path,
                                    frame_dir=frame_dir,
                                    max_frames_saved=max_frames_saved)
                        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
                        cand_vid_insts.append(ImageVideoInstance(
                            bytes=[None] * len(video_frame_paths),
                            paths=video_frame_paths,
                            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
                        ).to_dict())
                        cand_txt_insts.append(process_input_text(TGT_INST, text=cand_text, model_backbone=model_backbone, add_video_token=True))
                        cand_name_insts.append(cand_visual_filename)
                        cand_img_insts.append(None) # no image input for the video candidate
                        cand_aud_insts.append(None) # no audio input for the visual candidate
                    except: # simply skip the cand video if not exist, however, if the pos sample is a video, raise error
                        print_rank(f"Loading frames for {cand_video_path} failed.")
                        if cand_visual_filename == pos_filename:
                            raise FileNotFoundError(f"Positive sample video {cand_video_path} not found or failed to load!")
                        else:
                            print_rank(f"Skipping candidate video {cand_video_path}.")
                            continue
                else:
                    raise ValueError(f"Unsupported file format for candidate visual: {cand_visual_filename}")
            
        assert pos_filename in cand_name_insts and len(cand_name_insts) == len(set(cand_name_insts)), \
            f'Error: pos_filename={pos_filename} NOT FOUND in cand_name_instances={cand_name_insts}'
        cand_images.append(cand_img_insts)
        cand_videos.append(cand_vid_insts)
        cand_texts.append(cand_txt_insts)
        cand_audios.append(cand_aud_insts)
        dataset_infos.append({
            "cand_names": cand_name_insts,
            "label_name": pos_filename,
        })

    return {
        "query_text": query_texts, "query_image": query_images, "query_audio": [None]*len(query_texts),
        "cand_text": cand_texts, "cand_image": cand_images, "cand_video": cand_videos, "cand_audio": cand_audios,
        "dataset_infos": dataset_infos
    }

DATASET_PARSER_NAME = "t2iva"
DATASET_HF_PATH = "MINGYISU/t2iv" # can still use t2iv, will rename the dataset in the future
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_t2iva_dataset(model_args, data_args, *args, **kwargs):
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

    # print_master(f"Start preparing dataset {dataset_name} with model backbone {model_args.model_backbone} and image resolution {data_args.image_resolution}. Total number of samples: {len(dataset)}.")
    dataset = generate_omnidirectional_dataset(dataset, *args, **kwargs)
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "query_audio", "cand_text", "cand_image", "cand_video", "cand_audio", "dataset_infos"])

    return dataset, None
