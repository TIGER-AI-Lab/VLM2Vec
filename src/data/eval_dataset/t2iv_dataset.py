import os
import sys

from src.utils.basic_utils import print_rank, print_master

from datasets import load_dataset, Dataset
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance
from src.model.processor import process_input_text
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames

FORMAT_MAPPING = {
    'I': 'jpg',
    'V': 'mp4', 
    'A': 'wav'
}

INST_MAPPING = {
    'T': "Find the text that best matches the given image and video.",
    'I': "Find the image that best matches the given text: ",
    'V': "Find the video that best matches the given text: ",
    'A': "Find the audio that best matches the given text: "
}

def coco_filename(id):
    return f'COCO_val2014_{str(id).zfill(12)}'

def coco_filename_with_ext(id, modality='I'):
    ext = FORMAT_MAPPING[modality] # default to jpg if modality not found
    return f'{coco_filename(id)}.{ext}'

def coco_id(filename):
    return int(filename.split('_')[-1].split('.')[0])

def data_preprocess(dataset, *args, **kwargs):
    # this is to construct the positive samples for querying with different modalities
    # for each pos sample, we can query for both T, I, V
    img_ids, pos_filenames, qry_insts, qry_texts, tgt_texts, tgt_images = [], [], [], [], [], []
    for img_id, qry_text, neg_ids in zip(dataset['image_id'], dataset['qry_text'], dataset['hard_negatives']):
        # generate the candidate pool, which is the same for different query modalities
        tgt_txt_lst = []
        tgt_visual_lst = []
        for tgt_id in neg_ids+[img_id]: # add the positive sample into the candidate pool as well, ideally should have (len(neg_ids)+1)*2 candidates in total
            for modality in ['I', 'V']:
                tgt_txt_lst.append("") # no text input for the candidate for now, serving as a placeholder here
                tgt_visual_lst.append(coco_filename_with_ext(tgt_id, modality))

        # now generate the query instance (positive sample) for each modality
        for modality in ['I', 'V']:
            img_ids.append(img_id)
            pos_filename = coco_filename_with_ext(img_id, modality)
            pos_filenames.append(pos_filename)
            qry_insts.append(INST_MAPPING[modality]) # pos filename and qry insts are the only two that are different
            qry_texts.append(qry_text)
            tgt_texts.append(tgt_txt_lst)
            tgt_images.append(tgt_visual_lst)

    return Dataset.from_dict({
        "image_id": img_ids,
        "pos_filename": pos_filenames,
        "qry_inst": qry_insts,
        "qry_text": qry_texts,
        "tgt_text": tgt_texts,
        "tgt_image": tgt_images,
    })


# @add_metainfo_hook
# def data_prepare(batch_dict, *args, **kwargs):
#     image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
#     num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
#     image_root, video_root, frame_root = kwargs['image_root'], kwargs['video_root'], kwargs['frame_root']
#     dataset_name = kwargs['dataset_name']
#     model_backbone = kwargs['model_backbone']

#     TGT_INST = "Represent the given text, image or video."
#     query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
#     for pos_id, qry_text, negatives in zip(batch_dict['image_id'], batch_dict['qry_text'], batch_dict['hard_negatives']):
#         if not os.path.exists(os.path.join(video_root, coco_filename(pos_id)+'.mp4')):
#             print_rank(f"Video {coco_filename(pos_id)} does not exist, skipping this data point.")
#             continue
#         # we first load the candidate videos and images
#         cand_id_lst = negatives+[pos_id] # this is the list of file ids, we will load both videos and images into our candidate pool

#         # ====================== HARD CODING HERE =========================
#         # TO AVOID THE MULTI PROCESSING ISSUE OF MAP, which requires the same # of candidates of each query in a batch
#         if len(cand_id_lst) < 20:
#             # make sure all queries have the same number of cands, use 20 here
#             all_vid_ids = [coco_id(f) for f in os.listdir(video_root) if f.endswith('.mp4')]
#             unincluded_vid_ids = list(set(all_vid_ids) - set(cand_id_lst))
#             cand_id_lst += unincluded_vid_ids[:20-len(cand_id_lst)]
#         # ======================== HARD CODING ENDS ========================

#         cand_name_lst = []
#         cand_txt_lst = [] # corresponding tgt texts for the candidate videos and images
#         cand_visual_lst = [] # this is the list of processed candidate videos and images' paths
#         for cand_id in cand_id_lst:
#             video_filename = coco_filename(cand_id)+'.mp4'
#             image_filename = coco_filename(cand_id)+'.jpg'
#             cand_video_path = os.path.join(video_root, video_filename)
#             cand_image_path = os.path.join(image_root, image_filename)
#             if cand_id == pos_id:
#                 # assert os.path.exists(cand_video_path), f"Video {cand_video_path} does not exist."
#                 assert os.path.exists(cand_image_path), f"Image {cand_image_path} does not exist."
#                 # this is the positive video and image, we will add the text query later

#             # load candidate video frames
#             frame_dir = os.path.join(frame_root, video_filename)
#             try:
#                 save_frames(video_path=cand_video_path,
#                             frame_dir=frame_dir,
#                             max_frames_saved=max_frames_saved)
#                 video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
#                 cand_visual_lst.append(ImageVideoInstance(
#                     bytes=[None] * len(video_frame_paths),
#                     paths=video_frame_paths,
#                     resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
#                 ).to_dict())
#                 cand_txt_lst.append(process_input_text(TGT_INST, text="", model_backbone=model_backbone, add_video_token=True))
#                 cand_name_lst.append(video_filename)
#             except:
#                 print(f"Loading frames for {cand_video_path} failed.")

#             cand_visual_lst.append(ImageVideoInstance(
#                 bytes=[None] * len([cand_image_path]),
#                 paths=[cand_image_path],
#                 resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len([cand_image_path]),
#             ).to_dict())
#             cand_txt_lst.append(process_input_text(TGT_INST, text="", model_backbone=model_backbone, add_image_token=True))
#             cand_name_lst.append(image_filename)

#         # we will query for two separate instances with the same candidate pool: one with text query asking for image modality, the other with text query asking for video modality
#         pos_img_filename = coco_filename(pos_id) + '.jpg'
#         query_texts.append([process_input_text(
#             "Find the image that best matches the given text: ", 
#             text=qry_text, model_backbone=model_backbone, add_image_token=True)])
#         query_images.append([None]) # no visual input for the query, or it will be too simple
#         cand_texts.append(cand_txt_lst) 
#         cand_images.append(cand_visual_lst)
#         dataset_infos.append({
#             "cand_names": cand_name_lst,
#             "label_name": pos_img_filename
#         })

#         # now query for video modality
#         pos_vid_filename = coco_filename(pos_id) + '.mp4'
#         query_texts.append([process_input_text(
#             "Find the video that best matches the given text: ", 
#             text=qry_text, model_backbone=model_backbone, add_video_token=True)])
#         query_images.append([None]) # no visual input for the query, or it will be too simple
#         cand_texts.append(cand_txt_lst)
#         cand_images.append(cand_visual_lst)
#         dataset_infos.append({
#             "cand_names": cand_name_lst,
#             "label_name": pos_vid_filename,
#         })

#         # query for audio modality can be added in the future when the dataset provides audio files

#     print_master(len(query_texts))
#     print_master(len(query_images))
#     print_master(len(cand_texts))
#     print_master(len(cand_images))
#     print_master(len(dataset_infos))
#     assert len(query_texts) == len(query_images) == len(cand_texts) == len(cand_images) == len(dataset_infos), f"Length mismatch: {len(query_texts)}, {len(query_images)}, {len(cand_texts)}, {len(cand_images)}, {len(dataset_infos)}"
#     print_master(f"Finished preparing {len(query_texts)} query instances for dataset {dataset_name} with model backbone {model_backbone} and image resolution {image_resolution}. Each query instance has a candidate pool of {len(cand_texts[0])} videos and images.") # we assume each query has the same number of candidates, which should be the case for this dataset
#     return {
#         "query_text": query_texts, "query_image": query_images, 
#         "cand_text": cand_texts, "cand_image": cand_images,
#         "dataset_infos": dataset_infos
#     }

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    image_root, video_root, frame_root = kwargs['image_root'], kwargs['video_root'], kwargs['frame_root']
    dataset_name = kwargs['dataset_name']
    model_backbone = kwargs['model_backbone']

    TGT_INST = "Represent the given text, image or video."
    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for pos_id, pos_filename, qry_inst, qry_text, tgt_text_lst, tgt_visual_lst in zip(batch_dict['image_id'], batch_dict['pos_filename'], batch_dict['qry_inst'], batch_dict['qry_text'], batch_dict['tgt_text'], batch_dict['tgt_image']):
        query_text = process_input_text(qry_inst, text=qry_text, model_backbone=model_backbone, add_image_token=True)
        query_texts.append([query_text])
        query_images.append([None]) # no visual input for the query, or it will be too simple

        cand_name_instances = []
        cand_txt_instances = []
        cand_visual_instances = []
        for cand_text, cand_filename in zip(tgt_text_lst, tgt_visual_lst): # cand_text should be empty, but we still load it to make the code elegant and flexible for future datasets that may have non-empty candidate texts
            if cand_filename.endswith('.jpg'): # if this is image
                cand_img_path = os.path.join(image_root, cand_filename)
                assert os.path.exists(cand_img_path), f"Image {cand_img_path} does not exist."
                cand_visual_instances.append(ImageVideoInstance(
                    bytes=[None],
                    paths=[cand_img_path],
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
                ).to_dict())
                cand_txt_instances.append(process_input_text(TGT_INST, text=cand_text, model_backbone=model_backbone, add_image_token=True))
                cand_name_instances.append(cand_filename)
            elif cand_filename.endswith('.mp4'): # if is video
                cand_video_path = os.path.join(video_root, cand_filename)
                # assert os.path.exists(cand_vid_path), f"Video {cand_vid_path} does not exist."
                frame_dir = os.path.join(frame_root, cand_filename.split('.')[0]) # use the filename without extension as the frame dir name
                try:
                    save_frames(video_path=cand_video_path,
                                frame_dir=frame_dir,
                                max_frames_saved=max_frames_saved)
                    video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
                    cand_visual_instances.append(ImageVideoInstance(
                        bytes=[None] * len(video_frame_paths),
                        paths=video_frame_paths,
                        resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
                    ).to_dict())
                    cand_txt_instances.append(process_input_text(TGT_INST, text=cand_text, model_backbone=model_backbone, add_video_token=True))
                    cand_name_instances.append(cand_filename)
                except:
                    print_rank(f"Loading frames for {cand_video_path} failed.")
                    continue
            elif cand_filename.endswith('.wav'): # if is audio, to be implemented in the future
                raise NotImplementedError("Audio modality is not supported yet.")
            else:
                raise ValueError(f"Unsupported file format for candidate visual: {cand_filename}")
            
        assert pos_filename in cand_name_instances, f"Positive sample {pos_filename} not found in candidate pool {cand_name_instances} for query instance with image id {pos_id} and query text {qry_text}. This should not happen as we have added the positive sample into the candidate pool."
        cand_images.append(cand_visual_instances)
        cand_texts.append(cand_txt_instances)
        dataset_infos.append({
            "cand_names": cand_name_instances,
            "label_name": pos_filename,
        })

    return {
        "query_text": query_texts, "query_image": query_images, 
        "cand_text": cand_texts, "cand_image": cand_images,
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "t2iv"
DATASET_HF_PATH = "MINGYISU/t2iv"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_t2iv_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]

    dataset = load_dataset(DATASET_HF_PATH, split="test")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    # print_master(f"Start preparing dataset {dataset_name} with model backbone {model_args.model_backbone} and image resolution {data_args.image_resolution}. Total number of samples: {len(dataset)}.")
    dataset = data_preprocess(dataset, **kwargs)
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, None
