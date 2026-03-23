import os
import sys

from src.utils.basic_utils import print_rank, print_master
from datasets import load_dataset, Dataset
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance, coco_filename_with_ext
from src.model.processor import process_input_text
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames

MODALITY_INST_MAPPING = {
    'T': "Find the text that best matches the given image: ",
    'V': "Find the video that best matches the given image: ",
    'A': "Find the audio that best matches the given image: "
}
SPECIAL_SEP_TOKEN = "|<<<FILENAME|CAPTION>>>|" # this is a strange enough token to separate the text filename and text content, since there is no actual text file, use like this f"{filename}.txt{SPECIAL_SEP_TOKEN}{caption}"

def generate_omnidirectional_dataset(dataset, *args, **kwargs):
    """
    Generate an omnidirectional evaluation dataset. 
    The query always has an image, with different instructions asking for three modalities of targets, e.g. Text, Video, and Audio. The candidate pool for each query instance includes all four modalities. The presence of Image candidates in the candidate pool serves as distractors making the task more challenging. No Image target (Positive) will be queried. 
    Example:
        Query: [Image of a dog] + "Find the Video that best matches this image. " -> 
        Target (Positive): [Video of a dog running]
        Candidate pool: [Video of a dog running, Image of a dog, Audio of a dog barking, Text description of a dog]
    """
    CAND_MODS=['T', 'I', 'V', 'A'] # possible candidate modalities, where T=text, I=image, V=video, A=audio
    img_ids, pos_filenames, qry_texts, qry_images, tgt_texts, tgt_images, tgt_videos, tgt_audios = [], [], [], [], [], [], [], []
    # Generating candidate pools, same for all three query instances based on the same image_id
    for img_id, caption, neg_ids in zip(dataset['image_id'], dataset['qry_text'], dataset['hard_negatives']):
        tgt_lsts = {k: [] for k in CAND_MODS} # store the candidates of each modality separately for easier processing later
        candidate_ids_pool = list(set(neg_ids+[img_id])) 
        # three nested for loops looks weird but actually saves lots of lines :)
        for tgt_id in candidate_ids_pool:
            for mod in CAND_MODS:
                tgt_lsts[mod].append(f"{coco_filename_with_ext(tgt_id, 'T')}{SPECIAL_SEP_TOKEN}{caption}" 
                                     if mod == 'T' else coco_filename_with_ext(tgt_id, mod))
                for other_mod in [k for k in CAND_MODS if k != mod]:
                    tgt_lsts[other_mod].append("" if other_mod == 'T' else None)

        assert len(tgt_lsts['T']) == len(tgt_lsts['I']) == len(tgt_lsts['V']) == len(tgt_lsts['A']), \
            f"Error: Inconsistent candidate pool lengths: {len(tgt_lsts['T'])} text candidates, {len(tgt_lsts['I'])} image candidates, {len(tgt_lsts['V'])} video candidates, {len(tgt_lsts['A'])} audio candidates."
        QRY_MODS = ['T', 'V', 'A'] # we only query the three modalities without image, which is the query modality
        # now generate the query instance (positive sample) for each modality
        for mod in QRY_MODS:
            img_ids.append(img_id)
            pos_filename = coco_filename_with_ext(img_id, mod) # caption text is not actually a file, but we still name it .txt to distinguish btw the other three modalities in the candidate pool
            pos_filenames.append(pos_filename)
            qry_texts.append(MODALITY_INST_MAPPING[mod]) # pos filename and qry insts are the only two that are different
            qry_images.append(coco_filename_with_ext(img_id, 'I')) # the query image is always the same
            tgt_texts.append(tgt_lsts['T'])
            tgt_images.append(tgt_lsts['I'])
            tgt_videos.append(tgt_lsts['V'])
            tgt_audios.append(tgt_lsts['A'])

    return Dataset.from_dict({
        "image_id": img_ids,
        "pos_filename": pos_filenames,
        "qry_text": qry_texts,
        "qry_image": qry_images,
        "tgt_text": tgt_texts,
        "tgt_image": tgt_images,
        "tgt_video": tgt_videos,
        "tgt_audio": tgt_audios
    })

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    image_root, video_root, audio_root, frame_root = kwargs['image_root'], kwargs['video_root'], kwargs['audio_root'], kwargs['frame_root']
    dataset_name = kwargs['dataset_name']
    model_backbone = kwargs['model_backbone']

    TGT_INST = "Represent the given text, image, video, or audio."
    query_texts, query_images, cand_texts, cand_images, cand_videos, cand_audios, dataset_infos = [], [], [], [], [], [], []
    for (pos_id, pos_filename, qry_txt, qry_img, 
         tgt_txts, tgt_imgs, tgt_vids, tgt_auds) in \
        zip(batch_dict['image_id'], batch_dict['pos_filename'], batch_dict['qry_text'], batch_dict['qry_image'], \
            batch_dict['tgt_text'], batch_dict['tgt_image'], batch_dict['tgt_video'], batch_dict['tgt_audio']):
        
        query_texts.append([qry_txt])
        query_images.append([ImageVideoInstance(
                        bytes=[None],
                        paths=[os.path.join(image_root, qry_img)],
                        resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
                    ).to_dict()])
        
        # processed candidate pool for each sample in a batch
        name_insts, txt_insts, img_insts, vid_insts, aud_insts = [], [], [], [], [] 
        # compose paths for the files of the candidate pool
        for txt, img, vid, aud in zip(tgt_txts, tgt_imgs, tgt_vids, tgt_auds):
            if txt != "" and SPECIAL_SEP_TOKEN in txt: # if this is a concrete text candidate (no img, vid, aud input)
                assert img is None and vid is None and aud is None, f"Error: Text candidate should not have other modality candidates. Found txt={txt}, img={img}, vid={vid}, aud={aud}."
                filename, caption = txt.split(SPECIAL_SEP_TOKEN)
                txt_insts.append(caption)
                name_insts.append(filename)

            if aud is not None: # if this is audio candidate
                aud_path = os.path.join(audio_root, aud)
                assert os.path.exists(aud_path), f"Audio {aud_path} does not exist."
                aud_insts.append({"path": aud_path, "bytes": None})
                txt_insts.append(f"<|audio_pad|> {TGT_INST} {txt}") # actually no tgt txt here
                name_insts.append(aud)
            else:
                aud_insts.append(None)

            if img is not None:
                img_path = os.path.join(image_root, img)
                assert os.path.exists(img_path), f"Image {img_path} does not exist."
                img_insts.append(ImageVideoInstance(
                    bytes=[None],
                    paths=[img_path],
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
                ).to_dict())
                txt_insts.append(process_input_text(TGT_INST, text=txt, model_backbone=model_backbone, add_image_token=True))
                name_insts.append(img)
            else:
                img_insts.append(None)

            if vid is not None: # if is video
                vid_path = os.path.join(video_root, vid)
                assert os.path.exists(vid_path), \
                    f"Video {vid_path} does not exist."
                frame_dir = os.path.join(frame_root, vid.split('.')[0]) # use the filename without extension as the frame dir name
                try:
                    save_frames(video_path=vid_path,
                                frame_dir=frame_dir,
                                max_frames_saved=max_frames_saved)
                    video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
                    vid_insts.append(ImageVideoInstance(
                        bytes=[None] * len(video_frame_paths),
                        paths=video_frame_paths,
                        resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
                    ).to_dict())
                    txt_insts.append(process_input_text(TGT_INST, text=txt, model_backbone=model_backbone, add_video_token=True))
                    name_insts.append(vid)
                except: # simply skip the cand video if not exist, however, if the pos sample is a video, raise error
                    print_rank(f"Loading frames for {vid_path} failed.")
                    if vid == pos_filename:
                        raise FileNotFoundError(f"Positive sample video {vid_path} not found or failed to load!")
                    else:
                        print_rank(f"Skipping candidate video {vid_path}.")
                        # vid_insts.append(None) 
                        # txt_insts.append(None)
                        continue # if as expected, no candidate will have more than 1 modality, so we can simply skip, if we append None here, no inputs will be available
            else:
                vid_insts.append(None)
            
        assert pos_filename in name_insts and len(name_insts) == len(set(name_insts)), \
            f'Error: pos_filename={pos_filename} NOT FOUND in cand_name_instances={name_insts}'
        assert len(name_insts) == len(txt_insts) == len(img_insts) == len(vid_insts) == len(aud_insts), \
            f"Error: Inconsistent candidate instance lengths for pos_filename={pos_filename}: {len(name_insts)} names, {len(txt_insts)} txts, {len(img_insts)} imgs, {len(vid_insts)} vids, {len(aud_insts)} audios."
        cand_images.append(img_insts)
        cand_videos.append(vid_insts)
        cand_texts.append(txt_insts)
        cand_audios.append(aud_insts)
        dataset_infos.append({
            "cand_names": name_insts,
            "label_name": pos_filename,
        })

    return {
        "query_text": query_texts, "query_image": query_images, "query_audio": [None]*len(query_texts),
        "cand_text": cand_texts, "cand_image": cand_images, "cand_video": cand_videos, "cand_audio": cand_audios,
        "dataset_infos": dataset_infos
    }

DATASET_PARSER_NAME = "mscoco_cmret"
DATASET_HF_PATH = "MINGYISU/t2iv" # can still use t2iv, will rename the dataset in the future
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_mscoco_cmret_dataset(model_args, data_args, *args, **kwargs):
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
