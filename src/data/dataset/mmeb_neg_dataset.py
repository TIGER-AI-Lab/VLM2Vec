from datasets import load_dataset, concatenate_datasets
from PIL import Image
import os
import random
from datasets.features.image import image_to_bytes

from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.utils.basic_utils import print_master, print_rank

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image


def get_image_bytes_and_path(img_path, image_dir, model_backbone, image_resolution):
    '''
    caveat: datasets will convert PIL.Image.Image objects into Arrow-compatible types (aka bytes) behind the scene and only image.filename is reserved (datasets/features/image.py L311)
    solution: (20240227) defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
    '''
    if not img_path:
        return None
    full_img_path = os.path.join(image_dir, img_path)
    image = Image.open(full_img_path)
    backbone = model_backbone
    if backbone != PHI3V and image_resolution:
        image = process_image(image,  image_resolution)
    bytes = image_to_bytes(image)
    return {"bytes": bytes, "path": full_img_path}


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_dir = kwargs['image_dir']
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']
    num_hardneg = kwargs.get("num_hardneg", 0)

    batch_size = len(batch_dict['qry'])
    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    for qry_text, qry_image_path, pos_text, pos_image_path, neg_text_list, neg_image_path_list in \
        zip(batch_dict['qry'], batch_dict['qry_image_path'],
            batch_dict['pos_text'], batch_dict['pos_image_path'],
            batch_dict.get('neg_text', [['']] * batch_size), batch_dict.get('neg_image_path', [[None]] * batch_size)):
        if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
            print("empty inputs")
            continue
        
        # Handle negative sampling based on num_hardneg
        if num_hardneg == 0:
            # If num_hardneg is 0, use no negatives
            neg_text_list = ['']
            neg_image_path_list = [None]
        elif num_hardneg > 0 and len(neg_text_list) > num_hardneg:
            # If we have more negatives than needed, randomly sample
            sampled_indices = random.sample(range(len(neg_text_list)), num_hardneg)
            neg_text_list = [neg_text_list[i] for i in sampled_indices]
            neg_image_path_list = [neg_image_path_list[i] for i in sampled_indices]
        # If len(neg_text_list) <= num_hardneg, use all available negatives (no change needed)
        
        if model_backbone != PHI3V:
            qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            neg_text = [text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]) if text else '' for text in neg_text_list]
        
        query_texts.append(qry_text)
        pos_texts.append(pos_text)
        neg_texts.append(neg_text)
        # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
        qry_image = {"bytes": [None], "paths": [os.path.join(image_dir, qry_image_path) if qry_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        pos_image = {"bytes": [None], "paths": [os.path.join(image_dir, pos_image_path) if pos_image_path else None], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
        neg_image = [{"bytes": [None], "paths": [os.path.join(image_dir, neg_image_path) if neg_image_path else ''], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for neg_image_path in neg_image_path_list]
        
        query_images.append(qry_image)
        pos_images.append(pos_image)
        neg_images.append(neg_image)
    if len(query_texts) == 0:
        print('something went wrong')
    # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    return {"query_text": query_texts, "query_image": query_images,
            "pos_text": pos_texts, "pos_image": pos_images,
            "neg_text": neg_texts, "neg_image": neg_images}


DATASET_PARSER_NAME = "mmeb_neg"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_mmeb_neg_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    dataset_split = kwargs.get("dataset_split", "original")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    dataset = load_dataset(dataset_name, subset_name, split=f"{dataset_split}")

    column_names = dataset.column_names
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    num_rows = dataset.num_rows

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{subset_name}'
    remove_columns = ['qry', 'qry_image_path', 'pos_text', 'pos_image_path', 'neg_text']
    if 'neg_image_path' in column_names:
        remove_columns.append('neg_image_path')
    
    # Use batched processing like MMEB
    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=2048,
        remove_columns=remove_columns,
        drop_last_batch=False,  # keep final partial batches
    )
    
    dataset = dataset.cast(MULTIMODAL_FEATURES)
    print_master(f"Loaded {DATASET_PARSER_NAME}/{subset_name} dataset with {num_rows} samples")

    return dataset
