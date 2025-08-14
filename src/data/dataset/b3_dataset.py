from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import os
from datasets.features.image import image_to_bytes

from torch.jit import isinstance
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.utils import print_master, print_rank
from torch.utils.data import Dataset

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
def data_prepare(example, *args, **kwargs):
    image_dir = kwargs['image_dir']
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']
    
    qry_text = example['qry']
    qry_image_path = example['qry_image_path']
    pos_text = example['pos_text']
    pos_image_path = example['pos_image_path']
    neg_text_list = example.get('neg_text', [])
    neg_image_path_list = example.get('neg_image_path', [])
    
    # batch_size = len(batch_dict['qry'])
    # query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    # for qry_text, qry_image_path, pos_text, pos_image_path, neg_text_list, neg_image_path_list in \
    #     zip(batch_dict['qry'], batch_dict['qry_image_path'],
    #         batch_dict['pos_text'], batch_dict['pos_image_path'],
    #         batch_dict.get('neg_text', [''] * 1), batch_dict.get('neg_image_path', [None] * 1)):
        
    neg_text_list = [] if((not neg_text_list) or type(neg_text_list)==str) else neg_text_list
    neg_image_path_list = [] if((not neg_image_path_list) or type(neg_image_path_list)==str) else neg_image_path_list
    
    #! neg_text is a list. need to modify all following parts.
    # import ipdb; ipdb.set_trace()

    # if (not qry_text and not qry_image_path) or (not pos_text and not pos_image_path):
    #     print("empty inputs")
    #     continue
    if model_backbone != PHI3V:
        qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        neg_text_list = [neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone]) if neg_text else '' for neg_text in neg_text_list]
    # query_texts.append(qry_text)
    # pos_texts.append(pos_text)
    # pos_texts.extend(neg_text_list)
    # neg_texts.append(neg_text_list)
    # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
    qry_image = {"bytes": [None], "paths": [os.path.join(image_dir, qry_image_path) if qry_image_path else ''], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
    pos_image = {"bytes": [None], "paths": [os.path.join(image_dir, pos_image_path) if pos_image_path else ''], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]}
    # import ipdb; ipdb.set_trace()
    try:
        neg_image_path_list = [{"bytes": [None], "paths": [os.path.join(image_dir, neg_image_path) if neg_image_path else ''], "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for neg_image_path in neg_image_path_list]
    except:
        import ipdb; ipdb.set_trace()
    # query_images.append(qry_image)
    # pos_images.append(pos_image)
    # pos_images.extend(neg_image_path_list)
    # neg_images.append(neg_image_path_list)
    # import ipdb; ipdb.set_trace()

    if not qry_text:
        print('something went wrong')
    # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    
    return {"query_text": [qry_text], "query_image": [qry_image],
            "pos_text": pos_text, "pos_image": pos_image,
            "neg_text": neg_text_list, "neg_image": neg_image_path_list}


DATASET_PARSER_NAME = "b3"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_b3_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    dataset_split = kwargs.get("dataset_split", "original")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    dataset = load_dataset(dataset_name, subset_name, split=f"{dataset_split}")

    column_names = dataset.column_names
    # if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
    #     num_rows = int(num_sample_per_subset)
    #     dataset = dataset.select(range(num_rows))
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    #! big change. Not iterable dataset anymore.
    # dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards
    

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{subset_name}'
    # dataset = dataset.shuffle(buffer_size=8192, seed=training_args.seed)
    remove_columns = ['qry', 'qry_image_path', 'pos_text', 'pos_image_path']
    if 'neg_image_path' in column_names:
        remove_columns.append('neg_text')
        remove_columns.append('neg_image_path')
    dataset = dataset.map(lambda x, idx: {**data_prepare(x, **kwargs), "idx": idx}, with_indices=True, batched=False, remove_columns=remove_columns, load_from_cache_file=False, cache_file_name=None, keep_in_memory=True,)
    #! check here
    # dataset = dataset._resolve_features()
    # features = _infer_features_from_batch(dataset._head()) # not working: {ArrowInvalid}ArrowInvalid('Could not convert <PIL.Image.Image image mode=RGB size=128x128 at 0x7F7C794E9BD0> with type Image: did not recognize Python value type when inferring an Arrow data type')
    #! commenting casting
    # dataset = dataset.cast(MULTIMODAL_FEATURES)
    print_master(f"Loaded {DATASET_PARSER_NAME}/{subset_name} dataset with {num_rows} samples")

    # import ipdb; ipdb.set_trace()
    # num_rows in iterable_dataset is overridden, set it here for printing dataset stats
    #! commenting out num_rows for us
    # setattr(dataset, 'num_rows', num_rows)

    return dataset