import os, ast
from typing import List
from src.utils.basic_utils import print_master
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS
from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, ImageVideoInstance, RESOLUTION_MAPPING


def process_multi_images(image_basedir, image_paths) -> List[str | None]:
    if not image_paths:
        return [None]

    if isinstance(image_paths, str):
        try:
            image_paths = ast.literal_eval(image_paths)
        except (ValueError, SyntaxError):
            image_paths = [image_paths]

    img_path_list = []
    for image_path in image_paths:
        img_path_list.append(os.path.join(image_basedir, image_path))

    return img_path_list


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_text, qry_image_path, pos_text, pos_image_path, pos_id, retrieval_type in \
        zip(batch_dict['qry_text'], batch_dict['qry_image_path'], batch_dict['pos_text'], batch_dict['pos_image_path'], batch_dict['pos_id'], batch_dict['retrieval_type']):
        if model_backbone != PHI3V:
            qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        query_texts.append([qry_text])
        cand_texts.append([pos_text])
        # 20240227 defer image loading and transforming to data-loader to avoid repeatedly Serialization/Deserialization of PIL Images
        qry_img_paths = process_multi_images(image_root, qry_image_path)
        pos_img_paths = process_multi_images(image_root, pos_image_path)
        query_images.append(
            [
                ImageVideoInstance(
                    bytes=[None] * len(qry_img_paths),
                    paths=qry_img_paths,
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_img_paths),
                ).to_dict()
            ]
        )
        cand_images.append(
            [
                ImageVideoInstance(
                    bytes=[None] * len(pos_img_paths),
                    paths=pos_img_paths,
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(pos_img_paths),
                ).to_dict()
            ]
        )
        dataset_infos.append({
            "cand_names": [pos_id],
            "label_name": [pos_id],
            "retrieval_type": retrieval_type,
        })

    return {"query_text": query_texts, "query_image": query_images, "cand_text": cand_texts, "cand_image": cand_images, "dataset_infos": dataset_infos}


def corpus_prepare(batch_dict, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    cand_texts, cand_images, dataset_infos = [], [], []
    for cand_text, cand_image_path, cand_id in \
        zip(batch_dict['cand_text'], batch_dict['cand_image_path'], batch_dict['cand_id']):
        # some image_name are super long...

        if model_backbone != PHI3V:
            cand_text = cand_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        cand_texts.append([cand_text])
        cand_image_paths = process_multi_images(image_root, cand_image_path)
        cand_images.append(
            [
                ImageVideoInstance(
                    bytes=[None] * len(cand_image_paths),
                    paths=cand_image_paths,
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(cand_image_paths),
                ).to_dict()
            ]
        )
        dataset_infos.append({
            "cand_names": [cand_id],
        })

    return {"cand_text": cand_texts, "cand_image": cand_images, "dataset_infos": dataset_infos}

DATASET_PARSER_NAME = "gui"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_gui_dataset(model_args, data_args, **kwargs):
    # hf_dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    # hf_dataset_split = kwargs.get("dataset_split", "Mind2Web")
    hf_dataset_name = EVAL_DATASET_HF_PATH[kwargs['dataset_name']][0]
    hf_dataset_split = EVAL_DATASET_HF_PATH[kwargs['dataset_name']][2]
    dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
    corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    print_master(f"Loaded {kwargs['dataset_name']}")
    print_master(f"#hf_dataset_name={hf_dataset_name}")
    print_master(f"#hf_dataset_split={hf_dataset_split}")
    print_master(f"#queries={len(dataset)}")
    print_master(f"#cand={len(corpus)}")
    
    corpus = corpus.map(lambda x: corpus_prepare(x, **kwargs), batched=True,
                        batch_size=64, num_proc=1,
                        drop_last_batch=False, load_from_cache_file=False)
    corpus = corpus.select_columns(['cand_text', 'cand_image', 'dataset_infos'])
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=64, num_proc=1,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, corpus