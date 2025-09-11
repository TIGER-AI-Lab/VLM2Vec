from datasets import load_dataset
from PIL import Image
from datasets.features.image import image_to_bytes
import io, os, random

from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, convert_neg_fields, ImageVideoInstance, \
    MULTIMODAL_FEATURES, RESOLUTION_MAPPING
from src.model.processor import VLM_IMAGE_TOKENS


def process_query(query, prompt, image_token):
    parts = []
    if prompt:
        parts.append(prompt)
    if image_token and str(image_token).strip():
        parts.append(image_token)
    if query:
        parts.append(query)
    return " ".join(parts)


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']
    batch_size = len(batch_dict['query_id'])
    image_dir = kwargs['image_dir']
    num_hardneg = kwargs.get("num_hardneg", 0)
    answer_key = 'answer' if 'answer' in batch_dict else 'answers'

    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []
    for query_text, query_image, pos_image_ids, neg_image_ids, answer, source in \
        zip(batch_dict['query_text'], batch_dict.get('query_image', [None] * batch_size), batch_dict['positive_document_ids'], \
            batch_dict['negative_document_ids'], batch_dict.get(answer_key, [''] * batch_size), batch_dict.get('source', [''] * batch_size)):
        # ignore prompt since most prompts too long
        query = process_query(query_text, prompt="", image_token=VLM_IMAGE_TOKENS[model_backbone] if query_image is not None else "")
        # query = process_query(query, prompt=prompt, image_token=VLM_IMAGE_TOKENS[model_backbone])
        assert (query_image is not None) == (VLM_IMAGE_TOKENS[model_backbone] in query), f"query_image and image_token inconsistent, query_image={query_image}, query={query}"

        # Handle negative sampling based on num_hardneg
        neg_text = [''] * len(neg_image_ids)  # empty text for negative samples
        if num_hardneg == 0:
            # If num_hardneg is 0, use no negatives
            neg_text = ['']
            neg_image_ids = [None]
        elif num_hardneg > 0 and len(neg_text) > num_hardneg:
            # If we have more negatives than needed, randomly sample
            sampled_indices = random.sample(range(len(neg_text)), num_hardneg)
            neg_text = [neg_text[i] for i in sampled_indices]
            neg_image_ids = [neg_image_ids[i] for i in sampled_indices]
        # If len(neg_text) <= num_hardneg, use all available negatives (no change needed)

        # for wiki-ss-nq-news, answer is list
        if isinstance(answer, list):
            answer = ", ".join(answer)
        pos_image_tokens = "".join([VLM_IMAGE_TOKENS[model_backbone] for image_ids in pos_image_ids if image_ids])
        pos_text = process_query(answer, prompt="", image_token=pos_image_tokens)
        neg_image_tokens = [VLM_IMAGE_TOKENS[model_backbone] if neg_image_id else '' for neg_image_id in neg_image_ids]
        neg_text = [process_query(t, prompt="", image_token=tok) for t, tok in zip(neg_text, neg_image_tokens)]
        query_texts.append(query)
        pos_texts.append(pos_text)
        neg_texts.append(neg_text)
        query_images.append(ImageVideoInstance(
            bytes=[image_to_bytes(query_image) if query_image is not None else None],
            paths=[None],
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
        ).to_dict())

        pos_image_paths = []
        for corpus_id in pos_image_ids:
            image_path = f'{image_dir}/{corpus_id}.png'
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'Image path {image_path} not found.')
            pos_image_paths.append(image_path)

        pos_images.append(ImageVideoInstance(
            bytes=[None] * len(pos_image_paths),
            paths=pos_image_paths,
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(pos_image_paths),
        ).to_dict())

        neg_image = []
        for corpus_id in neg_image_ids:
            image_path = f'{image_dir}/{corpus_id}.png' if corpus_id else ''
            if corpus_id and not os.path.exists(image_path):
                raise FileNotFoundError(f'Image path {image_path} not found.')
            neg_image.append(ImageVideoInstance(
                bytes=[None],
                paths=[image_path],
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
            ).to_dict())
        neg_images.append(neg_image)

    if len(query_texts) == 0:
        print('something went wrong')
    # print_rank(f"global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)}, batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    return {"query_text": query_texts, "query_image": query_images,
            "pos_text": pos_texts, "pos_image": pos_images,
            "neg_text": neg_texts, "neg_image": neg_images}

def corpus_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_dir = kwargs['image_dir']
    batch_size = len(batch_dict['docid'])

    cand_texts, cand_images, dataset_infos = [], [], []
    for corpus_id, text, image, title in \
        zip(batch_dict['docid'], batch_dict.get('text', [''] * batch_size), batch_dict['image'], batch_dict.get('title', [''] * batch_size)):
        image_path = f'{image_dir}/{corpus_id}.png'
        if not os.path.exists(image_path):
            os.makedirs(image_dir, exist_ok=True)
            image.save(image_path)
        text = (((title + " ") if title else "") + (text if text else "")).strip()
        cand_texts.append([process_query(text, prompt="", image_token=VLM_IMAGE_TOKENS[model_backbone])])
        cand_images.append([ImageVideoInstance(
            bytes=[None],
            paths=[image_path],
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
        ).to_dict()])
        dataset_infos.append({
            "cand_names": [corpus_id],
        })

    return {"cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}

DATASET_PARSER_NAME = "vidore_neg"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_vidore_neg_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", None)
    dataset_split = kwargs.get("dataset_split", "train")
    dataset_path = kwargs.get("dataset_path", None)
    corpus_name = kwargs.get("corpus_name", None)
    corpus_split = kwargs.get("corpus_split", "train")
    corpus_path = kwargs.get("corpus_path", None)

    if dataset_name:
        dataset = load_dataset(dataset_name, split=dataset_split)
    elif dataset_path:
        dataset = load_dataset("parquet", data_files=dataset_path, split="train")
    
    if corpus_name:
        corpus = load_dataset(corpus_name, split=corpus_split)
    elif corpus_path:
        corpus = load_dataset("parquet", data_files=corpus_path, split="train")

    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)  # convert to IterableDataset and multiple shards
    corpus = corpus.to_iterable_dataset(num_shards=num_shards)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    if dataset_name:
        kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{dataset_name}'
    else:
        subset_name = kwargs.get("subset_name", None)
        kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{subset_name}'
    # dataset = dataset.shuffle(buffer_size=8192, seed=training_args.seed)
    remove_columns = ['docid', 'image', 'text', 'source']
    remove_columns = [c for c in remove_columns if c in corpus.column_names]
    corpus = corpus.map(lambda x: corpus_prepare(x, **kwargs), batched=True, batch_size=2048,
                        remove_columns=remove_columns,
                        drop_last_batch=False)
    remove_columns = ['query_id', 'query_text', 'query_image', 'positive_document_ids', 'negative_document_ids', 'answer', 'source']
    remove_columns = [c for c in remove_columns if c in dataset.column_names]
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=2048,
                          remove_columns=remove_columns,
                          drop_last_batch = True)
    # dataset = dataset._resolve_features()
    # features = _infer_features_from_batch(dataset._head()) # not working: {ArrowInvalid}ArrowInvalid('Could not convert <PIL.Image.Image image mode=RGB size=128x128 at 0x7F7C794E9BD0> with type Image: did not recognize Python value type when inferring an Arrow data type')
    dataset = dataset.cast(MULTIMODAL_FEATURES)
    setattr(dataset, 'num_rows', num_rows)
    # print_master(f"Loaded {DATASET_PARSER_NAME}/{dataset_name} dataset with {num_rows} samples")
    return dataset
