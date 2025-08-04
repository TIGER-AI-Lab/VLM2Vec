from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import os
from datasets.features.image import image_to_bytes

from torch.jit import isinstance
from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, \
    RESOLUTION_MAPPING
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS, INTERN_VL3
from src.utils import print_master, print_rank
from torch.utils.data import Dataset

from torch.utils.data import Dataset
import torch
import logging
from dataclasses import dataclass
import torch.distributed as dist
from typing import Iterator, List, Tuple, Union

logger = logging.getLogger(__name__)






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
    pos_text_list = [pos_text]+ neg_text_list
    pos_image_list = [pos_image]+neg_image_path_list
    
    return {"query_text": [qry_text], "query_image": [qry_image],
            "pos_text": pos_text_list, "pos_image": pos_image_list,
            "neg_text": [], "neg_image": []}


DATASET_PARSER_NAME = "mmeb"
@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_mmeb_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    dataset_split = kwargs.get("dataset_split", "original")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    # dataset = load_dataset(dataset_name, subset_name, split=f"{dataset_split}")

    parquet_path = os.path.join(".", dataset_name, subset_name, "train-00000-of-00001.parquet")
    print(">>>>>>>>>>>>>>", subset_name, "HN" in subset_name, parquet_path, flush=True)
    dataset = load_dataset(
        f"parquet",
        data_files=parquet_path,
        # split=f"{data_args.dataset_split}[:{num_sample_per_subset}]",
        split=f"{data_args.dataset_split}",
    )


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




class TrainTextImageDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        train_data = []
        print_rank(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
        for subset in data_args.subset_name:
            subset_data = load_dataset(
                self.data_args.dataset_name, subset,
                split=f"{self.data_args.dataset_split}[:{data_args.num_sample_per_subset}]",
            )
            train_data.append(subset_data)
        self.train_data = concatenate_datasets(train_data)

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image

    def __getitem__(self, data_idx) -> Tuple[str, List[str]]:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>get image called, {idx}", flush=True)
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )
        if 'neg_text' in self.train_data.column_names:
            neg_texts, neg_image_paths = self.train_data[data_idx]["neg_text"], self.train_data[data_idx]["neg_image_path"]
        else:
            neg_texts, neg_image_paths = [''] * len(data_idx), [] * len(data_idx)
        if isinstance(data_idx, int):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            neg_texts = [neg_texts]
            neg_image_paths = [neg_image_paths]
        _qry_texts, _qry_images, _pos_texts, _pos_images, _neg_texts, _neg_images = [], [], [], [], [], []
        backbone = self.model_args.model_backbone
        for qry_text, qry_image_path, pos_text, pos_image_path, neg_text, neg_image_path \
            in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths, neg_texts, neg_image_paths):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2
            if backbone != PHI3V:
                qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[backbone])
                pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[backbone])
                neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[backbone]) if neg_text else None
            qry_image = self._get_image(qry_image_path)
            pos_image = self._get_image(pos_image_path)
            neg_image = self._get_image(neg_image_path) if neg_image_path else None
            if (not qry_text and not qry_image) or (not pos_text and not pos_image):
                print("empty inputs")
                continue
            _qry_texts.append(qry_text)
            _qry_images.append(qry_image)
            _pos_texts.append(pos_text)
            _pos_images.append(pos_image)
            _neg_texts.append(neg_text)
            _neg_images.append(neg_image)

        return {"query_text": _qry_texts, "query_image": _qry_images,
                "pos_text": _pos_texts, "pos_image": _pos_images,
                "neg_text": _neg_texts, "neg_image": _neg_images}


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field, mod_instruction=None):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone

        self.eval_data = load_dataset(
            self.data_args.dataset_name,
            subset,
            split=self.data_args.dataset_split,
        )
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        # self.paired_dataset = datasets.Dataset.from_dict({
        #     "text": [pair["text"] for pair in self.paired_data],
        #     "img_path": [pair["img_path"] for pair in self.paired_data]
        # })
        if(("tgt" in text_field) and (mod_instruction is not None) and self.data_args.tgt_prefix_mod):
            print("Using TGT mod", mod_instruction, flush=True)            
            self.paired_dataset = datasets.Dataset.from_dict({
                                "text": [mod_instruction + pair["text"] for pair in self.paired_data],
                                "img_path": [pair["img_path"] for pair in self.paired_data]
                                })
            print(">>>>>>>>>>>>>inside tgt_mod_txt", flush=True)
            # import ipdb; ipdb.set_trace()
        else:
            print("Not using TGT mod", mod_instruction, flush=True)            
            self.paired_dataset = datasets.Dataset.from_dict({
                                "text": [pair["text"] for pair in self.paired_data],
                                "img_path": [pair["img_path"] for pair in self.paired_data]
                                })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.backbone != PHI3V:
            text = text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.backbone])
        # if(self.backbone==INTERN_VL3):
        #     full_img_path = os.path.join(self.data_args.image_dir, img_path)
        #     return text, [full_img_path]
        return text, self._get_image(img_path)

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
        return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif type(row[text_field]) == list:
                assert type(row[img_path_field]) == list and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        paired_data = sorted(paired_data, key=lambda k:k['img_path'])
        return paired_data


class FlickrDataset(Dataset):
    def __init__(self, modality, model_backbone):
        self.model_backbone = model_backbone
        self.modality = modality
        self.raw_data = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval", split="test")
        if modality == "image":
            self.eval_data, self.image_names = self.get_image_data()
        else:
            self.eval_data, self.image_names = self.get_text_data()

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        return self.eval_data[idx]

    def __getitem__(self, idx):
        text, image = self.eval_data[idx]
        if self.backbone != PHI3V:
            text = text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.backbone])
            if self.data_args.image_resolution:
                image = process_image(image, self.data_args.image_resolution)
        return text, image

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_backbone != PHI3V:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
        return image

    def get_image_data(self):
        eval_data, image_names = [], []
        # i2t
        inst = "<|image_1|> Find an image caption describing the given image."  # llava-1344-step1k4, i2t=94.0, t2i=80.26
        # inst = "<|image_1|> Represent the given image for image caption retrieval."  # llava-1344-step1k4, i2t=94.6, t2i=78.98
        # t2i
        # inst = "<|image_1|> Represent the given image."  # MSCOCO t2i

        for row in self.raw_data:
            eval_data.append((inst, row["image"]))
            image_names.append(row["filename"])
        return eval_data, image_names

    def get_text_data(self):
        eval_data, image_names = [], []
        # i2t
        inst = ""
        # t2i
        # inst = "Retrieve an image that matches the given caption: "
        # inst = "Find me an everyday image that matches the given caption."  # MSCOCO t2i
        for row in self.raw_data:
            for caption in row["caption"]:
                # eval_data.append((caption, None))
                eval_data.append((inst + caption, None))
                image_names.append(row["filename"])
        return eval_data, image_names


@dataclass
class CustomRandomSampler(torch.utils.data.sampler.RandomSampler):
    """
    Sampler used when training on multiple datasets to ensure each 
    batch only contains samples from one dataset for the majority of cases.
    """
    total_batch_size: int = 8
    ds_lens: List[int] = None
    _num_samples: int = None
    data_source: TrainTextImageDataset = None
    replacement: bool = False
    ordered_datasets: bool = False
    same_datasets: bool = False
    random_datasets: bool = False
    num_repeats: int = 1
    chunk_size: int = None
    def __len__(self):
        return len(self.data_source)

    # def __iter__(self) -> Iterator[int]:
    #     yield from [99999-i for i in range(len(self.data_source))]

    def __iter__(self) -> Iterator[int]:
        
        if not hasattr(self, "generator") or self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # We have multiple datasets each with a different number of samples
        # e.g. [100, 150, 50]
        # We would like to sample from them such that as much as possible each batch
        # only has samples from the same dataset.
        # For example if our batch size is 4 then
        # indices might be [0,1,2,3,100,101,102,103,150,151,152,153,50,51,52,53]
        # To do so:
        # 1. Shuffle the indices of each dataset separately
        # 2. Create batches with only samples from one dataset
        # 3. Keep the remaining samples which do not fit into a batch separate
        # 4. Then create mixed batches from the remaining samples
        # 5. Then yield randomly from all the batches
        # Testing:
        # ds_lens = [100, 150, 50]
        # batch_size = 8
        # Create random indices for each dataset

        if(self.ordered_datasets):
            print(">>>>>>>>>>>>>>>>>>>>>>Asking for ordered datasets", flush=True)
            ds_indices = []
            for n in self.ds_lens:
                assert n % self.chunk_size == 0, f"Dataset length {n} is not divisible by chunk_size"
                chunks = torch.arange(n).view(-1, self.chunk_size)  # shape (n // 16, 16)
                perm = torch.randperm(chunks.size(0))  # randomize chunks
                shuffled_chunks = chunks[perm]
                ds_indices.append(shuffled_chunks.flatten().tolist())

            ds_indices = [[i + sum(self.ds_lens[:j]) for i in ds_indices[j]] for j in range(len(self.ds_lens))]
            ds_batches = [list(torch.split(torch.tensor(ds_indices[j]), self.total_batch_size)) for j in range(len(self.ds_lens))]
        elif(self.same_datasets):
            ds_indices = [torch.randperm(n, generator=generator).tolist() for n in self.ds_lens]
            ds_indices = [[i + sum(self.ds_lens[:j]) for i in ds_indices[j]] for j in range(len(self.ds_lens))]
            ds_batches = [list(torch.split(torch.tensor(ds_indices[j]), self.total_batch_size)) for j in range(len(self.ds_lens))]
        elif(self.random_datasets):
            total_len = sum(self.ds_lens)
            shuffled_indices = torch.randperm(total_len, generator=generator)
            ds_batches = [list(torch.split(shuffled_indices, self.total_batch_size))]
        else:
            assert False, "none of odibn,sdibn,rdibn found"
        incomplete_indices = []

        for b in ds_batches:
            if len(b[-1]) < self.total_batch_size:
                incomplete_indices.append(b.pop())

        # import ipdb; ipdb.set_trace()
        # assert len(incomplete_indices)==0
        # if incomplete_indices:
        if False:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=generator).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(torch.tensor(incomplete_indices), self.total_batch_size))
            if len(mixed_batches[-1]) < self.total_batch_size:
                mixed_batches.pop()
            # Merge all batches to look like [...tensor([259, 273, 284, 289]), tensor([262, 280, 295, 258]), ...]
            ds_batches = sum(ds_batches, []) + mixed_batches
            logger.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches) - len(mixed_batches)} single-dataset batches & {len(mixed_batches)} mixed dataset batches.")
        else:
            ds_batches = sum(ds_batches, [])
            logger.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches)} single-dataset batches.")


        all_batches = []  # Start empty

        for _ in range(self.num_repeats):  # Iterate K times to include the first epoch as well
            epoch_order = torch.randperm(len(ds_batches), generator=generator).tolist()
            shuffled_batches = [ds_batches[i] for i in epoch_order]  # Keep batch structure
            all_batches.append(shuffled_batches)

        
        # Flatten across all epochs
        ds_batches = [batch for epoch_batches in all_batches for batch in epoch_batches]

        # Convert to list of indices
        ds_batches = [int(i) for i in torch.cat(ds_batches).tolist()]

        print(">>>>>>>>>>>>>>>>>>>>ds batches", ds_batches[:128], self.ordered_datasets, self.total_batch_size, flush=True)

        # Yield the indices
        yield from ds_batches



