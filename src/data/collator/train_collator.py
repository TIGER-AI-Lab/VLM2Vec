from itertools import repeat
from typing import Optional, List
from torch.jit import isinstance

import logging
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments, TrainingArguments
import torch
from qwen_vl_utils import smart_resize

from src.model.processor import LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, \
    QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, PHI3V, process_vlm_inputs_fns
from PIL import Image
import io
from src.utils.basic_utils import print_rank, print_master


logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000


def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = []
    for k in keys:
        if isinstance(arg_val[k], torch.Tensor):
            chunked_tensor = arg_val[k].split(chunk_size, dim=0)
        else:
            chunked_tensor = [arg_val[k][i: i + chunk_size] for i in list(range(0, len(arg_val[k]), chunk_size))]
        chunked_tensors.append(chunked_tensor)
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]
    chunked_inputs = [{arg_key: c} for c in chunked_arg_val]

    return chunked_inputs


def split_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]
    keys = list(arg_val.keys())

    # for input_ids and attention_mask, split directly
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in ["input_ids", "attention_mask"]]

    # for pixel_values and image_sizes, need to split based on the position of images
    input_ids = arg_val["input_ids"]
    # positions = torch.nonzero(((input_ids < 0) & (input_ids > -MAX_INPUT_ID)) | input_ids == LLAVE_IMAGE_TOKEN_ID, as_tuple=True)
    positions = torch.nonzero((input_ids < 0) & (input_ids > -PHI_IMAGE_TOKEN_MAX_INPUT_ID), as_tuple=True)
    row_contain_image = torch.unique(positions[0])  # indicates which row in input_ids contain images
    num_chunks = len(chunked_tensors[0])
    chunk_image_count = []
    for chunk_idx in range(num_chunks):
        chunk_image_count.append(torch.sum(
            (row_contain_image >= chunk_idx * chunk_size) & (row_contain_image < (chunk_idx + 1) * chunk_size)).item())
    if "pixel_values" in keys:
        pixel_values = arg_val["pixel_values"]
        image_sizes = arg_val["image_sizes"]
        chunked_tensors.append(torch.split(pixel_values, chunk_image_count))
        chunked_tensors.append(torch.split(image_sizes, chunk_image_count))

    chunked_arg_val = []
    for kk, tt in zip(repeat(keys), zip(*chunked_tensors)):
        if "pixel_values" in keys and tt[2].numel() == 0:  # this chunk doesn't contain image
            chunked_arg_val.append(dict(zip(kk[:2], tt[:2])))
        else:
            chunked_arg_val.append(dict(zip(kk, tt)))

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    """
    Get either qry_reps or tgt_reps.
    """
    if x["qry_reps"] is None:
        return x["tgt_reps"]
    else:
        return x["qry_reps"]


@dataclass
class TrainTextImageDataCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        neg_inputs = self._get_batch_inputs(examples, "neg_text", "neg_image")
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_keyname, image_keyname):
        texts, images = [], []
        for example in examples:
            # @ruimeng filter invalid data examples here will lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                text, image = '  ', None
            text, image = example[text_keyname], example[image_keyname]
            if type(text) == list:
                if len(text) == 0 or len(image) == 0:
                    text, image = '  ', None
                else:
                    text, image = text[0], image[0]
            texts.append(text)
            images.append(image)
        inputs = {'text': texts, 'image': images}
        return inputs


@dataclass
class MultimodalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    batch_size: Optional[int] = None  # used to verify if a batch has invalid data

    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        for example in batch:
            # @ruimeng filter invalid data examples here may lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                text, visual_input = '  ', None
            else:
                text, raw_images = example[text_keyname], example[image_keyname]
                if type(raw_images) == dict:
                    visual_input = []
                    assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
                    num_images = len(raw_images['resolutions'])
                    for image_idx in range(num_images):
                        bytes = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                        path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                        image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                        if bytes is None and path is None:
                            image = None
                        elif bytes is not None:
                            # vidore, image inputs are already bytes
                            if bytes == b'':
                                image = None
                            else:
                                image = Image.open(io.BytesIO(bytes))
                        elif path is not None:
                            # mmeb/video datasets, lazy image loading and processing
                            if path == '':
                                image = None
                            else:
                                with Image.open(path) as img:
                                    image = img.convert("RGB")
                        else:
                            print_rank(f"\n{'=' * 50}\nsomething went wrong with a data point from {example['global_dataset_name']}, neither bytes or path is given. \n\t\tquery_text: {example['query_text']}")
                        if not self.data_args.resize_use_processor and image is not None and image_resolution:
                            image = image.resize(image_resolution)
                        if image is not None and (self.data_args.image_decay_factor is not None and image_resolution is None):
                            assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                            assert self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], "image_decay_factor is only supported for Qwen models"
                            # TODO: this is a hacky way to decay image resolution, need to be refactored
                            max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_max_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))
                            width, height = image.size
                            resized_height, resized_width = smart_resize(
                                height,
                                width,
                                min_pixels=self.data_args.resize_min_pixels,
                                max_pixels=max_pixels,
                            )
                            image = image.resize((resized_width, resized_height))  
                        visual_input.append(image)
                else:
                    visual_input = None
            texts.append(text)
            visual_inputs.append(visual_input)
        inputs = {'text': texts, 'images': visual_inputs}
        return inputs

    def _get_negative_inputs(self, batch, text_keyname, image_keyname):
        # Keep nested structure: list of lists for each example
        all_texts, all_visual_inputs = [], []
        for example in batch:
            # @ruimeng filter invalid data examples here may lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                dummy_texts, dummy_visual_inputs = [], []
                all_texts.append(dummy_texts)
                all_visual_inputs.append(dummy_visual_inputs)
            else:
                text_list, raw_images_list = example[text_keyname], example[image_keyname]

                assert len(text_list) == len(raw_images_list), f"Example {example.get('global_dataset_name', 'unknown')} has mismatched text/image lengths"

                example_texts, example_visual_inputs = [], []
                for text, raw_images in zip(text_list, raw_images_list):
                    visual_input = []
                    if type(raw_images) == dict:
                        assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
                        num_images = len(raw_images['resolutions'])
                        for image_idx in range(num_images):
                            bytes = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                            path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                            image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                            if bytes is None and ((path is None) or (not path)):
                                image = None
                            elif bytes is not None:
                                # vidore, image inputs are already bytes
                                image = Image.open(io.BytesIO(bytes))
                            elif (path is not None) and (path):
                                # mmeb/video datasets, lazy image loading and processing
                                with Image.open(path) as img:
                                    image = img.convert("RGB")
                            else:
                                print_rank(f"\n{'=' * 50}\nsomething went wrong with a data point from {example['global_dataset_name']}, neither bytes or path is given. \n\t\tquery_text: {example['query_text']}")
                            if not self.data_args.resize_use_processor and image is not None and image_resolution:
                                image = image.resize(image_resolution)
                            if image is not None and (self.data_args.image_decay_factor is not None and image_resolution is None):
                                assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                                assert self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], "image_decay_factor is only supported for Qwen models"
                                # TODO: this is a hacky way to decay image resolution, need to be refactored
                                max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_max_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))
                                width, height = image.size
                                resized_height, resized_width = smart_resize(
                                    height,
                                    width,
                                    min_pixels=self.data_args.resize_min_pixels,
                                    max_pixels=max_pixels,
                                )
                                image = image.resize((resized_width, resized_height))  
                            visual_input.append(image)
                    example_texts.append(text)
                    example_visual_inputs.append(visual_input)
                all_texts.append(example_texts)
                all_visual_inputs.append(example_visual_inputs)
        inputs = {'text': all_texts, 'images': all_visual_inputs}
        return inputs

    # Check if negatives exist and are non-empty
    @staticmethod
    def has_valid_negatives(example):
        neg_text_list = example.get('neg_text', [])
        if not neg_text_list:  # Empty list
            return False
        # Check if all texts are empty strings or nested empty lists
        for text in neg_text_list:
            if isinstance(text, list[str]):
                # Handle nested list case (list of list of strings)
                if text and any(t.strip() for t in text if isinstance(t, str)):
                    return True
            elif isinstance(text, str) and text.strip():
                # Handle simple string case
                return True
        return False


    def __call__(self, examples):
        """
        :param examples: 'query_text', 'query_image_path', 'pos_text', 'pos_image_path', 'neg_text', 'neg_image_path'
        """
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")

        bs = len(qry_inputs['text'])
        assert bs > 0, 'An empty batch'
        # pad batch to batch_size to avoid hanging in distributed training
        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expect batch size {self.batch_size}, but got batch size of {bs}")
            pass
        process_fn = process_vlm_inputs_fns[self.training_args.model_backbone]
        processed_qry_inputs = process_fn(qry_inputs, processor=self.processor, max_length=self.data_args.max_len)
        processed_qry_inputs['text'] = [e['query_text'] for e in examples]
        processed_qry_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]

        # Check if any example has valid negatives (cache results to avoid double computation)
        valid_negatives_cache = [self.has_valid_negatives(example) for example in examples]
        has_negatives = any(valid_negatives_cache)

        # Create processed_tgt_inputs when negatives exist
        if has_negatives:
            neg_inputs = self._get_negative_inputs(examples, "neg_text", "neg_image")
            # Create target inputs combining positives and negatives
            tgt_texts, tgt_images = [], []
            max_row_length = 0
            
            # First pass: collect all texts and images, find max_row_length
            for i, example in enumerate(examples):
                # Start with 1 positive
                row_texts = [pos_inputs['text'][i]]
                row_images = [pos_inputs['images'][i]]
                
                # Add negatives for this example if they exist (use cached result)
                if valid_negatives_cache[i]:
                    example_neg_texts = neg_inputs['text'][i]  # Get negatives for this specific example
                    example_neg_images = neg_inputs['images'][i]
                    
                    # Add safety check for mismatched lengths
                    assert len(example_neg_texts) == len(example_neg_images), f"Example {i} has mismatched neg text/image lengths"

                    # Add all available negatives for this example
                    for neg_text, neg_image in zip(example_neg_texts, example_neg_images):
                        row_texts.append(neg_text)
                        row_images.append(neg_image)

                max_row_length = max(max_row_length, len(row_texts))
                tgt_texts.append(row_texts)
                tgt_images.append(row_images)
            
            padded_tgt_texts, padded_tgt_images = [], []
            
            for row_texts, row_images in zip(tgt_texts, tgt_images):
                # Pad or truncate to max_row_length
                assert len(row_texts) <= max_row_length, f"row_texts length {len(row_texts)} exceeds max_row_length {max_row_length}"
                # Pad with empty strings and None images
                row_texts.extend([''] * (max_row_length - len(row_texts)))
                row_images.extend([None] * (max_row_length - len(row_images)))
                
                padded_tgt_texts.extend(row_texts)
                padded_tgt_images.extend(row_images)
            
            # Process the combined target inputs
            tgt_inputs = {'text': padded_tgt_texts, 'images': padded_tgt_images}
            processed_tgt_inputs = process_fn(tgt_inputs, processor=self.processor, max_length=self.data_args.max_len)
            processed_tgt_inputs['text'] = padded_tgt_texts
            processed_tgt_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples for _ in range(max_row_length)]
            
        else:
            processed_tgt_inputs = process_fn(pos_inputs, processor=self.processor, max_length=self.data_args.max_len)
            processed_tgt_inputs['text'] = [e['pos_text'] for e in examples]
            processed_tgt_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]

        # print_rank(f"\t\tQry collator: processed_qry_inputs['input_ids'].shape={processed_qry_inputs['input_ids'].shape}\t\tPos collator: processed_pos_inputs['input_ids'].shape={processed_tgt_inputs['input_ids'].shape}")
        # from collections import Counter
        # counter_by_source = Counter([e["global_dataset_name"] for e in examples])
        # for dname, count in counter_by_source.items():
        #     print_rank(f"\t\tdataset_name={dname}, count={count}")

        return processed_qry_inputs, processed_tgt_inputs
