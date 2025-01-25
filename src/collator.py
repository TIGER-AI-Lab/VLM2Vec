from itertools import repeat

import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments
import torch

from src.dataset import Phi_Image_token
from src.utils import LLAVA_NEXT, QWEN2_VL, PHI3V, print_rank

logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVE_IMAGE_TOKEN_ID = 32000

def process_vlm_inputs(model_inputs: dict, processor, backbone_name, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            if backbone_name == LLAVA_NEXT:
                inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
            elif backbone_name == QWEN2_VL:
                inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            elif backbone_name == PHI3V:
                inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            if backbone_name == LLAVA_NEXT:
                inputs = processor(images=image, text=text, return_tensors="np", max_length=max_length, truncation=True)
            elif backbone_name == QWEN2_VL:
                inputs = processor(images=[image], text=[text], return_tensors="np", max_length=max_length, truncation=True)
            elif backbone_name == PHI3V:
                inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        if backbone_name == LLAVA_NEXT:
            # dummy image inputs based on the first valid data point
            pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
            image_size_for_padding = torch.from_numpy(list(v for v in image_sizes if v is not None)[0])
            # make the batch full tensors
            pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            image_sizes = [torch.from_numpy(v) if v is not None else image_size_for_padding for v in image_sizes]
            image_sizes = torch.cat(image_sizes, dim=0)
        if backbone_name == QWEN2_VL:
            pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
            pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            if image_grid_thw:
                image_grid_thw_for_padding = torch.from_numpy(list(v for v in image_grid_thw if v is not None)[0])
                image_grid_thw = [torch.from_numpy(v) if v is not None else image_grid_thw_for_padding for v in image_grid_thw]
                image_grid_thw = torch.cat(image_grid_thw, dim=0)
                inputs['image_grid_thw'] = image_grid_thw
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    # print_rank('[text.shape]' + str(input_ids.shape))
    # if image_exists:
    #     print_rank('[image.shape]' + str(inputs['pixel_values'].shape))

    return inputs


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


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
class TrainRawInputCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, 0, 1)
        pos_inputs = self._get_batch_inputs(examples, 2, 3)
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_idx, image_idx):
        texts, images = [], []
        for example in examples:
            text, image = example[text_idx], example[image_idx]
            texts.append(text)
            images.append(image)
        inputs = {'text': texts, 'image': images}
        return inputs


@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        examples = {'text': [e[0] for e in examples], 'image': [e[1] for e in examples]}
        inputs = process_vlm_inputs(examples,
                                    processor = self.processor,
                                    backbone_name = self.model_args.model_backbone,
                                    max_length = self.data_args.max_len)
        return inputs


@dataclass
class DeprecatedTrainCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, 0, 1)
        pos_inputs = self._get_batch_inputs(examples, 2, 3)
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_idx, image_idx):
        input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
        image_exist = False
        backbone = self.model_args.model_backbone
        for example in examples:
            text, image = example[text_idx], example[image_idx]
            if image is None:
                if backbone == LLAVA_NEXT:
                    inputs = self.processor(images=None, text=text, return_tensors="pt")
                elif backbone == QWEN2_VL:
                    inputs = self.processor(text=[text], images=None, return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                elif backbone == PHI3V:
                    inputs = self.processor(text, None, return_tensors="pt", max_length=self.data_args.max_len,
                                            truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(None)
                image_sizes.append(None)
            else:
                image_exist = True
                if backbone == LLAVA_NEXT:
                    inputs = self.processor(images=image, text=text, return_tensors="pt")
                elif backbone == QWEN2_VL:
                    inputs = self.processor(images=[image], text=[text], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                elif backbone == PHI3V:
                    inputs = self.processor(text=text, images=[image], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(inputs['pixel_values'])
                if 'image_sizes' in inputs:
                    image_sizes.append(inputs['image_sizes'])
                elif 'image_grid_thw' in inputs:
                    image_grid_thw.append(inputs['image_grid_thw'])

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        if not image_exist:
            dummy_pixel_values = torch.zeros(input_ids.shape[0], 1)
            dummy_image_sizes = torch.zeros(input_ids.shape[0], 1)
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': dummy_pixel_values,
                'image_sizes': dummy_image_sizes,
            }
        else:
            pixel_values_shape = list(set(v.shape for v in pixel_values if v is not None))[0]
            pixel_values = [v if v is not None else torch.zeros(pixel_values_shape) for v in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'image_sizes': image_sizes,
            }
            if self.model_args.model_backbone != QWEN2_VL:
                image_sizes_shape = list(set(v.shape for v in image_sizes if v is not None))[0]
                image_sizes = [v if v is not None else torch.zeros(image_sizes_shape) for v in image_sizes]
                image_sizes = torch.cat(image_sizes, dim=0)
                inputs['image_sizes'] = image_sizes
            elif image_grid_thw: # for qwen2 which the model processes image patches
                image_grid_thw = torch.cat(image_grid_thw, dim=0)
                inputs['image_grid_thw'] = image_grid_thw

        return inputs


@dataclass
class DeprecatedEvalCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, image_sizes = [], [], []
        image_exist = False
        for example in examples:
            text, image = example
            if image is None:
                if self.model_args.model_backbone == LLAVA_NEXT:
                    inputs = self.processor(images=None, text=text, return_tensors="pt")
                else:
                    inputs = self.processor(text, None, return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(None)
                image_sizes.append(None)
            else:
                image_exist = True
                if self.model_args.model_backbone == LLAVA_NEXT:
                    inputs = self.processor(images=image, text=text, return_tensors="pt")
                else:
                    inputs = self.processor(text, [image], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(inputs['pixel_values'])
                image_sizes.append(inputs['image_sizes'])

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        if not image_exist:
            dummy_pixel_values = torch.zeros(input_ids.shape[0], 1)
            dummy_image_sizes = torch.ones(input_ids.shape[0], 1)
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': dummy_pixel_values,
                'image_sizes': dummy_image_sizes,
            }
        else:
            pixel_values_shape = list(set(v.shape for v in pixel_values if v is not None))[0]
            pixel_values = [v if v is not None else torch.zeros(pixel_values_shape) for v in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            image_sizes_shape = list(set(v.shape for v in image_sizes if v is not None))[0]
            image_sizes = [v if v is not None else torch.ones(image_sizes_shape) for v in image_sizes]
            image_sizes = torch.cat(image_sizes, dim=0)
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'image_sizes': image_sizes,
            }

        return inputs


@dataclass
class CLIPCollator:
    data_args: DataArguments
    vis_processors: AutoProcessor
    txt_processors: AutoTokenizer

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, attention_mask = [], [], []
        image_exist, text_exist = False, False
        for example in examples:
            text, image = example
            if image is not None:
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_inputs = self.vis_processors(images=image, return_tensors="pt")
                image_exist = True
                pixel_values.append(image_inputs['pixel_values'])
            if text:
                text_exist = True
            text_inputs = self.txt_processors(text, padding=getattr(self.data_args, "padding", True), max_length=self.data_args.max_len, truncation=True, return_tensors="pt")
            input_ids.append(text_inputs["input_ids"].squeeze(0))
        if text_exist:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.txt_processors.pad_token_id
            )
            attention_mask = input_ids.ne(self.txt_processors.pad_token_id)
        if image_exist:
            pixel_values = torch.cat(pixel_values, dim=0)
        if text_exist and image_exist:
            assert input_ids.size()[0]==pixel_values.size()[0]
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }

        return inputs


@dataclass
class OpenCLIPCollator:
    data_args: DataArguments
    vis_processors: AutoProcessor
    txt_processors: AutoTokenizer

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, attention_mask = [], [], []
        image_exist, text_exist = False, False
        for example in examples:
            text, image = example
            if image is not None:
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_inputs = self.vis_processors(image).unsqueeze(0)
                image_exist = True
                pixel_values.append(image_inputs)
            if text:
                text_exist = True
            text_inputs = self.txt_processors(text)
            input_ids.append(text_inputs)
        if text_exist:
            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = input_ids.ne(0)
        if image_exist:
            pixel_values = torch.cat(pixel_values, dim=0)
        if text_exist and image_exist:
            assert input_ids.size()[0]==pixel_values.size()[0]
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }

        return inputs
