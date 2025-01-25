import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments
import torch


logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
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
        image_mask = []
        image_exist = False
        for example in examples:
            text, image = example[text_idx], example[image_idx]
            if image is None:
                image_mask.append(0)
                if self.model_args.model_backbone == "llava_next":
                    inputs = self.processor(images=None, text=text, return_tensors="pt")
                elif self.model_args.model_backbone == "qwen":
                    inputs = self.processor(text=[text], images=None, return_tensors="pt",
                                            max_length=self.data_args.max_len, truncation=True)
                else:
                    inputs = self.processor(text=text, images=None, return_tensors="pt",
                                            max_length=self.data_args.max_len, truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            else:
                image_mask.append(1)
                image_exist = True
                if self.model_args.model_backbone == "llava_next":
                    inputs = self.processor(images=image, text=text, return_tensors="pt")
                    pixel_values.append(inputs['pixel_values'])
                elif self.model_args.model_backbone == "qwen":
                    inputs = self.processor(text=[text], images=[image], return_tensors="pt",
                                            max_length=self.data_args.max_len, truncation=False)
                    pixel_values.append(inputs['pixel_values'].unsqueeze(0))
                else:
                    inputs = self.processor(text=text, images=[image], return_tensors="pt",
                                            max_length=self.data_args.max_len, truncation=True)
                    pixel_values.append(inputs['pixel_values'])
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                if "image_sizes" in inputs:
                    image_sizes.append(inputs['image_sizes'])
                if "image_grid_thw" in inputs:
                    image_grid_thw.append(inputs['image_grid_thw'])

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if image_exist:
            inputs['image_mask'] = torch.Tensor(image_mask)
            pixel_values = torch.cat(pixel_values, dim=0)
            inputs['pixel_values'] = pixel_values
            if image_sizes:
                image_sizes = torch.cat(image_sizes, dim=0)
                inputs['image_sizes'] = image_sizes
            elif image_grid_thw:
                image_grid_thw = torch.cat(image_grid_thw, dim=0)
                inputs['image_grid_thw'] = image_grid_thw

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
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, image_sizes = [], [], []
        image_exist = False
        for example in examples:
            text, image = example
            if image is None:
                if self.model_args.model_backbone == "llava_next":
                    inputs = self.processor(images=None, text=text, return_tensors="pt")
                else:
                    inputs = self.processor(text, None, return_tensors="pt", max_length=self.data_args.max_len,
                                            truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(None)
                image_sizes.append(None)
            else:
                image_exist = True
                if self.model_args.model_backbone == "llava_next":
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
