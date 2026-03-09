import logging
import math

import PIL
from transformers.image_utils import ChannelDimension

from src.model.baseline_backbone.colpali import ColPaliProcessor

logger = logging.getLogger(__name__)

import torch
import numpy as np
from src.utils.basic_utils import print_master

from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from src.model.vlm_backbone.qwen2_vl_tokenselection import \
    Qwen2VLForConditionalGeneration as Qwen2VLTokenSelectionForConditionalGeneration, \
    Qwen2VLProcessor as Qwen2VLTokenSelectionProcessor
from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import InternVideo2_Stage2
from src.model.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.model.vlm_backbone.qwen2_5_vl_tokenselection import \
    Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_TokenSelectionForConditionalGeneration
from src.model.vlm_backbone.omni_embed import OmniEmbedForConditionalGeneration


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000
TEXT_ONLY_MAX_LEN = 2048


def _text_only_max_length(max_length):
    return TEXT_ONLY_MAX_LEN if max_length is None else max_length


def _infer_image_min_size(processor, fallback=28):
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    try:
        patch_val = int(patch_size)
        merge_val = int(merge_size)
    except Exception:
        return fallback
    if patch_val > 0 and merge_val > 0:
        return patch_val * merge_val
    return fallback


def _pad_to_min_image_size(image, min_size):
    if not isinstance(image, PIL.Image.Image):
        return image
    width, height = image.size
    if width >= min_size and height >= min_size:
        return image
    if width <= 0 or height <= 0:
        return image
    new_width = max(width, min_size)
    new_height = max(height, min_size)
    if new_width == width and new_height == height:
        return image
    if image.mode == "RGBA":
        fill = (0, 0, 0, 0)
    elif image.mode == "RGB":
        fill = (0, 0, 0)
    else:
        fill = 0
    canvas = PIL.Image.new(image.mode, (new_width, new_height), color=fill)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    canvas.paste(image, (left, top))
    return canvas


def _install_qwen_omni_warning_filters():
    root_logger = logging.getLogger()
    if getattr(root_logger, "_suppress_qwen_omni_warnings", False):
        return

    class _SuppressQwenOmniWarnings(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "System prompt modified, audio output may not work as expected." in msg:
                return False
            if "video processor config saved in `preprocessor.json` file which is deprecated" in msg:
                return False
            if "Unrecognized keys in `rope_scaling` for 'rope_type'='default': {'mrope_section'}" in msg:
                return False
            return True

    root_logger.addFilter(_SuppressQwenOmniWarnings())
    root_logger._suppress_qwen_omni_warnings = True

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl_tokenselection'
QWEN2_5_VL_TOKENSELECTION = 'qwen2_5_vl_tokenselection'
QWEN2_5_OMNI = 'qwen2_5_omni'  # Qwen2.5-Omni / Omni-Embed
NVOMNIEMBED = 'nvomniembed'  # NVIDIA omni-embed-nemotron
INTERNVIDEO2 = 'internvideo2'
GME = 'gme'  # QWEN2-VL
LamRA = 'lamra'  # QWEN2-VL
LamRA_QWEN2_5 = 'lamra_qwen25'  # QWEN2.5-VL
COLPALI = 'colpali'  # PaliGemma-3B
E5_V = 'e5_v'  # Llava_next
MODEL2BACKBONE = {  # keys are from hf_config.model_type or manually added if not provided
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_vl_tokenselection': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
    'qwen2_vl_tokenselection': QWEN2_VL_TOKENSELECTION,
    'qwen2_5_vl_tokenselection': QWEN2_5_VL_TOKENSELECTION,
    'qwen2_5_omni': QWEN2_5_OMNI,
    'qwen2_5_omni_thinker': QWEN2_5_OMNI,
    'nvomniembed': NVOMNIEMBED,
    'internvideo2': INTERNVIDEO2,
    'gme': GME, 
    'lamra': LamRA,
    'lamra_qwen25': LamRA,
    'colpali': COLPALI,
    'e5_v': E5_V,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|image_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|image_pad|>",
    QWEN2_5_OMNI: "<|image_pad|>",
    NVOMNIEMBED: "<|image_pad|>",
    GME: "<|image_pad|>",
    LamRA: "<|image_pad|>",
    LamRA_QWEN2_5: "<|image_pad|>",
    INTERNVIDEO2: "",
    COLPALI: "",
    E5_V: "<image>",
}

VLM_VIDEO_TOKENS = {
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|video_pad|>",
    QWEN2_5_VL: "<|video_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|video_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|video_pad|>",
    QWEN2_5_OMNI: "<|video_pad|>",
    NVOMNIEMBED: "<|video_pad|>",
    GME: "<|video_pad|>",
    LamRA: "<|video_pad|>",
    LamRA_QWEN2_5: "<|video_pad|>",
    INTERNVIDEO2: "",
    COLPALI: "",
    E5_V: "<image>",
}

backbone2model = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
    QWEN2_VL_TOKENSELECTION: Qwen2VLTokenSelectionForConditionalGeneration,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_5_VL_TokenSelectionForConditionalGeneration,
    QWEN2_5_OMNI: OmniEmbedForConditionalGeneration,
    INTERNVIDEO2: InternVideo2_Stage2,
    E5_V: LlavaNextForConditionalGeneration,
}


def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    Note: due to this change, https://github.com/huggingface/transformers/commit/9215cc62d4366072aacafa4e44028c1ca187167b#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356L1102
    """
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading processor from: {model_name_or_path}')
    if model_args.model_backbone == PHI3V:
        from src.model.baseline_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == LLAVA_NEXT:
        from src.model.baseline_backbone.llava_next import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
    elif model_args.model_backbone in [QWEN2_VL, GME, LamRA]:
        from src.model.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer, size=size
        )
    elif model_args.model_backbone == QWEN2_VL_TOKENSELECTION:
        from src.model.vlm_backbone.qwen2_vl_tokenselection.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        if data_args is not None:
            image_processor.do_resize = data_args.resize_use_processor
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    elif model_args.model_backbone in [QWEN2_5_VL, LamRA_QWEN2_5]:
        from src.model.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, image_processor=image_processor, tokenizer=tokenizer)
    elif model_args.model_backbone == QWEN2_5_VL_TOKENSELECTION:
        # TODO: qwen2.5 token selection not working yet
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    elif model_args.model_backbone == QWEN2_5_OMNI:
        # Qwen2.5-Omni / Omni-Embed: use official processor with apply_chat_template.
        from src.model.olm_backbone.qwen2_5_moni.processing_qwen2_5_omni import Qwen2_5OmniEmbeddingProcessor
        _install_qwen_omni_warning_filters()

        processor_path = model_args.processor_name if model_args.processor_name else model_args.model_name
        processor = Qwen2_5OmniEmbeddingProcessor.from_pretrained(processor_path, trust_remote_code=True)
        root_logger = logging.getLogger()
        if not getattr(root_logger, "_suppress_qwen_omni_prompt_warning", False):
            class _SuppressQwenOmniPromptWarning(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    msg = record.getMessage()
                    return "System prompt modified, audio output may not work as expected." not in msg

            root_logger.addFilter(_SuppressQwenOmniPromptWarning())
            root_logger._suppress_qwen_omni_prompt_warning = True
    elif model_args.model_backbone == NVOMNIEMBED:
        from transformers import AutoProcessor
        _install_qwen_omni_warning_filters()
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        
    elif model_args.model_backbone == INTERNVIDEO2:
        return None
    elif model_args.model_backbone == COLPALI:
        from transformers import AutoProcessor
        processor = ColPaliProcessor.from_pretrained(model_args.model_name)
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO: NOT FINISHED YET!
    input_ids, pixel_values, image_sizes = [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        # in theory, each batch item should contain a list of frames, but we still check for exceptions here
        # if no images as input (not likely to happen in mmeb pro cases)
        if images is None or (type(images)==list and any(i is None for i in images)):
            inputs = processor(
                images=None,
                text=text,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
        else:
            image_exists = True
            # in theory, valid images should be a list of frames
            assert isinstance(images, list), f"images should be a list, but got {type(images)}"
            inputs = processor(images=images, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_sizes.append(inputs['image_sizes'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask,
        # 'texts': texts,
        # 'images': visual_inputs,
    }
    image_exists = any([p is not None for p in pixel_values])
    if image_exists:
        pixel_values = torch.from_numpy(np.array(pixel_values)).float()
        pixel_values_shape = pixel_values.shape
        pixel_values = pixel_values.reshape(pixel_values_shape[0] * pixel_values_shape[1], *pixel_values_shape[2:])
        image_sizes = torch.tensor(np.array(image_sizes)).long()
        image_sizes_shape = image_sizes.shape
        image_sizes = image_sizes.reshape(image_sizes_shape[0] * image_sizes_shape[1], *image_sizes_shape[2:])
        inputs['pixel_values'] = torch.from_numpy(np.array(pixel_values)).float()
        inputs['image_sizes'] = torch.tensor(np.array(image_sizes)).long()
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(
                text,
                None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
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
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Qwen2_VL_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]

    # 1. iterate each pair and process, since processors do not support processing for mixed batch (contains data w/ and w/o visual inputs)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(
                text=[text],
                images=None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            try:
                if vlm_image_token in text:
                    if isinstance(images, PIL.Image.Image):
                        # images is a single image
                        images = [images]
                    for iid, image in enumerate(images):
                        # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                        if image.size[0] < 28 or image.size[1] < 28:
                            image = image.resize((56, 56))
                            images[iid] = image
                    inputs = processor(text=[text], images=images, return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
                elif vlm_video_token in text:
                    # TODO: check text/video data validity
                    inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
                else:
                    raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in the text: {text}")
            except Exception as e:
                for i in images:
                    print(i.filename)
                raise e
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(), 
        'texts': texts,
        'images': visual_inputs,
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    return inputs

def Gme_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    inputs = {
        'texts': model_inputs['text'],
        'images': model_inputs['images'],
    }
    return inputs


def Qwen2_VL_TokenSelection_process_fn(model_inputs: dict, processor: Qwen2VLTokenSelectionProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    patch_pos, select_mask = [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(
                text=[text],
                images=None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            patch_pos.append(None)
            select_mask.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            image_exists = True
            # TODO only
            # handling multi-image data from videos, cannot deal with mixed image + video data
            if VLM_IMAGE_TOKENS[QWEN2_VL] in text:
                inputs = processor(text=[text], images=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            elif VLM_VIDEO_TOKENS[QWEN2_VL] in text:
                assert len(images) > 1, f"Video data must have more than 1 frame, got {len(images)}"
                inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            else:
                raise NotImplementedError(f"Unsupported visual token in text: {text}")
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
                if 'patch_pos' in inputs:
                    patch_pos.append(inputs['patch_pos'])
                if 'select_mask' in inputs:
                    select_mask.append(inputs['select_mask'])
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                patch_pos.append(None)
                select_mask.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']

    if image_exists:
        if patch_pos:
            patch_pos_shape_for_padding = list(v.shape for v in patch_pos if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else (torch.zeros(patch_pos_shape_for_padding) - 1) for v in patch_pos]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=-1) for pos in key_tmp]
            patch_pos = torch.cat(padded_key, dim=0)
        if select_mask:
            select_mask_shape_for_padding = list(v.shape for v in select_mask if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else torch.ones(select_mask_shape_for_padding).bool() for v in select_mask]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=True) for pos in key_tmp]
            select_mask = torch.cat(padded_key, dim=0)

    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long()
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw
    inputs['patch_pos'] = patch_pos
    inputs['select_mask'] = select_mask

    return inputs


def InternVL_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO not working yet
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(
                text,
                None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
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
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def ColPali_process_fn(model_inputs: dict, processor, max_length=None):
    texts, images = model_inputs['text'], model_inputs['images']
    
    input_ids_batch = []
    attention_mask_batch = []
    pixel_values_batch = []

    for text, image in zip(texts, images):
        if image is not None:
            inputs = processor.process_images([image])
            pixel_values_batch.append(inputs['pixel_values'])
        else:
            inputs = processor.process_queries([text], max_length=_text_only_max_length(max_length))
            pixel_values_batch.append(None)

        input_ids_batch.append(inputs['input_ids'].squeeze().tolist())
        attention_mask_batch.append(inputs['attention_mask'].squeeze().tolist())

    # Pad input_ids and attention_mask
    padded_text_inputs = processor.tokenizer.pad(
        {'input_ids': input_ids_batch, 'attention_mask': attention_mask_batch},
        return_tensors="pt"
    )
    
    final_input_ids = padded_text_inputs['input_ids']
    final_attention_mask = padded_text_inputs['attention_mask']

    # Handle pixel_values
    if any(pv is not None for pv in pixel_values_batch):
        # Find a representative shape for pixel_values
        representative_pv_shape = None
        for pv in pixel_values_batch:
            if pv is not None:
                representative_pv_shape = pv.shape
                break
        
        processed_pixel_values = []
        for pv in pixel_values_batch:
            if pv is None:
                # Create a zero tensor of the representative shape
                processed_pixel_values.append(torch.zeros(representative_pv_shape))
            else:
                processed_pixel_values.append(pv)
        final_pixel_values = torch.cat(processed_pixel_values)
    else:
        # No images in the batch at all
        batch_size = len(texts)
        # SigLIP expects 3 channels (RGB) and a square image of 448x448 based on the error (1024 patches = 32x32 patches, with patch_size=14, 32*14=448)
        default_channels = 3
        default_height = 448
        default_width = 448
        final_pixel_values = torch.zeros(batch_size, default_channels, default_height, default_width)

    return {
        'input_ids': final_input_ids,
        'attention_mask': final_attention_mask,
        'pixel_values': final_pixel_values,
    }


def InternVideo2_process_fn(model_inputs: dict, processor, max_length=None):
    if all(x is None for x in model_inputs["images"]):
        # Text side
        from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        inputs = tokenizer(
            model_inputs["text"],
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt")
    else:
        # Video side
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert from PIL image to tensor (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                 std=[0.229, 0.224, 0.225])  # ImageNet std
        ])
        frame_list = model_inputs["images"]
        # to make image inputs be exact 4 frames
        # Case 1: frame_list is flat (not a list of lists), e.g., [PIL, PIL, ...]
        if type(frame_list[0]) is not list:
            frame_list = [[img.copy() for _ in range(4)] for img in frame_list]
        # Case 2: frame_list is already a list of lists, ensure each has exactly 4 images
        elif type(frame_list[0]) is list and len(frame_list[0]) != 4:
            new_list = []
            for frames in frame_list:
                if len(frames) < 4:
                    frames = frames + [frames[-1].copy() for _ in range(4 - len(frames))]
                elif len(frames) > 4:
                    # Sample 4 indices uniformly across the sequence
                    indices = np.linspace(0, len(frames) - 1, num=4, dtype=int)
                    frames = [frames[i] for i in indices]
                new_list.append(frames)
            frame_list = new_list
        pixel_values = [
            torch.stack([preprocess(img) for img in frames], dim=0)  # (num_frames, C, H, W)
            for frames in frame_list
        ]

        pixel_values = torch.stack(pixel_values, dim=0)  # (B, num_frames, C, H, W)
        inputs = {'pixel_values': pixel_values}

    return inputs


def Omni_process_fn(model_inputs: dict, processor, max_length=None):
    """
    Qwen2.5-Omni / Omni-Embed processor for EVAL (collate-time).
    Safe rules:
    - Always returns torch tensors for input_ids / attention_mask (LongTensor).
    - Supports text-only / text+image / text+video / text+audio (audio optional).
    - Does NOT rely on Dataset.map storing tensors (avoid Arrow -> list issue).
    """
    import inspect
    import torch
    import PIL
    import numpy as np
    import torchaudio
    import io
    import random

    texts = model_inputs.get("text", [])
    images = model_inputs.get("images", None)
    videos = model_inputs.get("videos", None)
    audios = model_inputs.get("audios", None)         # could be None or list, keep as 'audios' for compatibility
    audio_sample_rate = model_inputs.get("audio_sample_rate", None)

    if texts is None:
        texts = []
    if not texts:
        raise ValueError("Omni_process_fn: at least one text is required.")

    # Ensure all arrays have the same length as texts
    batch_size = len(texts)
    if images is None:
        images = [None] * batch_size
    if videos is None:
        videos = [None] * batch_size
    if audios is None:
        audios = [None] * batch_size

    # Check lengths and pad/truncate if necessary
    if len(images) != batch_size:
        if len(images) < batch_size:
            images.extend([None] * (batch_size - len(images)))
        else:
            images = images[:batch_size]
    if len(videos) != batch_size:
        if len(videos) < batch_size:
            videos.extend([None] * (batch_size - len(videos)))
        else:
            videos = videos[:batch_size]
    if len(audios) != batch_size:
        if len(audios) < batch_size:
            audios.extend([None] * (batch_size - len(audios)))
        else:
            audios = audios[:batch_size]

    if not any(t and str(t).strip() for t in texts):
        raise ValueError("Omni_process_fn: at least one non-empty text is required.")

    base_processor = getattr(processor, "base", processor)
    min_image_size = _infer_image_min_size(processor)

    def _squeeze_leading_ones(x, max_squeeze=2):
        if x is None:
            return None
        if not isinstance(x, torch.Tensor):
            return x
        for _ in range(max_squeeze):
            if x.dim() >= 1 and x.size(0) == 1:
                x = x.squeeze(0)
            else:
                break
        return x

    def _to_pil_image(x):
        if x is None or isinstance(x, str):
            return x
        if isinstance(x, PIL.Image.Image):
            return _pad_to_min_image_size(x, min_image_size)
        arr = None
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu()
            if t.dim() == 3 and t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
                t = t.permute(1, 2, 0)
            arr = t.numpy()
        elif isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
        if arr is None:
            return x
        if arr.dtype.kind == "f":
            arr = (arr * 255.0).clip(0, 255).astype("uint8")
        elif arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype("uint8")
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return _pad_to_min_image_size(PIL.Image.fromarray(arr), min_image_size)

    def _to_video_input(v):
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, list):
            frames = [f for f in v if f is not None]
            return [_to_pil_image(f) for f in frames] if frames else None
        if isinstance(v, (torch.Tensor, np.ndarray)):
            t = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            if t.ndim != 4:
                raise ValueError(f"Video tensor must be 4D, got shape {tuple(t.shape)}")
            if t.shape[-1] in (1, 3):
                frames = [t[i] for i in range(t.shape[0])]
            else:
                frames = [np.transpose(t[i], (1, 2, 0)) for i in range(t.shape[0])]
            return [_to_pil_image(f) for f in frames]
        return v

    def _to_audio_input(a):
        if a is None:
            return None
        if isinstance(a, torch.Tensor):
            if a.ndim != 1:
                raise ValueError(f"Audio must be 1D torch.Tensor waveform, got {type(a)} shape={getattr(a,'shape',None)}")
            return a.detach().cpu().numpy()
        if isinstance(a, np.ndarray):
            return a
        return np.asarray(a)

    def _load_audio_item(item, target_sr, min_samples, max_samples):
        if item is None:
            return None

        if isinstance(item, dict) and "array" in item:
            wav = torch.tensor(item["array"], dtype=torch.float32)
            sr = int(item.get("sampling_rate", target_sr))
            if wav.ndim > 1:
                wav = wav.mean(0)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            if wav.numel() < min_samples:
                return None
            if wav.numel() > max_samples:
                start = random.randint(0, wav.numel() - max_samples)
                wav = wav[start : start + max_samples]
            return wav

        if isinstance(item, torch.Tensor):
            wav = item.float()
            if wav.ndim > 1:
                wav = wav.mean(0)
            if wav.numel() < min_samples:
                return None
            if wav.numel() > max_samples:
                start = random.randint(0, wav.numel() - max_samples)
                wav = wav[start : start + max_samples]
            return wav

        if isinstance(item, np.ndarray):
            wav = torch.tensor(item, dtype=torch.float32)
            if wav.ndim > 1:
                wav = wav.mean(0)
            if wav.numel() < min_samples:
                return None
            if wav.numel() > max_samples:
                start = random.randint(0, wav.numel() - max_samples)
                wav = wav[start : start + max_samples]
            return wav

        if isinstance(item, dict):
            a_path = item.get("path") or item.get("audio_path") or item.get("video_path")
            a_bytes = item.get("bytes", None)
            start_t = float(item.get("start", 0.0) or 0.0)
            end_v = item.get("end", None)
            end_t = float(end_v) if end_v is not None else None

            if a_bytes is not None:
                wave, sr = torchaudio.load(io.BytesIO(a_bytes))
            elif a_path:
                info = torchaudio.info(a_path)
                sr = info.sample_rate
                frame_offset = int(start_t * sr)
                num_frames = int((end_t - start_t) * sr) if end_t is not None else -1
                if num_frames == 0:
                    return None
                wave, _ = torchaudio.load(a_path, frame_offset=frame_offset, num_frames=num_frames)
            else:
                return None

            if wave.dim() > 1:
                wave = wave.mean(0)
            if sr != target_sr:
                wave = torchaudio.functional.resample(wave, sr, target_sr)
            if wave.numel() < min_samples:
                return None
            if wave.numel() > max_samples:
                start = random.randint(0, wave.numel() - max_samples)
                wave = wave[start : start + max_samples]
            return wave

        return None

    # -----------------------------------------
    # 1) Build normalized batches (AutoProcessor path)
    # -----------------------------------------
    images_batch = []
    videos_batch = []
    audios_batch = []
    has_video = False
    has_image = False
    target_sr = int(audio_sample_rate or 16000)
    min_audio_samples = int(target_sr * 0.025)
    max_audio_seconds = model_inputs.get("audio_max_seconds", None)
    if max_audio_seconds is not None:
        max_audio_samples = int(float(max_audio_seconds) * target_sr)
    else:
        max_audio_frames = int(model_inputs.get("audio_max_frames", 1024))
        max_audio_samples = max_audio_frames * 160
    for text, image, video, audio in zip(texts, images, videos, audios):
        text = str(text)
        if not text.strip():
            raise ValueError("Omni_process_fn: empty text is not allowed.")

        # If video is not provided but image is a multi-frame list, treat it as video.
        if video is None and isinstance(image, list):
            non_null = [v for v in image if v is not None]
            if len(non_null) > 1:
                video = image
                image = None

        image_in = _to_pil_image(image)
        video_in = _to_video_input(video)
        if audio is None:
            audio_in = None
        else:
            audio_in = _load_audio_item(audio, target_sr, min_audio_samples, max_audio_samples)
            if audio_in is not None:
                audio_in = _to_audio_input(audio_in)
        if isinstance(image_in, list):
            image_in = [im for im in image_in if im is not None]
            if not image_in:
                image_in = None
        if isinstance(image_in, PIL.Image.Image):
            w, h = image_in.size
            if w < 28 or h < 28:
                image_in = None
        if isinstance(image_in, list):
            filtered = []
            for im in image_in:
                if isinstance(im, PIL.Image.Image):
                    w, h = im.size
                    if w < 28 or h < 28:
                        continue
                filtered.append(im)
            image_in = filtered if filtered else None
        if isinstance(video_in, list):
            filtered = []
            for im in video_in:
                if isinstance(im, PIL.Image.Image):
                    w, h = im.size
                    if w < 28 or h < 28:
                        continue
                filtered.append(im)
            video_in = filtered if filtered else None

        has_image = has_image or (image_in is not None)
        has_video = has_video or (video_in is not None)
        images_batch.append(image_in)
        videos_batch.append(video_in)
        audios_batch.append(audio_in)

    # Group by modality signature to avoid processor constraints
    # For Qwen2.5-Omni, ensure consistent modality across each group
    idxs = list(range(len(texts)))
    groups = {}

    for i in idxs:
        has_image = images_batch[i] is not None
        has_video = videos_batch[i] is not None
        has_audio = audios_batch[i] is not None

        if has_image and has_video:
            raise ValueError(f"Sample {i}: cannot have both image and video")

        has_visual = has_image or has_video

        # Group by exact modality combination
        if has_visual and has_audio:
            group_key = "visual_audio"
        elif has_visual and not has_audio:
            group_key = "visual_only"
        elif has_audio and not has_visual:
            group_key = "audio_only"
        else:
            group_key = "text_only"

        groups.setdefault(group_key, []).append(i)

    out_list = []
    order = []
    for group_key, sub in groups.items():
        # For audio groups, filter to only include samples with valid audio
        if group_key in ["visual_audio", "audio_only"]:
            valid_sub = [i for i in sub if audios_batch[i] is not None]
            if not valid_sub:
                continue  # Skip this group if no valid audio samples
            sub = valid_sub

        sub_texts = [texts[i] for i in sub]
        sub_images = [images_batch[i] for i in sub]
        sub_videos = [videos_batch[i] for i in sub]
        sub_audios = [audios_batch[i] for i in sub]

        kwargs = dict(
            text=sub_texts,
            return_tensors="pt",
        )
        if group_key == "text_only":
            if max_length is None:
                kwargs["max_length"] = 2048
            else:
                kwargs["max_length"] = min(int(max_length), 2048)
        elif max_length is not None:
            kwargs["max_length"] = max_length

        has_images = any(im is not None for im in sub_images)
        has_videos = any(v is not None for v in sub_videos)

        if has_images and has_videos:
            raise ValueError("Omni_process_fn: cannot mix images and videos in the same group.")

        if has_images:
            kwargs["images"] = sub_images
        if has_videos:
            kwargs["videos"] = sub_videos

        # For audio groups, all samples should have valid audio by construction
        if group_key in ["visual_audio", "audio_only"]:
            kwargs["audio"] = sub_audios  # All should be valid
            # Don't pass sampling_rate directly - audio is already resampled

        outputs = base_processor(**kwargs)
        out_list.append(outputs)
        order.extend(sub)

    outputs = {}
    if out_list:
        merged = {}
        for out in out_list:
            for k, v in out.items():
                merged.setdefault(k, []).append(v)
        for k, chunks in merged.items():
            if isinstance(chunks[0], torch.Tensor):
                outputs[k] = torch.cat(chunks, dim=0)
            else:
                tmp = []
                for c in chunks:
                    tmp.extend(c)
                outputs[k] = tmp

        inv = torch.empty(len(order), dtype=torch.long)
        for pos, idx in enumerate(order):
            inv[idx] = pos
        for k, v in list(outputs.items()):
            if isinstance(v, torch.Tensor) and v.size(0) == len(order):
                outputs[k] = v.index_select(0, inv.to(v.device))

    if "audio_attention_mask" in outputs and "feature_attention_mask" not in outputs:
        outputs["feature_attention_mask"] = outputs.pop("audio_attention_mask")

    feats = outputs.get("input_features", None)
    fam = outputs.get("feature_attention_mask", None)
    if isinstance(feats, torch.Tensor) and feats.dim() == 3:
        if feats.shape[1] != 128 and feats.shape[2] == 128:
            feats = feats.transpose(1, 2)
        if fam is None:
            fam = torch.ones(feats.shape[0], feats.shape[2], dtype=torch.long)
        if isinstance(fam, torch.Tensor) and fam.dim() == 2:
            if fam.shape[1] != feats.shape[2]:
                min_len = min(fam.shape[1], feats.shape[2])
                feats = feats[:, :, :min_len]
                fam = fam[:, :min_len]
            outputs["input_features"] = feats
            outputs["feature_attention_mask"] = fam
            # Force audio lengths to match feature mask to avoid RoPE index overflow.
            outputs["audio_feature_lengths"] = fam.sum(-1).long()

    # Ensure input_ids/attention_mask alignment.
    ids = outputs.get("input_ids", None)
    am = outputs.get("attention_mask", None)
    if isinstance(ids, torch.Tensor) and isinstance(am, torch.Tensor):
        if ids.dim() == 2 and am.dim() == 2 and ids.shape[1] != am.shape[1]:
            min_len = min(ids.shape[1], am.shape[1])
            outputs["input_ids"] = ids[:, :min_len]
            outputs["attention_mask"] = am[:, :min_len]

    # squeeze any leading singleton dims produced by processor
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in outputs:
            outputs[key] = _squeeze_leading_ones(outputs.get(key), max_squeeze=2)

    return outputs


def NVOmni_process_fn(model_inputs: dict, processor, max_length=None):
    """
    NVIDIA omni-embed-nemotron processor for EVAL (collate-time).
    Align with the official example: apply_chat_template + process_mm_info.
    """
    texts = model_inputs.get("text", []) or []
    images = model_inputs.get("images", None)
    videos = model_inputs.get("videos", None)
    audios = model_inputs.get("audios", None)
    audio_sample_rate = model_inputs.get("audio_sample_rate", None)
    min_image_size = _infer_image_min_size(processor)

    def _strip_special_tokens(text: str) -> str:
        if text is None:
            return ""
        stripped = text
        for tok in (VLM_IMAGE_TOKENS.get(NVOMNIEMBED), VLM_VIDEO_TOKENS.get(NVOMNIEMBED)):
            if tok:
                stripped = stripped.replace(tok, " ")
        return " ".join(stripped.split())

    # Build Qwen-Omni style documents with per-sample content.
    documents = []
    for idx, t in enumerate(texts):
        content = []
        clean_text = _strip_special_tokens(t)
        if not clean_text:
            clean_text = " "
        content.append({"type": "text", "text": clean_text})

        if images is not None and idx < len(images):
            img_item = images[idx]
            if isinstance(img_item, list):
                for im in img_item:
                    if im is not None:
                        content.append({"type": "image", "image": _pad_to_min_image_size(im, min_image_size)})
            elif img_item is not None:
                content.append({"type": "image", "image": _pad_to_min_image_size(img_item, min_image_size)})

        if videos is not None and idx < len(videos):
            vid_item = videos[idx]
            if vid_item is not None:
                if isinstance(vid_item, list):
                    vid_item = [_pad_to_min_image_size(frame, min_image_size) for frame in vid_item]
                content.append({"type": "video", "video": vid_item})

        if audios is not None and idx < len(audios):
            aud_item = audios[idx]
            if isinstance(aud_item, torch.Tensor):
                aud_item = aud_item.detach().cpu().numpy()
            elif aud_item is not None and not isinstance(aud_item, np.ndarray):
                aud_item = np.asarray(aud_item, dtype=np.float32)
            if aud_item is not None:
                content.append({"type": "audio", "audio": aud_item})

        documents.append([{"role": "user", "content": content}])

    def _is_allowed_media(value, media_type: str) -> bool:
        if isinstance(value, str):
            return True
        if media_type == "image":
            return isinstance(value, PIL.Image.Image)
        if media_type == "video":
            return isinstance(value, list)  # list of frames/objects
        if media_type == "audio":
            return isinstance(value, (np.ndarray, torch.Tensor))
        return False

    def _validate_document_schema(doc, sample_idx: int):
        if not isinstance(doc, dict):
            raise ValueError(f"NVOmni schema error: sample[{sample_idx}] is not dict: {type(doc)}")
        if "role" not in doc or "content" not in doc:
            raise ValueError(f"NVOmni schema error: sample[{sample_idx}] missing 'role' or 'content'")
        if not isinstance(doc["content"], list):
            raise ValueError(f"NVOmni schema error: sample[{sample_idx}].content is not list: {type(doc['content'])}")
        for item_idx, item in enumerate(doc["content"]):
            if not isinstance(item, dict):
                raise ValueError(
                    f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}] is not dict: {type(item)}"
                )
            if "type" not in item:
                raise ValueError(
                    f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}] missing 'type'"
                )
            item_type = item["type"]
            if item_type == "text":
                if not isinstance(item.get("text", None), str):
                    raise ValueError(
                        f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}].text must be str, got {type(item.get('text', None))}"
                    )
            elif item_type in {"image", "video", "audio"}:
                key = item_type
                if key not in item:
                    raise ValueError(
                        f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}] missing '{key}'"
                    )
                if not _is_allowed_media(item[key], item_type):
                    raise ValueError(
                        f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}].{key} "
                        f"has invalid type {type(item[key])}; allow str or supported objects"
                    )
            else:
                raise ValueError(
                    f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}] unsupported type '{item_type}'"
                )

    for i, doc_list in enumerate(documents):
        if not isinstance(doc_list, list) or len(doc_list) != 1:
            raise ValueError(f"NVOmni schema error: sample[{i}] must be single-message list, got {type(doc_list)} len={len(doc_list) if isinstance(doc_list, list) else 'n/a'}")
        _validate_document_schema(doc_list[0], i)

    # Align with official example: apply_chat_template + process_mm_info.
    try:
        from qwen_omni_utils import process_mm_info as _process_mm_info
    except Exception:
        def _process_mm_info(docs, use_audio_in_video=False):
            del use_audio_in_video
            _audios, _images, _videos = [], [], []
            for msg in docs:
                for part in msg.get("content", []):
                    ptype = part.get("type")
                    if ptype == "image":
                        _images.append(part.get("image", None))
                    elif ptype == "video":
                        _videos.append(part.get("video", None))
                    elif ptype == "audio":
                        _audios.append(part.get("audio", None))
            return _audios or None, _images or None, _videos or None

    def _apply_chat_template(doc):
        if hasattr(processor, "apply_chat_template"):
            return processor.apply_chat_template(doc, add_generation_prompt=False, tokenize=False)
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "apply_chat_template"):
            return processor.tokenizer.apply_chat_template(doc, add_generation_prompt=False, tokenize=False)
        raise AttributeError("Processor has no apply_chat_template; install a compatible AutoProcessor/tokenizer.")

    def _normalize_chat_text(value):
        if value is None:
            return " "
        if isinstance(value, list):
            return "".join([v if isinstance(v, str) else str(v) for v in value]) or " "
        if not isinstance(value, str):
            return str(value)
        return value

    documents_texts = [_normalize_chat_text(_apply_chat_template(doc)) for doc in documents]

    def _unwrap_single(x):
        if x is None:
            return None
        if isinstance(x, list) and len(x) == 1:
            return x[0]
        return x

    batch_audios, batch_images, batch_videos = [], [], []
    for doc in documents:
        a_i, i_i, v_i = _process_mm_info(doc, use_audio_in_video=False)
        batch_audios.append(_unwrap_single(a_i))
        batch_images.append(_unwrap_single(i_i))
        # Keep videos as-is (likely list of frames per sample).
        batch_videos.append(v_i)

    def _drop_if_any_none(items):
        if items is None:
            return None
        if len(items) == 0:
            return None
        if any(x is None for x in items):
            return None
        return items

    audios = _drop_if_any_none(batch_audios)
    images = _drop_if_any_none(batch_images)
    videos = _drop_if_any_none(batch_videos)

    # Qwen2.5 Omni image processor requires a uniform list shape:
    # either list[PIL] or list[list[PIL]] for the whole batch.
    if images is not None:
        if any(isinstance(x, list) for x in images):
            images = [x if isinstance(x, list) else [x] for x in images]

    # Text-only batches can be very long; cap to avoid OOM.
    if images is None and videos is None and audios is None:
        if max_length is None:
            text_max_length = 2048
        else:
            text_max_length = min(int(max_length), 2048)
    else:
        text_max_length = int(max_length) if max_length is not None else 204800

    text_kwargs = {
        "truncation": True,
        "padding": True,
        "max_length": text_max_length,
    }
    videos_kwargs = {
        "min_pixels": 32 * 14 * 14,
        "max_pixels": 64 * 28 * 28,
        "use_audio_in_video": False,
    }
    images_kwargs = {
        "min_pixels": 32 * 14 * 14,
        "max_pixels": 64 * 28 * 28,
    }
    audio_kwargs = {"max_length": 2048000}
    if audio_sample_rate is not None:
        audio_kwargs["sampling_rate"] = int(audio_sample_rate)

    mm_kwargs = {
        "return_tensors": "pt",
        "text_kwargs": text_kwargs,
        "videos_kwargs": videos_kwargs,
        "images_kwargs": images_kwargs,
    }
    if audios is not None:
        mm_kwargs["audio_kwargs"] = audio_kwargs
    if images is not None:
        mm_kwargs["images"] = images
    if videos is not None:
        mm_kwargs["videos"] = videos
    if audios is not None:
        mm_kwargs["audio"] = audios

    inputs = processor(text=documents_texts, **mm_kwargs)

    feats = inputs.get("input_features", None)
    fam = inputs.get("feature_attention_mask", None)
    if isinstance(feats, torch.Tensor) and feats.dim() == 3:
        # Expect (B, C=128, T) for qwen2_5_omni audio.
        if feats.shape[1] != 128 and feats.shape[2] == 128:
            feats = feats.transpose(1, 2)
        if fam is None:
            fam = torch.ones(feats.shape[0], feats.shape[2], dtype=torch.long)
        if isinstance(fam, torch.Tensor) and fam.dim() == 2:
            if fam.shape[1] != feats.shape[2]:
                min_len = min(fam.shape[1], feats.shape[2])
                feats = feats[:, :, :min_len]
                fam = fam[:, :min_len]
            inputs["input_features"] = feats
            inputs["feature_attention_mask"] = fam

    # Drop qwen2_5_omni-specific fields to avoid forward arg mismatch.
    inputs.pop("audio_attention_mask", None)
    inputs.pop("audio_feature_lengths", None)
    return inputs


def e5_v_prompt_template(text, add_video_token, add_image_token):
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
    if text is not None and add_video_token is False and add_image_token is False:  # only text
        prompt = llama3_template.format('{}\nSummary above sentence in one word: '.format(text))
    if text is None and add_video_token:  # only video
        prompt = llama3_template.format('<image>\nSummary above video in one word: ')
    if text is None and add_image_token:  # only image
        prompt = llama3_template.format('<image>\nSummary above image in one word: ')
    if text is not None and add_video_token:  # video + text
        prompt = llama3_template.format('<image>\n{}\nSummary above video and text in one word: '.format(text))
    if text is not None and add_image_token:
        prompt = llama3_template.format('<image>\n{}\nSummary above image and text in one word: '.format(text))

    return prompt


PROMPT_TEMPLATE_DICT = {
    "e5_v": e5_v_prompt_template,
}


def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    # Formulate input text based on text, special token and instruction.
    # TBD: Reorganize the hard-code part for baselines such as internvideo2
    if model_backbone == "internvideo2":
        return text
    elif model_backbone in [GME, LamRA, LamRA_QWEN2_5]:
        if text:
            return instruction + " " + text # GME and LamRA do not need special tokens
        else:
            return instruction + " "
    elif model_backbone == E5_V:
        return PROMPT_TEMPLATE_DICT[model_backbone](text, add_video_token, add_image_token)

    prompt = instruction
    if text:
        prompt = prompt + " " + text
    if add_video_token:
        video_token = VLM_VIDEO_TOKENS[model_backbone]
        prompt = video_token + " " + prompt
    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]
        prompt = image_token + " " + prompt

    return prompt


process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,
    LLAVA_NEXT: Llava_NEXT_process_fn,
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL: Qwen2_VL_process_fn,
    QWEN2_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
    QWEN2_5_OMNI: Omni_process_fn,
    NVOMNIEMBED: NVOmni_process_fn,
    INTERNVIDEO2: InternVideo2_process_fn,
    GME: Gme_process_fn,
    LamRA: Gme_process_fn,
    LamRA_QWEN2_5: Gme_process_fn,
    COLPALI: ColPali_process_fn,
    E5_V: Llava_NEXT_process_fn,
}
