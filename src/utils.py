import torch
from src.logging import get_logger
logger = get_logger(__name__)


def load_processor(model_args):
    """
    Load processor based on VLM backbone.
    """
    print_master('Loading processor')
    if model_args.model_backbone == PHI3V:
        from src.vlm_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
    elif model_args.model_backbone == LLAVA_NEXT:
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            trust_remote_code=True
        )
    elif model_args.model_backbone == QWEN2_VL:
        from src.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast

        model_name = model_args.processor_name if model_args.processor_name else model_args.model_name
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name,
            image_processor=image_processor,
            tokenizer=tokenizer,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    processor.tokenizer.padding_side = "right"
    return processor

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
MODEL2BACKBONE = {
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())
def get_backbone_name(hf_config):
    assert hf_config.model_type in SUPPORTED_MODELS, f"Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def print_rank(message):
    """If distributed is initialized, print the rank."""
    if torch.distributed.is_initialized():
        logger.info(f'rank{torch.distributed.get_rank()}: ' + message)
    else:
        logger.info(message)


def print_master(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)
