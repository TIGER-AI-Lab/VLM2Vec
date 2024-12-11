def load_processor(model_args):
    """
    Load processor based on VLM backbone.
    """
    if model_args.model_backbone == "phi":
        from src.vlm_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
    elif model_args.model_backbone == "llava":
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            trust_remote_code=True
        )
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    processor.tokenizer.padding_side = "right"
    return processor
