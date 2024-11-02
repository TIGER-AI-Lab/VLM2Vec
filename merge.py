from src.arguments import ModelArguments
from transformers import HfArgumentParser, AutoProcessor

from src.model import MMEBModel
from evaluation.eval_utils import get_pred


def main():
    parser = HfArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses()
    model_args: ModelArguments

    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

    processor.tokenizer.padding_side = "right"
    model = MMEBModel.load(model_args)
    model.encoder._hf_peft_config_loaded = False
    model.encoder.save_pretrained('full_model/', safe_serialization=False)


if __name__ == "__main__":
    main()