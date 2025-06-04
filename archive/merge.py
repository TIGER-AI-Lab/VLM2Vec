from src.arguments import ModelArguments
from src.model import MMEBModel


def main():
    model_args = ModelArguments(
        model_name='Qwen/Qwen2-VL-2B-Instruct',
        checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-2B',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )

    model = MMEBModel.load(model_args)
    model.encoder._hf_peft_config_loaded = False
    model.encoder.save_pretrained('full_model/', safe_serialization=False)


if __name__ == "__main__":
    main()
