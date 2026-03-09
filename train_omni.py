import logging
import os.path
import sys

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

import torch
import wandb
import yaml
from transformers import HfArgumentParser

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.train_collator_omni import OmniAutoProcessorCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.processor import load_processor, get_backbone_name
from src.trainer_omni import OmniEmbedder, OmniBiEncoder, OmniEmbedTrainer, log_trainable_stats
from src.utils.basic_utils import print_rank, find_latest_checkpoint


def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # Initialize distributed/device state early to avoid AcceleratorState reset after Accelerator creation.
    _ = training_args._setup_devices

    # DEBUG PRINTS for Distributed Setup
    print("Distributed init debug info:")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

    if torch.distributed.is_available():
        print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
        if torch.distributed.is_initialized():
            print(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")
            print(f"torch.distributed.get_world_size(): {torch.distributed.get_world_size()}")

    # Check for existing checkpoints
    if training_args.resume_from == 'auto':
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    else:
        resume_checkpoint_dir = None
        logger.info("No checkpoint found. Starting fresh training.")

    # Initialize WandB if enabled
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)

    encoder = OmniEmbedder(
        model_name_or_path=model_args.model_name,
        model_type=model_args.model_type,
        torch_dtype=torch.bfloat16,
        device_map=None,
        pooling="mean",
        normalize=True,
        train_proj_adapters=False,
        lora=getattr(model_args, "lora", False),
        lora_r=getattr(model_args, "lora_r", 16),
        lora_alpha=getattr(model_args, "lora_alpha", 64),
        lora_dropout=getattr(model_args, "lora_dropout", 0.1),
        lora_target_modules=getattr(model_args, "lora_target_modules", None),
        full_finetune=getattr(model_args, "full_finetune", False),
    )
    model_backbone = get_backbone_name(hf_config=encoder.model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')

    processor = load_processor(model_args, data_args)
    setattr(encoder, 'processor', processor)

    model = OmniBiEncoder(encoder=encoder, temperature=model_args.temperature, loss_fn=None)
    if training_args.device is not None:
        model = model.to(training_args.device)
    log_trainable_stats(model)

    def _resolve_dataset_path(raw_path: str, data_basedir: str = None) -> str:
        if not isinstance(raw_path, str) or not raw_path:
            return raw_path
        resolved = os.path.expanduser(os.path.expandvars(raw_path))
        if data_basedir and not os.path.isabs(resolved):
            return os.path.join(data_basedir, resolved)
        return resolved

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)

        data_basedir = (
            os.path.expanduser(os.path.expandvars(data_args.data_basedir))
            if data_args.data_basedir else None
        )

        # Resolve path-like fields so one config can migrate across machines:
        # use --data_basedir as a common dataset root for relative paths.
        path_keys = {"data_path", "image_dir", "frame_root", "audio_root", "video_root"}
        for _, task_config in dataset_config.items():
            if not isinstance(task_config, dict):
                continue
            for key in path_keys:
                if key in task_config and isinstance(task_config[key], str):
                    task_config[key] = _resolve_dataset_path(task_config[key], data_basedir=data_basedir)

        train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)

    train_collator = OmniAutoProcessorCollator(
        processor=processor,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
    )

    trainer = OmniEmbedTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        max_length=data_args.max_len,
        save_thinker_only=True,
    )
    train_dataset.trainer = trainer

    trainer.train(resume_from_checkpoint=resume_checkpoint_dir)
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
