# Adapted from Tevatron code
import logging
import sys
import torch
import wandb

from transformers import (
    HfArgumentParser,
)

from src.dataset import TrainTextImageDataset
from src.collator import TrainTextImageDataCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.trainer import GradCacheLateProcessTrainer
from src.utils import print_rank
from src.model_utils import load_processor, get_backbone_name


logger = logging.getLogger(__name__)


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

    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")

    model = MMEBModel.build(model_args, training_args)
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args)
    setattr(model, 'processor', processor)

    train_dataset = TrainTextImageDataset(data_args, model_args)
    collator = TrainTextImageDataCollator(data_args, model_args, processor)

    trainer_cls = GradCacheLateProcessTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        max_length=data_args.max_len
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
