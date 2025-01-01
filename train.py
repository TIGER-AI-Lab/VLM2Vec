# Adapted from Tevatron code
import logging
import sys

from transformers import (
    HfArgumentParser,
)

from src import utils
from src.dataset import TrainDataset
from src.collator import DeprecatedTrainCollator, TrainRawInputCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.trainer import MMEBTrainer, GradCacheTrainer, GradCacheLateProcessTrainer
from src.utils import load_processor, print_rank
import wandb
import torch
import torch.distributed as dist


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

    # if (dist.is_initialized() and torch.distributed.get_rank() == 0) or (not dist.is_initialized()):
    #     print_rank('init wandb')
    #     wandb.init(project=training_args.project_name, name=training_args.run_name, mode="disabled")

    model = MMEBModel.build(model_args, training_args)
    model_backbone = utils.get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args)
    setattr(model, 'processor', processor)
    train_dataset = TrainDataset(data_args, model_args)

    # collator = TrainCollator(data_args, model_args, processor)
    # trainer_cls = GradCacheTrainer if training_args.grad_cache else MMEBTrainer

    collator = TrainRawInputCollator(data_args, model_args, processor)
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
