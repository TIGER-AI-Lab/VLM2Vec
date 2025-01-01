import functools

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from typing import Optional
import os
import torch

from src.collator import process_vlm_inputs, split_vlm_inputs, get_dense_rep, split_and_process_vlm_inputs
from src.loss import SimpleContrastiveLoss, DistributedContrastiveLoss
from grad_cache.grad_cache import GradCache

class MMEBTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MMEBTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self.processor = args.processor
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, *args, **kwargs):
        qry_inputs, tgt_inputs = inputs
        return model(qry=qry_inputs, tgt=tgt_inputs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


class GradCacheTrainer(Trainer):
    """
    Adapted from gradcache repo.
    """
    def __init__(self, *args, **kwargs):
        super(GradCacheTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        loss_fn_cls = DistributedContrastiveLoss if self.is_ddp else SimpleContrastiveLoss
        loss_fn = loss_fn_cls(temperature=self.model.temperature)

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_vlm_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        model.train()
        queries, passages = inputs
        queries, passages = {'qry': queries}, {'tgt': passages}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        print(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.encoder.config.to_json_file(os.path.join(output_dir, 'config.json'))


class GradCacheLateProcessTrainer(Trainer):
    """
    Adapted from gradcache repo.
    """
    def __init__(self, *args, **kwargs):
        self.max_length = kwargs.get("max_length", 512)
        if "max_length" in kwargs:
            del kwargs["max_length"]
        super(GradCacheLateProcessTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        loss_fn_cls = DistributedContrastiveLoss if self.is_ddp else SimpleContrastiveLoss
        loss_fn = loss_fn_cls(temperature=self.model.temperature)

        process_fn = functools.partial(process_vlm_inputs, processor=self.processing_class,
                                       backbone_name=self.args.model_backbone, max_length=self.max_length)

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_and_process_vlm_inputs,
            process_fn=process_fn,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        model.train()
        queries, targets = inputs
        queries, targets = {'qry': queries}, {'tgt': targets}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, targets, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        print(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.encoder.config.to_json_file(os.path.join(output_dir, 'config.json'))
