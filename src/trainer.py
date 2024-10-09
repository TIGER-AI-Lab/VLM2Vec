from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from typing import Optional
import os
import torch
from src.loss import SimpleContrastiveLoss, DistributedContrastiveLoss
from itertools import repeat
from grad_cache.grad_cache import GradCache


MAX_INPUT_ID = int(1e9)

class MMEBTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MMEBTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs):
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


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def split_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]
    keys = list(arg_val.keys())

    # for input_ids and attention_mask, split directly
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in ["input_ids", "attention_mask"]]

    # for pixel_values and image_sizes, need to split based on the position of images
    input_ids = arg_val["input_ids"]
    positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=True)
    num_chunks = len(chunked_tensors[0])
    row_contain_image = torch.unique(positions[0])  # indicates which row in input_ids contain images
    chunk_image_count = []
    for chunk_idx in range(num_chunks):
        chunk_image_count.append(torch.sum(
            (row_contain_image >= chunk_idx * chunk_size) & (row_contain_image < (chunk_idx + 1) * chunk_size)).item())
    if "pixel_values" in keys:
        pixel_values = arg_val["pixel_values"]
        image_sizes = arg_val["image_sizes"]
        chunked_tensors.append(torch.split(pixel_values, chunk_image_count))
        chunked_tensors.append(torch.split(image_sizes, chunk_image_count))

    chunked_arg_val = []
    for kk, tt in zip(repeat(keys), zip(*chunked_tensors)):
        if "pixel_values" in keys and tt[2].numel() == 0:  # this chunk doesn't contain image
            chunked_arg_val.append(dict(zip(kk[:2], tt[:2])))
        else:
            chunked_arg_val.append(dict(zip(kk, tt)))

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    """
    Get either qry_reps or tgt_reps.
    """
    if x["qry_reps"] is None:
        return x["tgt_reps"]
    else:
        return x["qry_reps"]


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

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        queries, passages = inputs
        queries, passages = {'qry': queries}, {'tgt': passages}

        if self.args.local_rank == 0:
            print(f"q.shape={queries['qry']['input_ids'].shape}")
            print(f"d.shape={passages['tgt']['input_ids'].shape}")
            if 'pixel_values' in queries['qry']:
                print(f"q_img.shape={queries['qry']['pixel_values'].shape}")
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
