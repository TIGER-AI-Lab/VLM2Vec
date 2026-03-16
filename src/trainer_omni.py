# train_omni_embed.py
# Standalone Omni embedding training (shared bi-encoder) with:
# - in-batch contrastive (InfoNCE)
# - DDP global negatives (best-effort; depends on src.loss DistributedContrastiveLoss)
# - freeze vision/audio + drop talker (save thinker-only)
# - multimodal mixed inputs (text/image/video/audio optional)
# - processor-based collator (pad/stack in collate-time)
#
# Run (single GPU):
#   python train_omni_embed.py
#
# Run (DDP):
#   torchrun --nproc_per_node=2 train_omni_embed.py

from __future__ import annotations

import os
import shutil
from functools import partial
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel, Trainer, HfArgumentParser
from transformers.trainer_utils import seed_worker
from transformers.utils import logging
import yaml

# Import from VLM2Vec project
from src.arguments import ModelArguments, DataArguments, TrainingArguments as VLMTrainingArguments
from src.loss_omni import InfoNCELoss, DDPInfoNCELoss, OmniTwoStageLoss
from src.data.collator.train_collator_omni import OmniAutoProcessorCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.processor import load_processor, get_backbone_name
from src.model.olm_backbone.qwen2_5_moni.qwen2_5omni_model_load import load_qwen2_5omni_thinker
from src.utils.dist_utils import ddp_all_gather_variable

logger = logging.get_logger(__name__)


# =========================
# DDP helpers
# =========================
def is_ddp_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

# =========================
# Model loading helpers
# =========================
def null_tp_plan(cfg):
    # avoid tp-plan issues in some remote code
    if hasattr(cfg, "base_model_tp_plan"):
        cfg.base_model_tp_plan = None
    if hasattr(cfg, "thinker_config") and cfg.thinker_config is not None:
        tc = cfg.thinker_config
        if hasattr(tc, "base_model_tp_plan"):
            tc.base_model_tp_plan = None
        if hasattr(tc, "text_config") and tc.text_config is not None and hasattr(tc.text_config, "base_model_tp_plan"):
            tc.text_config.base_model_tp_plan = None
    return cfg


# =========================
# Trainable stats
# =========================
def log_trainable_stats(model: nn.Module, topk: int = 200):
    total_params = 0
    trainable_params = 0
    trainable_names: List[str] = []
    for name, p in model.named_parameters():
        num = p.numel()
        total_params += num
        if p.requires_grad:
            trainable_params += num
            trainable_names.append(name)

    frozen_params = total_params - trainable_params
    frozen_ratio = (frozen_params / total_params) if total_params > 0 else 0.0

    print("Trainable params stats:")
    print(f"  total_params: {total_params}")
    print(f"  trainable_params: {trainable_params}")
    print(f"  frozen_ratio: {frozen_ratio:.4f}")
    print(f"  trainable_parameter_names (show up to {topk}):")
    for name in sorted(trainable_names)[:topk]:
        print(f"    {name}")
    if len(trainable_names) > topk:
        print(f"    ... ({len(trainable_names) - topk} more)")


# =========================
# Prefix-based freeze / save
# =========================
def set_trainable_for_embedding(
    model: nn.Module,
    train_thinker_model: bool = True,
    train_proj_adapters: bool = False,
):
    """
    You already confirmed real prefixes from safetensors index:
      - thinker.visual
      - thinker.audio_tower
      - thinker.model
      - thinker.lm_head
      - talker.*
    Policy (recommended):
      Freeze: thinker.visual.*, thinker.audio_tower.*, talker.*, thinker.lm_head.*
      Train : thinker.model.* (optionally also train proj/adapter)
    """
    # 1) freeze all by default
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 2) whitelist trainable parts
    if train_thinker_model:
        for name, p in model.named_parameters():
            if "talker." in name:
                continue
            if (
                "thinker.visual." in name
                or "thinker.audio_tower." in name
                or "thinker.lm_head." in name
                or "visual." in name
                or "audio_tower." in name
                or "lm_head." in name
            ):
                continue
            if "thinker.model." in name or "base_model.model." in name or name.startswith("model."):
                p.requires_grad = True
            elif name.startswith("base."):
                p.requires_grad = True

    # optional: if you have projection/adapters you want to tune (rarely needed if you only want LLM)
    if train_proj_adapters:
        proj_keys = ("proj", "project", "projection", "adapter", "lora", "pool")
        for name, p in model.named_parameters():
            if name.startswith("thinker.") and any(k in name.lower() for k in proj_keys):
                p.requires_grad = True

    # 3) hard-freeze heads & talker explicitly (for safety / readability)
    for name, p in model.named_parameters():
        if "talker." in name:
            p.requires_grad = False
        if "thinker.visual." in name or "visual." in name:
            p.requires_grad = False
        if "thinker.audio_tower." in name or "audio_tower." in name:
            p.requires_grad = False
        if "thinker.lm_head." in name or "lm_head." in name:
            p.requires_grad = False


def build_thinker_state_dict(full_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only thinker.* weights; drop talker.* entirely."""
    return {k: v for k, v in full_state_dict.items() if k.startswith("thinker.")}


# =========================
# Embedding encoder
# =========================
def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(attn_mask, torch.Tensor):
        attn_mask = torch.as_tensor(attn_mask, device=last_hidden.device)
    else:
        attn_mask = attn_mask.to(last_hidden.device)
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(attn_mask, torch.Tensor):
        attn_mask = torch.as_tensor(attn_mask, device=last_hidden.device)
    else:
        attn_mask = attn_mask.to(last_hidden.device)
    last_idx = attn_mask.long().sum(dim=1) - 1
    last_idx = last_idx.clamp_min(0)
    b_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
    return last_hidden[b_idx, last_idx]


class OmniEmbedder(nn.Module):
    """
    Omni multimodal inputs -> [B,D] embeddings (L2 normalized).
    Processor is attached later (encoder.processor = processor).
    """
    def __init__(
        self,
        model_name_or_path: str,
        model_type: Optional[str] = None,
        torch_dtype=torch.bfloat16,
        device_map=None,
        pooling: str = "mean",
        normalize: bool = True,
        # training policy
        train_proj_adapters: bool = False,
        # LoRA
        lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[str] = None,
        # full finetune (no freezing)
        full_finetune: bool = False,
    ):
        super().__init__()
        if lora and full_finetune:
            raise ValueError("LoRA and full_finetune cannot both be enabled.")
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        cfg = null_tp_plan(cfg)

        # best-effort: disable talker/audio output in config (does NOT remove parameters by itself)
        if hasattr(cfg, "enable_talker"):
            cfg.enable_talker = False
        if hasattr(cfg, "enable_audio_output"):
            cfg.enable_audio_output = False

        if model_type == "qwen2_5_omni":
            base_model = load_qwen2_5omni_thinker(
                model_name_or_path,
                config=cfg,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        else:
            base_model = AutoModel.from_pretrained(
                model_name_or_path,
                config=cfg,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

        if lora:
            from peft import LoraConfig, get_peft_model

            target_modules = lora_target_modules.split(",") if lora_target_modules else []
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False,
            )
            # Inject LoRA only into the text model (thinker.model), not vision/audio towers.
            if hasattr(base_model, "model") and base_model.model is not None:
                base_model.model = get_peft_model(base_model.model, lora_config)
            else:
                base_model = get_peft_model(base_model, lora_config)

        self.model = base_model

        if full_finetune:
            for _, p in self.model.named_parameters():
                p.requires_grad = True
        elif lora:
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            for name, p in self.model.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad = True
        else:
            # IMPORTANT: prefix-based freeze based on your confirmed prefixes
            set_trainable_for_embedding(
                self.model,
                train_thinker_model=True,
                train_proj_adapters=train_proj_adapters,
            )

        pooling = str(pooling).strip().lower()
        if pooling in {"lasttoken", "last_token", "eos"}:
            pooling = "last"
        if pooling not in {"mean", "last"}:
            raise ValueError(f"Unsupported pooling={pooling}. Expected one of: mean, last, lasttoken")
        self.pooling = pooling
        self.normalize = normalize

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _extract_hidden(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            return outputs[0]
        if isinstance(outputs, torch.Tensor):
            return outputs
        raise ValueError(f"Unknown output type: {type(outputs)}")

    def forward(self, **inputs) -> torch.Tensor:
        dev = self.device
        # Drop non-model keys from collator
        inputs.pop("text", None)
        inputs.pop("global_dataset_name", None)
        def _stack_with_fill(vals):
            if not isinstance(vals, list):
                return vals
            template = next((v for v in vals if v is not None), None)
            if template is None:
                return None
            if not isinstance(template, torch.Tensor):
                template = torch.as_tensor(template)
            stacked = []
            for v in vals:
                if v is None:
                    stacked.append(torch.zeros_like(template))
                else:
                    stacked.append(v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
            return torch.stack(stacked, dim=0)

        if "audio_attention_mask" in inputs and "feature_attention_mask" not in inputs:
            inputs["feature_attention_mask"] = inputs.pop("audio_attention_mask")

        for key in ("input_features", "feature_attention_mask", "audio_feature_lengths"):
            if key in inputs:
                inputs[key] = _stack_with_fill(inputs[key])

        # Some processor paths return batched numpy arrays (especially audio masks).
        # Convert them to tensors before shape checks / model forward.
        for k, v in list(inputs.items()):
            if isinstance(v, np.ndarray) and v.dtype != np.object_:
                inputs[k] = torch.as_tensor(v)

        if isinstance(inputs.get("input_features"), torch.Tensor) and isinstance(inputs.get("feature_attention_mask"), torch.Tensor):
            feat_t = inputs["input_features"].shape[-1]
            mask_t = inputs["feature_attention_mask"].shape[-1]
            if feat_t != mask_t:
                min_t = min(feat_t, mask_t)
                inputs["input_features"] = inputs["input_features"][..., :min_t]
                inputs["feature_attention_mask"] = inputs["feature_attention_mask"][:, :min_t]
            if inputs["input_features"].shape[-1] == 0:
                inputs.pop("input_features", None)
                inputs.pop("feature_attention_mask", None)
                inputs.pop("audio_feature_lengths", None)
            else:
                feats = inputs["input_features"]
                masks = inputs["feature_attention_mask"]
                audio_lens = masks.sum(dim=1)
                invalid_audio = audio_lens <= 0
                if torch.any(invalid_audio):
                    # Keep other samples intact: zero-out only invalid audio rows.
                    # Qwen2.5-Omni audio tower cannot handle zero-length rows mixed in batch,
                    # so we compact to valid audio rows after row-wise sanitization.
                    feats = feats.clone()
                    masks = masks.clone()
                    feats[invalid_audio] = 0
                    masks[invalid_audio] = 0
                    valid_audio_rows = torch.nonzero(masks.sum(dim=1) > 0, as_tuple=False).squeeze(1)
                    if valid_audio_rows.numel() == 0:
                        inputs.pop("input_features", None)
                        inputs.pop("feature_attention_mask", None)
                        inputs.pop("audio_feature_lengths", None)
                    else:
                        inputs["input_features"] = feats.index_select(0, valid_audio_rows)
                        inputs["feature_attention_mask"] = masks.index_select(0, valid_audio_rows)
                        afl = inputs.get("audio_feature_lengths", None)
                        if isinstance(afl, torch.Tensor):
                            if afl.dim() > 0 and afl.size(0) == masks.size(0):
                                inputs["audio_feature_lengths"] = afl.index_select(0, valid_audio_rows)
                            else:
                                inputs.pop("audio_feature_lengths", None)

        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(dev)

        if not hasattr(self, "_debug_fwd_once") and os.environ.get("VLM2VEC_DEBUG"):
            self._debug_fwd_once = True
            devs = {k: (v.device if isinstance(v, torch.Tensor) else type(v)) for k, v in inputs.items()}
            shapes = {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else None) for k, v in inputs.items()}
            print(f"[DEBUG] fwd devices: {devs}")
            print(f"[DEBUG] fwd shapes: {shapes}")

        # No generation; just hidden states
        try:
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        except TypeError:
            outputs = self.model(**inputs)

        if hasattr(outputs, "embeddings") and outputs.embeddings is not None:
            emb = outputs.embeddings
            return emb

        hs = self._extract_hidden(outputs)

        if hs.dim() == 2:
            emb = hs
        else:
            attn_mask = inputs.get("attention_mask", None)
            if attn_mask is None:
                attn_mask = torch.ones(hs.shape[:2], device=hs.device, dtype=torch.long)
            if self.pooling == "last":
                emb = last_token_pool(hs, attn_mask)
            else:
                emb = mean_pool(hs, attn_mask)

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


class OmniBiEncoder(nn.Module):
    """
    Shared bi-encoder: one encoder for qry & tgt.
    Loss is selected in Trainer after dist init.
    """
    def __init__(self, encoder: OmniEmbedder, temperature: float = 0.02, loss_fn: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        self.temperature = float(temperature)
        self.loss_fn = loss_fn

    @property
    def device(self):
        return self.encoder.device

    def encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.encoder(**batch)

    def forward(self, qry: Dict[str, Any], tgt: Dict[str, Any]) -> torch.Tensor:
        if self.loss_fn is None:
            raise RuntimeError("OmniBiEncoder.loss_fn is None. It should be set by OmniEmbedTrainer before training.")
        q = self.encode(qry)
        p = self.encode(tgt)
        return self.loss_fn(q, p)


# =========================
# Trainer
# =========================
class OmniEmbedTrainer(Trainer):
    """
    Changes vs vanilla Trainer:
      1) Choose loss AFTER dist init
      2) Override get_train_dataloader to avoid DataLoaderDispatcher wrapping issues
      3) Scale loss by world_size (matches MMEBTrainer pattern)
      4) Save thinker-only weights (drop talker) for embedding use-case
    """
    def __init__(self, *args, max_length=None, model_args=None, save_thinker_only: bool = True, **kwargs):
        self.max_length = max_length
        self.model_args = model_args
        self.save_thinker_only = save_thinker_only
        super().__init__(*args, **kwargs)

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

        temperature = getattr(self.model, "temperature", None)
        if temperature is None and self.model_args is not None:
            temperature = getattr(self.model_args, "temperature", 0.02)
        if temperature is None:
            temperature = 0.02

        self.loss_stage = str(getattr(self.args, "loss_stage", "infonce")).strip().lower()
        self.loss_alpha = float(getattr(self.args, "loss_alpha", 0.5))
        if self.loss_stage not in {"infonce", "jepa", "mixed"}:
            raise ValueError(f"Unsupported loss_stage={self.loss_stage}, expected one of infonce/jepa/mixed")

        loss_cls = DDPInfoNCELoss if (self.is_ddp and dist.get_world_size() > 1) else InfoNCELoss
        self.model.loss_fn = loss_cls(temperature=float(temperature), normalize=True)
        self.model.loss_stage = self.loss_stage

        if self.loss_stage in {"jepa", "mixed"}:
            emb_dim = self._infer_embed_dim(self.model)
            if emb_dim is None and self.model_args is not None:
                cfg_source = (
                    getattr(self.model_args, "checkpoint_path", None)
                    or getattr(self.model_args, "model_name", None)
                )
                emb_dim = self._infer_embed_dim_from_config_source(cfg_source)
            if emb_dim is None:
                raise ValueError(
                    "Cannot infer embedding dim for OmniTwoStageLoss. "
                    "Please ensure model config exposes hidden_size."
                )
            jepa_hidden = int(getattr(self.args, "jepa_predictor_hidden", 0))
            if jepa_hidden <= 0:
                jepa_hidden = None
            ref_param = next(self.model.parameters())
            self.model.two_stage_loss = OmniTwoStageLoss(
                emb_dim=int(emb_dim),
                infonce_temperature=float(temperature),
                use_ddp_infonce=(self.is_ddp and dist.get_world_size() > 1),
                jepa_predictor_hidden=jepa_hidden,
                normalize=True,
            ).to(device=ref_param.device, dtype=ref_param.dtype)

    @staticmethod
    def _infer_embed_dim(model) -> Optional[int]:
        # Try common hidden-size fields from wrapped backbone configs.
        cfg_candidates = []
        try:
            if hasattr(model, "encoder") and hasattr(model.encoder, "model") and hasattr(model.encoder.model, "config"):
                cfg_candidates.append(model.encoder.model.config)
        except Exception:
            pass
        try:
            if hasattr(model, "module") and hasattr(model.module, "encoder") and hasattr(model.module.encoder, "model") and hasattr(model.module.encoder.model, "config"):
                cfg_candidates.append(model.module.encoder.model.config)
        except Exception:
            pass

        for cfg in cfg_candidates:
            if cfg is None:
                continue

            # Top-level (e.g. talker hidden_size, or simple backbones)
            for attr in ("hidden_size", "d_model", "model_dim"):
                v = getattr(cfg, attr, None)
                if isinstance(v, int) and v > 0:
                    return v

            # Qwen2.5-Omni thinker hierarchy
            thinker_cfg = getattr(cfg, "thinker_config", None)
            if thinker_cfg is not None:
                for sub in (
                    thinker_cfg,
                    getattr(thinker_cfg, "text_config", None),
                    getattr(thinker_cfg, "model_config", None),
                    getattr(thinker_cfg, "vision_config", None),
                    getattr(thinker_cfg, "audio_config", None),
                ):
                    if sub is None:
                        continue
                    for attr in ("hidden_size", "d_model", "model_dim", "embed_dim", "output_dim"):
                        v = getattr(sub, attr, None)
                        if isinstance(v, int) and v > 0:
                            return v

            # Extra fallbacks for nested dict-like configs
            try:
                cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
            except Exception:
                cfg_dict = None
            if isinstance(cfg_dict, dict):
                for key_path in [
                    ("thinker_config", "text_config", "hidden_size"),
                    ("thinker_config", "vision_config", "embed_dim"),
                    ("thinker_config", "audio_config", "output_dim"),
                    ("hidden_size",),
                ]:
                    cur = cfg_dict
                    ok = True
                    for k in key_path:
                        if isinstance(cur, dict) and k in cur:
                            cur = cur[k]
                        else:
                            ok = False
                            break
                    if ok and isinstance(cur, int) and cur > 0:
                        return cur
        return None

    @classmethod
    def _infer_embed_dim_from_config_source(cls, config_source: Optional[str]) -> Optional[int]:
        if not isinstance(config_source, str) or not config_source:
            return None
        try:
            cfg = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
        except Exception:
            return None

        # Reuse the same inference logic by wrapping cfg in a minimal object shape.
        class _Wrap:
            pass

        wrapped = _Wrap()
        wrapped.encoder = _Wrap()
        wrapped.encoder.model = _Wrap()
        wrapped.encoder.model.config = cfg
        return cls._infer_embed_dim(wrapped)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        qry_batch, tgt_batch = inputs
        real_model = model.module if hasattr(model, "module") else model
        loss = None
        valid_mask = qry_batch.get("valid_example_mask", None)
        if valid_mask is None:
            valid_mask = tgt_batch.get("valid_example_mask", None)

        if valid_mask is not None:
            if isinstance(valid_mask, list):
                valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
            elif isinstance(valid_mask, torch.Tensor):
                valid_mask = valid_mask.bool()
            else:
                valid_mask = None

        if valid_mask is not None:
            if isinstance(valid_mask, torch.Tensor):
                valid_mask = valid_mask.to(next(real_model.parameters()).device)
            valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                loss = torch.tensor(0.0, device=next(real_model.parameters()).device)
            else:
                def _slice_batch(batch, idxs):
                    sliced = {}
                    for key, val in batch.items():
                        if key == "valid_example_mask":
                            continue
                        if isinstance(val, torch.Tensor) and val.size(0) == valid_mask.size(0):
                            sliced[key] = val.index_select(0, idxs)
                        elif isinstance(val, list) and len(val) == valid_mask.size(0):
                            sliced[key] = [val[i] for i in idxs.tolist()]
                        else:
                            sliced[key] = val
                    return sliced

                qry_batch = _slice_batch(qry_batch, valid_idx)
                tgt_batch = _slice_batch(tgt_batch, valid_idx)

        if loss is None:
            q_reps = real_model.encode(qry_batch)
            p_reps = real_model.encode(tgt_batch)
            if q_reps is None or p_reps is None:
                loss = torch.tensor(0.0, device=next(real_model.parameters()).device)
            else:
                stage = getattr(real_model, "loss_stage", "infonce")
                if stage == "infonce":
                    loss = real_model.loss_fn(q_reps, p_reps, reduction="mean")
                elif stage == "jepa":
                    out = real_model.two_stage_loss(stage="jepa", z_c=q_reps, z_t=p_reps, reduction="mean")
                    loss = out.loss
                elif stage == "mixed":
                    out = real_model.two_stage_loss(
                        stage="mixed",
                        z_c=q_reps,
                        z_t=p_reps,
                        q=q_reps,
                        d=p_reps,
                        alpha=self.loss_alpha,
                        reduction="mean",
                    )
                    loss = out.loss
                else:
                    raise ValueError(f"Unknown loss stage: {stage}")
        if not hasattr(self, "_debug_loss_once"):
            self._debug_loss_once = True
            q_ids = qry_batch.get("input_ids")
            p_ids = tgt_batch.get("input_ids")
            q_size = int(q_ids.shape[0]) if isinstance(q_ids, torch.Tensor) else 0
            p_size = int(p_ids.shape[0]) if isinstance(p_ids, torch.Tensor) else 0
            batch_task = "unknown"
            gd = qry_batch.get("global_dataset_name")
            if isinstance(gd, list) and gd:
                batch_task = gd[0]
            print("num_valid_labels", q_size)
            print("loss_full_precision", f"{loss.item():.12f}")
            print("batch_task", batch_task)
            print(
                "has_video",
                (qry_batch.get("pixel_values_videos") is not None) or (tgt_batch.get("pixel_values_videos") is not None),
                "has_audio",
                (qry_batch.get("input_features") is not None) or (tgt_batch.get("input_features") is not None),
            )
            print("num_pairs", q_size * p_size, "num_pos", min(q_size, p_size))
        if not hasattr(self, "_debug_step_counter"):
            self._debug_step_counter = 0
        self._debug_step_counter += 1
        if self._debug_step_counter % 100 == 0:
            q_ids = qry_batch.get("input_ids")
            p_ids = tgt_batch.get("input_ids")
            q_size = int(q_ids.shape[0]) if isinstance(q_ids, torch.Tensor) else 0
            p_size = int(p_ids.shape[0]) if isinstance(p_ids, torch.Tensor) else 0
            num_pairs = q_size * p_size
            empty_pairs = num_pairs == 0
            loss_fp = float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float(loss)

            try:
                q = real_model.encode(qry_batch)
                p = real_model.encode(tgt_batch)
                logits = torch.matmul(q, p.transpose(0, 1))
                top1 = (logits.argmax(dim=1) == torch.arange(logits.size(0), device=logits.device)).float().mean()
                cos_diag = torch.sum(F.normalize(q, dim=-1) * F.normalize(p, dim=-1), dim=-1)
                cos_stats = (
                    float(cos_diag.mean().item()),
                    float(cos_diag.min().item()),
                    float(cos_diag.max().item()),
                )
            except Exception:
                top1 = torch.tensor(float("nan"))
                cos_stats = (float("nan"), float("nan"), float("nan"))

            def _hash_tensor(t):
                if not isinstance(t, torch.Tensor):
                    return None
                return hash(t.detach().cpu().contiguous().numpy().tobytes())

            def _hash_text_list(texts):
                if not isinstance(texts, list):
                    return None
                flat = [str(t) for t in texts]
                return hash(tuple(flat))

            q_texts = qry_batch.get("text", None)
            p_texts = tgt_batch.get("text", None)
            q_text_hash = _hash_text_list(q_texts)
            p_text_hash = _hash_text_list(p_texts)

            print("loss_full_precision", f"{loss_fp:.12f}", "empty_pairs", empty_pairs)
            print("num_pairs", num_pairs, "effective_B", min(q_size, p_size), "top1_acc", float(top1))
            print("hash_text_q", q_text_hash, "hash_text_p", p_text_hash)
            print("hash_tokens_q", _hash_tensor(q_ids), "hash_tokens_p", _hash_tensor(p_ids))
            print("cos_qp_mean/min/max", f"{cos_stats[0]:.6f}", f"{cos_stats[1]:.6f}", f"{cos_stats[2]:.6f}")
        if isinstance(loss, torch.Tensor):
            device = next(real_model.parameters()).device
            if loss.device != device:
                loss = loss.to(device)
        loss = loss / float(1.0)
        return (loss, None) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Ensure we do NOT drop columns (tuple/dicts batches)
        # Prefer setting training_args.remove_unused_columns=False outside.
        try:
            if getattr(self.args, "remove_unused_columns", None):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
        except Exception:
            pass

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        else:
            dataloader_params["sampler"] = None
            dataloader_params["shuffle"] = False
            dataloader_params["drop_last"] = True
            dataloader_params["prefetch_factor"] = None

        return DataLoader(train_dataset, **dataloader_params)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        enc: OmniEmbedder = self.model.encoder
        model_name_or_path = getattr(self.model_args, "model_name", None) if self.model_args is not None else None
        if self.model_args is not None and getattr(self.model_args, "lora", False):
            try:
                from peft import PeftModel
            except Exception as e:
                raise RuntimeError("LoRA is enabled but PEFT is not available.") from e

            peft_model = None
            peft_source = None
            candidates = [
                ("enc.model.base_model", getattr(enc.model, "base_model", None)),
                ("enc.model.model", getattr(enc.model, "model", None)),
            ]
            base_model = getattr(enc.model, "base_model", None)
            if base_model is not None:
                candidates.append(("enc.model.base_model.model", getattr(base_model, "model", None)))
            for name, m in candidates:
                if m is None:
                    continue
                if isinstance(m, PeftModel) or hasattr(m, "peft_config"):
                    peft_model = m
                    peft_source = name
                    break

            if peft_model is None:
                raise RuntimeError("LoRA is enabled but no PEFT adapter was found on enc.model.")

            logger.info(f"Saving LoRA adapter from {peft_source}.")
            peft_model.save_pretrained(output_dir, safe_serialization=True)
            adapter_cfg = os.path.join(output_dir, "adapter_config.json")
            adapter_weights = os.path.join(output_dir, "adapter_model.safetensors")
            if not (os.path.exists(adapter_cfg) and os.path.exists(adapter_weights)):
                raise RuntimeError(
                    "LoRA save failed: adapter_config.json or adapter_model.safetensors is missing. "
                    "LoRA may not be injected or output_dir is incorrect."
                )

            try:
                if hasattr(enc, "processor") and enc.processor is not None:
                    enc.processor.save_pretrained(output_dir)
            except Exception:
                pass

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if self._should_export_full_dir(output_dir):
                self._export_hf_full_dir(
                    output_dir=output_dir,
                    enc=enc,
                    model_name_or_path=model_name_or_path,
                    adapter_dir=output_dir,
                )
            return

        if state_dict is None:
            state_dict = enc.model.state_dict()

        if self.save_thinker_only:
            # drop talker entirely
            state_dict = build_thinker_state_dict(state_dict)
            # Save as a plain checkpoint (safe for any remote code)
            torch.save(state_dict, os.path.join(output_dir, "thinker_only.bin"))
        else:
            # fallback: save whole model
            try:
                enc.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
            except Exception:
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        # Save processor
        try:
            if hasattr(enc, "processor") and enc.processor is not None:
                enc.processor.save_pretrained(output_dir)
        except Exception:
            pass

        # Save config.json (useful for reconstruction)
        try:
            if hasattr(enc.model, "config") and enc.model.config is not None:
                enc.model.config.to_json_file(os.path.join(output_dir, "config.json"))
        except Exception:
            pass

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        if self._should_export_full_dir(output_dir):
            self._export_hf_full_dir(
                output_dir=output_dir,
                enc=enc,
                model_name_or_path=model_name_or_path,
                adapter_dir=None,
            )

    def _should_export_full_dir(self, output_dir: str) -> bool:
        # Export on final save, and optionally once at a specific checkpoint for quick validation.
        if output_dir == self.args.output_dir:
            return True
        export_step = int(getattr(self.args, "export_full_checkpoint", 0) or 0)
        if export_step <= 0:
            return False
        return os.path.basename(output_dir) == f"checkpoint-{export_step}"

    def _export_hf_full_dir(
        self,
        output_dir: str,
        enc: OmniEmbedder,
        model_name_or_path: Optional[str],
        adapter_dir: Optional[str],
    ):
        if not model_name_or_path or not os.path.isdir(model_name_or_path):
            logger.info("Skip HF export: base model directory is missing.")
            return

        def _copy_if_exists(fname: str):
            src = os.path.join(model_name_or_path, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        # Copy base tokenizer/processor/config files if present.
        copy_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "added_tokens.json",
            "merges.txt",
            "vocab.json",
            "chat_template.jinja",
            "generation_config.json",
            "README.md",
            "LICENSE",
            ".gitattributes",
        ]
        for fname in copy_files:
            _copy_if_exists(fname)

        model_backbone = getattr(self.model_args, "model_backbone", None) if self.model_args is not None else None
        model_type = getattr(self.model_args, "model_type", None) if self.model_args is not None else None
        is_qwen2_5_omni = model_backbone == "qwen2_5_omni" or model_type == "qwen2_5_omni"

        if is_qwen2_5_omni:
            from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

            with torch.no_grad():
                full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="cpu",
                )

                if adapter_dir:
                    from peft import LoraConfig, PeftModel

                    lora_config = LoraConfig.from_pretrained(adapter_dir)
                    peft_model = PeftModel.from_pretrained(
                        full_model.thinker.model,
                        adapter_dir,
                        config=lora_config,
                        is_trainable=False,
                    )
                    merged = peft_model.merge_and_unload()
                    full_model.thinker.model = merged
                else:
                    src_thinker = getattr(enc.model, "base_model", None) or enc.model
                    full_model.thinker.load_state_dict(src_thinker.state_dict(), strict=False)

                full_model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
        else:
            with torch.no_grad():
                try:
                    enc.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
                except Exception:
                    torch.save(enc.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))


# =========================
# Main
# =========================
def main():
    # Delegate to canonical entrypoint to avoid dual-main drift.
    from train_omni import main as canonical_main
    return canonical_main()


if __name__ == "__main__":
    main()
