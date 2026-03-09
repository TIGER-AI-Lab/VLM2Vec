"""
VLM2Vec adapted OmniEmbed model.
Extends the original NVOmniEmbedModel to handle list-format inputs from Qwen2_VL_process_fn.
"""

import torch
import numpy as np
from transformers import AutoModel, AutoConfig
from typing import Optional, List, Union


class OmniEmbedForConditionalGeneration(torch.nn.Module):
    """
    Wrapper for NVOmniEmbedModel that handles list-format visual inputs.

    Debug-first version:
    - For list pixel_values / pixel_values_videos, DO NOT cat patches (it destroys batch boundaries)
      Use stack+pad to preserve alignment and avoid constant embeddings.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # 打印 base_model 的 padding_side 和 config.padding_side（执行前）
        print(f"Base model padding_side (before): {getattr(self.base_model, 'padding_side', 'Not set')}")
        print(f"Base model config.padding_side (before): {getattr(getattr(self.base_model, 'config', None), 'padding_side', 'Not set')}")

        # Expose text encoder for text-only inference (memory optimization)
        # In omni-embed-nemotron-3b this is BidirectQwen2_5OmniThinkerTextModel
        self.model = base_model.model

        # Patch upstream rotary dtype mismatch without touching site-packages
        self._maybe_patch_rotary_dtype()

        # Ensure the underlying omni models use left padding internally
        self._force_left_padding(self.base_model)
        self._force_left_padding(getattr(self.base_model, "model", None))

        # 打印 base_model 的 padding_side 和 config.padding_side（执行后）
        print(f"Base model padding_side (after): {getattr(self.base_model, 'padding_side', 'Not set')}")
        print(f"Base model config.padding_side (after): {getattr(getattr(self.base_model, 'config', None), 'padding_side', 'Not set')}")
        if hasattr(self.base_model, 'model'):
            print(f"Base model.model padding_side (after): {getattr(self.base_model.model, 'padding_side', 'Not set')}")
            print(f"Base model.model config.padding_side (after): {getattr(getattr(self.base_model.model, 'config', None), 'padding_side', 'Not set')}")

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """Load the base model and wrap it."""
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if getattr(cfg, "model_type", None) == "qwen2_5_omni":
            from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

            full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
            base_model = full_model.thinker if hasattr(full_model, "thinker") else full_model
        else:
            base_model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        return cls(base_model)

    @staticmethod
    def _stack_list_or_none(
        values: List,
        to_tensor_dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """
        Convert a list[Tensor/np.ndarray/None] to a stacked Tensor [B, ...].
        Pads dim0 to max_len, fills None with zeros_like(template).
        Returns None if all entries are None.
        """
        if not isinstance(values, list):
            return values

        if all(v is None for v in values):
            return None

        template = next(v for v in values if v is not None)
        if isinstance(template, np.ndarray):
            template = torch.from_numpy(template)
        elif not isinstance(template, torch.Tensor):
            template = torch.as_tensor(template)

        tensors = []
        max_len = template.shape[0] if template.dim() > 0 else 1
        tail_shape = template.shape[1:] if template.dim() > 1 else ()

        for v in values:
            if v is None:
                tensors.append(None)
                continue
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            elif not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            cur_len = v.shape[0] if v.dim() > 0 else 1
            max_len = max(max_len, cur_len)
            tensors.append(v)

        # verify tail shape consistency when possible
        for t in tensors:
            if t is None:
                continue
            if t.dim() > 1 and tail_shape and t.shape[1:] != tail_shape:
                raise ValueError(f"Inconsistent tensor tail shapes: {t.shape[1:]} vs {tail_shape}")
            if not tail_shape and t.dim() > 1:
                tail_shape = t.shape[1:]

        def _pad_to_max(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 0:
                x = x.view(1)
            cur_len = x.shape[0]
            if cur_len == max_len:
                return x
            pad_shape = (max_len - cur_len,) + tuple(x.shape[1:])
            pad_tensor = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad_tensor], dim=0)

        stacked = torch.stack(
            [
                torch.zeros((max_len,) + tail_shape, dtype=template.dtype)
                if t is None
                else _pad_to_max(t)
                for t in tensors
            ],
            dim=0,
        )

        if to_tensor_dtype is not None:
            stacked = stacked.to(dtype=to_tensor_dtype)
        return stacked

    @staticmethod
    def _cat_patches(values: List) -> Optional[torch.Tensor]:
        """
        Concatenate per-sample patch tensors along dim=0 (variable lengths).
        Each entry must have shape [num_patches, hidden_dim].
        Returns None if all entries are None.
        """
        if not isinstance(values, list):
            return values
        if all(v is None for v in values):
            return None

        tensors = []
        hidden_dim = None
        for v in values:
            if v is None:
                continue
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            elif not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            if v.dim() != 2:
                raise ValueError(f"Expected 2D patch tensor, got shape {tuple(v.shape)}")
            if hidden_dim is None:
                hidden_dim = v.shape[1]
            elif v.shape[1] != hidden_dim:
                raise ValueError(f"Patch hidden dim mismatch: {v.shape[1]} vs {hidden_dim}")
            tensors.append(v)

        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _normalize_video_patches(values: List) -> Optional[torch.Tensor]:
        """
        (Kept for future use; debug version does not call it.)
        Normalize video inputs that may be list-of-frames per sample.
        """
        if not isinstance(values, list):
            return values
        if all(v is None for v in values):
            return None

        per_video = []
        for v in values:
            if v is None:
                continue
            if isinstance(v, list):
                v = OmniEmbedForConditionalGeneration._cat_patches(v)
            elif isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            elif not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            per_video.append(v)

        if not per_video:
            return None
        return OmniEmbedForConditionalGeneration._cat_patches(per_video)

    @staticmethod
    def _stack_grid_thw(values: List) -> Optional[torch.Tensor]:
        """
        Stack image/video grid_thw list into shape [B, 3].
        Accepts entries shaped (3,), (1,3), or tensors/arrays with 3 elements.
        """
        if not isinstance(values, list):
            return values
        if all(v is None for v in values):
            return None

        tensors = []
        for v in values:
            if v is None:
                tensors.append(None)
                continue
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            elif not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            v = v.reshape(-1)
            if v.numel() != 3:
                raise ValueError(f"Expected grid_thw with 3 elements, got shape {tuple(v.shape)} and numel {v.numel()}")
            tensors.append(v)

        template = next(t for t in tensors if t is not None)
        stacked = torch.stack([torch.zeros_like(template) if t is None else t for t in tensors], dim=0)
        return stacked

    @staticmethod
    def _force_left_padding(m):
        """Force left padding to align with omni flash-attn assumptions."""
        if m is None:
            return
        if hasattr(m, "padding_side"):
            m.padding_side = "left"
        if hasattr(m, "config") and hasattr(m.config, "padding_side"):
            m.config.padding_side = "left"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[Union[torch.Tensor, List]] = None,
        image_grid_thw: Optional[Union[torch.Tensor, List]] = None,
        pixel_values_videos: Optional[Union[torch.Tensor, List]] = None,
        video_grid_thw: Optional[Union[torch.Tensor, List]] = None,
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        # Get device and dtype from model
        device = input_ids.device if input_ids is not None else next(self.base_model.parameters()).device
        base_dtype = next(self.base_model.parameters()).dtype  # usually bf16
        visual_dtype = base_dtype

        # --------- Process pixel_values (images) ---------
        if pixel_values is not None:
            if isinstance(pixel_values, list):
                # Keep grid_thw alignment by concatenating per-sample patch sequences.
                pixel_values = self._cat_patches(pixel_values)
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(device=device, dtype=visual_dtype)

        # --------- Process image_grid_thw (image grid) ---------
        if image_grid_thw is not None:
            if isinstance(image_grid_thw, list):
                image_grid_thw = self._stack_grid_thw(image_grid_thw)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

        # --------- Process pixel_values_videos (video frames) ---------
        if pixel_values_videos is not None:
            if isinstance(pixel_values_videos, list):
                # Keep grid_thw alignment by concatenating per-sample patch sequences.
                pixel_values_videos = self._cat_patches(pixel_values_videos)
            if isinstance(pixel_values_videos, torch.Tensor):
                pixel_values_videos = pixel_values_videos.to(device=device, dtype=visual_dtype)

        # --------- Process video_grid_thw (video grid) ---------
        if video_grid_thw is not None:
            if isinstance(video_grid_thw, list):
                video_grid_thw = self._stack_grid_thw(video_grid_thw)
            if video_grid_thw is not None:
                video_grid_thw = video_grid_thw.to(device)

        # --------- Process audio inputs ---------
        if input_features is None and audio_values is not None:
            input_features = audio_values
        if feature_attention_mask is None and audio_attention_mask is not None:
            feature_attention_mask = audio_attention_mask

        if input_features is not None:
            input_features = input_features.to(device=device, dtype=visual_dtype)
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(device)
        if audio_feature_lengths is not None:
            audio_feature_lengths = audio_feature_lengths.to(device)

        # --------- Audio sanity guard ---------
        # Keep audio tensor/mask consistent and prevent zero-length rows from
        # crashing Qwen2.5-Omni audio avg_pool1d.
        if isinstance(input_features, torch.Tensor) and input_features.numel() == 0:
            input_features = None
            feature_attention_mask = None
            audio_feature_lengths = None
        elif isinstance(input_features, torch.Tensor):
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)
            if input_features.dim() == 3 and input_features.shape[0] == 0:
                input_features = None
                feature_attention_mask = None
                audio_feature_lengths = None
            elif input_features.dim() == 3:
                bsz, _, feat_t = input_features.shape
                if not isinstance(feature_attention_mask, torch.Tensor):
                    feature_attention_mask = torch.ones((bsz, feat_t), dtype=torch.long, device=device)
                else:
                    if feature_attention_mask.dim() == 1:
                        feature_attention_mask = feature_attention_mask.unsqueeze(0)
                    if feature_attention_mask.dim() != 2:
                        feature_attention_mask = feature_attention_mask.reshape(feature_attention_mask.shape[0], -1)

                    if feature_attention_mask.shape[0] != bsz:
                        min_b = min(bsz, feature_attention_mask.shape[0])
                        input_features = input_features[:min_b]
                        feature_attention_mask = feature_attention_mask[:min_b]
                        bsz = min_b

                    mask_t = feature_attention_mask.shape[-1]
                    if feat_t != mask_t:
                        if feat_t == 0 or mask_t == 0:
                            input_features = input_features.new_zeros((bsz, input_features.shape[1], 1))
                            feature_attention_mask = feature_attention_mask.new_zeros((bsz, 1), dtype=torch.long)
                        else:
                            min_t = min(feat_t, mask_t)
                            input_features = input_features[..., :min_t]
                            feature_attention_mask = feature_attention_mask[:, :min_t]

                if isinstance(feature_attention_mask, torch.Tensor):
                    feature_attention_mask = feature_attention_mask.long()
                    if feature_attention_mask.shape[-1] == 0:
                        input_features = input_features.new_zeros((input_features.shape[0], input_features.shape[1], 1))
                        feature_attention_mask = feature_attention_mask.new_zeros((feature_attention_mask.shape[0], 1), dtype=torch.long)

                    audio_lens = feature_attention_mask.sum(dim=1)
                    invalid_audio = audio_lens <= 0
                    if torch.any(invalid_audio):
                        input_features = input_features.clone()
                        feature_attention_mask = feature_attention_mask.clone()
                        input_features[invalid_audio, :, 0] = 0
                        feature_attention_mask[invalid_audio, 0] = 1

                    audio_feature_lengths = feature_attention_mask.sum(dim=1).long()

        # --------- Modality presence check (text-only is allowed) ---------
        has_visual = (pixel_values is not None) or (pixel_values_videos is not None)
        has_audio = (input_features is not None)
        has_text = (input_ids is not None)
        
        # ✅ 支持纯文本：只要有 input_ids 就可以，视觉和音频是可选的
        if not has_text:
            raise ValueError(
                f"OmniEmbed.forward: input_ids (text) must be provided.\n"
                f"  input_ids: {type(input_ids)}\n"
                f"Text is required; visual/audio are optional."
            )

        # --------- Debug: prevent silent all-zero inputs (only check if modality is present) ---------
        def _is_all_zero(x) -> bool:
            return isinstance(x, torch.Tensor) and x.numel() > 0 and x.abs().max().item() == 0

        # 只有当模态存在时才检查全0（避免在纯文本场景报错）
        if has_visual and _is_all_zero(pixel_values):
            raise ValueError("[OmniEmbed.forward] pixel_values is ALL-ZERO. Image loading/processor likely failed.")
        if has_visual and _is_all_zero(pixel_values_videos):
            raise ValueError("[OmniEmbed.forward] pixel_values_videos is ALL-ZERO. Video loading/processor likely failed.")
        if has_audio and _is_all_zero(input_features):
            raise ValueError("[OmniEmbed.forward] input_features is ALL-ZERO. Audio loading/processor likely failed.")

        # --------- Call base model ---------
        autocast_enabled = device.type == "cuda" and visual_dtype in (torch.float16, torch.bfloat16)
        if autocast_enabled:
            with torch.autocast(device_type=device.type, dtype=visual_dtype):
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    audio_feature_lengths=audio_feature_lengths,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )
        else:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

    def _maybe_patch_rotary_dtype(self):
        """
        Qwen2.5 Omni vision flash-attn path casts q/k to float32 while freqs stay bf16,
        leading to flash_attn dtype mismatch. Monkey-patch helper to align dtypes.
        """
        try:
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen_omni
            from flash_attn.layers.rotary import apply_rotary_emb
        except Exception:
            return

        cls = getattr(qwen_omni, "Qwen2_5OmniVisionFlashAttention2", None)
        if cls is None:
            return

        if getattr(cls._apply_rotary_pos_emb_flashatt, "_vlm2vec_patched", False):
            return

        def _apply_rotary_pos_emb_flashatt(self, tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
            tensor_ = tensor.to(freqs.dtype)
            cos = freqs.cos()
            sin = freqs.sin()
            return apply_rotary_emb(tensor_, cos, sin).type_as(tensor)

        _apply_rotary_pos_emb_flashatt._vlm2vec_patched = True
        cls._apply_rotary_pos_emb_flashatt = _apply_rotary_pos_emb_flashatt

    def save_pretrained(self, output_dir: str):
        """Save the base model."""
        self.base_model.save_pretrained(output_dir)
