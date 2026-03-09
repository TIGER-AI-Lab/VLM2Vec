from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Qwen2_5OmniThinkerForConditionalGeneration
from configuration_qwen2_5_omni import Qwen2_5OmniEmbeddingConfig

@dataclass
class EmbeddingOutput:
    embeddings: torch.FloatTensor
    last_hidden_state: torch.FloatTensor | None = None  # optional, for debugging

def _mean_pool(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # hs: [B, L, H], mask: [B, L]
    m = mask.unsqueeze(-1).to(hs.dtype)
    return (hs * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)

class Qwen2_5OmniEmbeddingModel(PreTrainedModel):
    config_class = Qwen2_5OmniEmbeddingConfig

    def __init__(self, config: Qwen2_5OmniEmbeddingConfig):
        super().__init__(config)
        self.base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype="auto",
            device_map=None,   # 让外部 from_pretrained(device_map=...) 控制
        )

        # 可选：投影头（如果你训练时加了）
        self.proj = None
        if config.projection_dim is not None:
            hidden = getattr(self.base.config, "hidden_size", None)
            if hidden is None:
                raise ValueError("base model config has no hidden_size; set it explicitly.")
            self.proj = nn.Linear(hidden, config.projection_dim, bias=False)

    def forward(self, **inputs):
        # 强制稳定输出
        inputs["output_hidden_states"] = True
        inputs["return_dict"] = True
        inputs["use_cache"] = False

        out = self.base(**inputs)
        hs = out.hidden_states[-1]  # [B, L, H]
        mask = inputs.get("attention_mask", None)
        if mask is None:
            raise ValueError("attention_mask is required for mean pooling.")

        if self.config.pooling == "mean":
            emb = _mean_pool(hs, mask)
        elif self.config.pooling == "eos":
            # eos pooling 风险更大；建议只做 ablation
            lengths = mask.long().sum(dim=1) - 1
            emb = hs[torch.arange(hs.size(0), device=hs.device), lengths]
        else:
            raise ValueError(f"Unknown pooling={self.config.pooling}")

        if self.proj is not None:
            emb = self.proj(emb)

        if self.config.normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        return EmbeddingOutput(embeddings=emb, last_hidden_state=hs)