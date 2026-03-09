import torch


def load_qwen2_5omni_thinker(model_name_or_path: str, **kwargs) -> torch.nn.Module:
    """
    Load Qwen2.5-Omni and return the thinker submodel for embedding/training.
    This avoids AutoModel which cannot resolve Qwen2_5OmniConfig.
    """
    from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

    full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
    return full_model.thinker if hasattr(full_model, "thinker") else full_model


class Qwen2_5OmniBackbone(torch.nn.Module):
    """
    Thin wrapper over the Qwen2.5-Omni thinker for use as a backbone.
    """

    def __init__(self, thinker: torch.nn.Module):
        super().__init__()
        self.base_model = thinker
        self.config = thinker.config
        self.model = getattr(thinker, "model", None)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "Qwen2_5OmniBackbone":
        thinker = load_qwen2_5omni_thinker(model_name_or_path, **kwargs)
        return cls(thinker)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
