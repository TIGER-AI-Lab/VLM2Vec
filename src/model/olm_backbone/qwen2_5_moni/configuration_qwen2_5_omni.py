from transformers import PretrainedConfig

class Qwen2_5OmniEmbeddingConfig(PretrainedConfig):
    model_type = "qwen2_5_omni_embedding"

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-Omni-3B",
        pooling: str = "mean",
        normalize: bool = True,
        projection_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.pooling = pooling
        self.normalize = normalize
        self.projection_dim = projection_dim