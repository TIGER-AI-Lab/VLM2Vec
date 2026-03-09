from transformers import AutoProcessor

class Qwen2_5OmniEmbeddingProcessor:
    """
    轻量 wrapper：内部直接使用 AutoProcessor。
    """
    def __init__(self, base_processor):
        self.base = base_processor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        base = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(base)

    def save_pretrained(self, save_directory, **kwargs):
        return self.base.save_pretrained(save_directory, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        audio=None,
        *,
        fps: int | float | None = None,
        load_audio_from_video: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: str = "pt",
        **kwargs,
    ):
        # Support both single items (image/video) and batches (images/videos)
        if images is None and 'image' in kwargs:
            images = kwargs.pop('image')
        if videos is None and 'video' in kwargs:
            videos = kwargs.pop('video')

        if text is None and images is None and videos is None and audio is None:
            raise ValueError("At least one modality must be provided.")

        return self.base(
            text=text,
            images=images,
            videos=videos,
            audio=audio,
            return_tensors=return_tensors,
            **kwargs,
        )
