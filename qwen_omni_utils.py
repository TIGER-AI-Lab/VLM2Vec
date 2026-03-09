"""
Minimal utilities to align with the Qwen-Omni/Nemotron official example.
Provides process_mm_info used by the nvomniembed evaluation pipeline.
"""

from typing import Any, List, Tuple, Optional


def process_mm_info(documents: List[dict], use_audio_in_video: bool = False) -> Tuple[Optional[List[Any]], Optional[List[Any]], Optional[List[Any]]]:
    """
    Extract audio, images, and videos from a Qwen-Omni style document list.

    Expected input format (per official example):
    [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "..."},
          {"type": "image", "image": <PIL.Image or path>},
          {"type": "video", "video": <list of frames or path>},
          {"type": "audio", "audio": <np.ndarray/torch.Tensor/path>},
        ],
      }
    ]
    """
    del use_audio_in_video  # reserved for compatibility

    audios: List[Any] = []
    images: List[Any] = []
    videos: List[Any] = []

    if not documents:
        return None, None, None

    for msg in documents:
        content = msg.get("content", []) if isinstance(msg, dict) else []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "image":
                images.append(part.get("image", None))
            elif ptype == "video":
                videos.append(part.get("video", None))
            elif ptype == "audio":
                audios.append(part.get("audio", None))

    # Normalize: return None when empty
    if not audios:
        audios = None
    if not images:
        images = None
    if not videos:
        videos = None

    return audios, images, videos

