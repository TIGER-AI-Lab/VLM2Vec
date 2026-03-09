from typing import Optional, List


DATASET_INSTRUCTION = {
    # Audio Classification
    "ESC-50": "Recognize the environmental sound category of the audio.",
    "UrbanSound8K": "Recognize the urban sound category of the audio.",
    "NSynth": "Recognize the musical instrument category of the audio.",
    "SpeechCommands": "Recognize the spoken word category in the audio.",
    "CREMA-D": "Recognize the emotion expressed in the speech audio.",

    # Text-to-Audio Retrieval
    "Clotho": "Retrieve audio clips that best match the given textual description.",
    "SoundDescs": "Retrieve audio samples that best match the given sound description.",

    # Audio-to-Image
    "SpeechCOCO": "Retrieve images that best match the given spoken description.",

    # Audio-to-Video
    "AVE": "Retrieve the video that best matches the given audio event.",

    # Audio Event Grounding
    "TUTSound": "Retrieve the sound event categories that best match the given audio.",
}


def build_query_text(dataset_name: str, raw_text: Optional[str] = None) -> List[str]:
    """
    统一的音频数据集查询文本构建函数。

    Args:
        dataset_name: 数据集名称，必须在 DATASET_INSTRUCTION 中
        raw_text: 原始查询文本，可选

    Returns:
        包含单个非空字符串的列表
    """
    if dataset_name not in DATASET_INSTRUCTION:
        raise KeyError(f"Dataset '{dataset_name}' not found in DATASET_INSTRUCTION. Available: {list(DATASET_INSTRUCTION.keys())}")

    instr = DATASET_INSTRUCTION[dataset_name]

    if raw_text is None or raw_text.strip() == "":
        return [instr]
    else:
        return [f"{instr}\nQuery: {raw_text.strip()}"]