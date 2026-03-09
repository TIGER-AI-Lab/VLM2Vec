# coding=utf-8
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import logging

logger = logging.get_logger(__name__)

AudioInput = Union[np.ndarray, torch.Tensor, List[float], List[int]]


def _to_tensor(audio: AudioInput) -> torch.Tensor:
    if isinstance(audio, torch.Tensor):
        return audio
    return torch.as_tensor(audio)


class Qwen2_5_OmniAudioProcessor:
    """
    音频处理器：将原始波形转换为 128-dim 梅尔频谱图
    与 Qwen2.5-Omni 模型兼容（期望 input_features shape: [B, 128, T]）
    """
    model_input_names = ["input_features", "feature_attention_mask", "audio_feature_lengths"]

    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
        # 梅尔频谱图参数（与 Qwen2AudioProcessor 保持一致）
        n_mels: int = 128,
        n_fft: int = 400,          # 25ms window @ 16kHz
        hop_length: int = 160,     # 10ms stride @ 16kHz
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
        self.dtype = dtype
        
        # 梅尔频谱图参数
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        
        # 创建 MelSpectrogram 转换器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
        )

    @staticmethod
    def _ensure_channel_first(wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        elif wave.dim() != 2:
            raise ValueError(f"Unsupported audio shape: {tuple(wave.shape)}; expect [T] or [C, T].")
        return wave

    def _process_single(self, audio: AudioInput) -> torch.Tensor:
        wave = _to_tensor(audio).to(dtype=self.dtype)
        wave = self._ensure_channel_first(wave)
        if self.mono and wave.size(0) > 1:
            wave = wave.mean(dim=0, keepdim=True)
        if self.normalize and wave.numel() > 0:
            peak = wave.abs().max()
            if peak > 0:
                wave = wave / peak
        return wave

    def preprocess(self, audios: Union[AudioInput, List[AudioInput]], return_tensors: Optional[str] = None, sampling_rate: Optional[int] = None) -> BatchFeature:
        """
        将音频波形转换为梅尔频谱图
        
        Args:
            audios: 音频波形列表（每个是 1D numpy.ndarray 或 torch.Tensor）
            return_tensors: 返回张量类型 ("pt" for PyTorch)
            sampling_rate: 采样率（可选，用于兼容性检查）
        
        Returns:
            BatchFeature 包含:
                - input_features: [B, 128, T_mel] 梅尔频谱图
                - feature_attention_mask: [B, T_mel] attention mask
                - audio_feature_lengths: [B] 每个样本的有效长度
        """
        if not isinstance(audios, (list, tuple)):
            audios = [audios]

        # 检查采样率一致性
        if sampling_rate is not None and sampling_rate != self.sample_rate:
            logger.warning(f"⚠️  Input sampling_rate={sampling_rate} != processor.sample_rate={self.sample_rate}. "
                         f"Make sure audios are resampled to {self.sample_rate}Hz before calling processor.")

        processed_waveforms: List[torch.Tensor] = []
        mel_spectrograms: List[torch.Tensor] = []
        mel_lengths: List[int] = []

        for idx, a in enumerate(audios):
            if a is None:
                raise ValueError(
                    f"[OmniAudioProcessor] received None at index={idx}. "
                    "This means audio loading/collator failed."
                )
            if isinstance(a, dict):
                raise ValueError(
                    f"[OmniAudioProcessor] received dict at index={idx} keys={list(a.keys())}. "
                    "Audio should be waveform tensor/ndarray BEFORE processor."
                )

            # 1) 预处理波形：归一化、转单声道等
            wave = self._process_single(a)  # [1, T] or [C, T]
            
            # 2) 提取梅尔频谱图
            # wave shape: [1, T] -> mel shape: [1, n_mels, T_mel]
            with torch.no_grad():
                mel_spec = self.mel_transform(wave)  # [C, n_mels, T_mel]
                
                # 转为对数刻度（模仿 Whisper/Qwen2Audio 的做法）
                mel_spec = torch.log(mel_spec + 1e-9)  # 避免 log(0)
                
                # 如果是多声道，取平均（通常已经是单声道了）
                if mel_spec.size(0) > 1:
                    mel_spec = mel_spec.mean(dim=0, keepdim=True)  # [1, n_mels, T_mel]
                
                # Squeeze channel dimension: [1, n_mels, T_mel] -> [n_mels, T_mel]
                mel_spec = mel_spec.squeeze(0)  # [n_mels=128, T_mel]
            
            mel_spectrograms.append(mel_spec)
            mel_lengths.append(int(mel_spec.shape[-1]))  # T_mel

        # 3) Padding 到最大长度
        max_mel_len = max(mel_lengths) if mel_lengths else 0
        if max_mel_len == 0:
            raise ValueError("[OmniAudioProcessor] max_mel_len==0, all audios are empty.")

        batch_list, mask_list = [], []
        for mel_spec, l in zip(mel_spectrograms, mel_lengths):
            # mel_spec shape: [n_mels, T_mel]
            pad = max_mel_len - l
            if pad > 0:
                # Pad 时间维度（最后一维）
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad), value=-9.0)  # 用一个小负值填充
            
            # Attention mask
            mask = torch.zeros((max_mel_len,), dtype=torch.long)
            mask[:l] = 1
            
            batch_list.append(mel_spec)
            mask_list.append(mask)

        # 4) Stack 成 batch
        input_features = torch.stack(batch_list, dim=0)              # [B, n_mels=128, T_mel]
        feature_attention_mask = torch.stack(mask_list, dim=0)       # [B, T_mel]
        audio_feature_lengths = feature_attention_mask.sum(dim=1)    # [B]

        # 只在第一次或每100个batch打印一次（减少日志输出）
        if not hasattr(self, '_batch_count'):
            self._batch_count = 0
        self._batch_count += 1
        if self._batch_count == 1 or self._batch_count % 100 == 0:
            logger.info(f"✅ Generated mel-spectrogram: {input_features.shape} (B={input_features.shape[0]}, n_mels={input_features.shape[1]}, T={input_features.shape[2]})")

        data = {
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "audio_feature_lengths": audio_feature_lengths,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs)

    def to_dict(self):
        return {
            "sample_rate": self.sample_rate,
            "mono": self.mono,
            "normalize": self.normalize,
            "dtype": str(self.dtype).replace("torch.", ""),  # e.g. "float32"
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "f_min": self.f_min,
            "f_max": self.f_max,
        }

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        dtype_name = str(config_dict.get("dtype", "float32")).replace("torch.", "")
        dtype_name = {"bf16": "bfloat16", "fp16": "float16"}.get(dtype_name, dtype_name)
        return cls(
            sample_rate=int(config_dict.get("sample_rate", 16000)),
            mono=bool(config_dict.get("mono", True)),
            normalize=bool(config_dict.get("normalize", True)),
            dtype=getattr(torch, dtype_name),
            n_mels=int(config_dict.get("n_mels", 128)),
            n_fft=int(config_dict.get("n_fft", 400)),
            hop_length=int(config_dict.get("hop_length", 160)),
            f_min=float(config_dict.get("f_min", 0.0)),
            f_max=config_dict.get("f_max", 8000.0),
        )