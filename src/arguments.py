from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name: str = field(
        default=None, metadata={"help": "huggingface model name or path"}
    )
    model_type: str = field(
        default=None, metadata={"help": "lavis model type"}
    )
    checkpoint_path: str = field(
        default=None, metadata={"help": "a local model path"}
    )
    pooling: str = field(
        default='eos',
        metadata={"help": "pooling method for encoder"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "normalize query and passage representations"}
    )
    temperature: float = field(
        default=0.02,
        metadata={"help": "temperature for softmax"}
    )
    lora: bool = field(
        default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "lora r"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "lora alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"}
    )
    lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
        metadata={"help": "lora target modules"}
    )
    num_crops: int = field(
        default=16,
        metadata={"help": "number of crops used in image encoder"}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    subset_name: List[str] = field(
        default=None, metadata={"help": "Useful for datasets with subsets"}
    )
    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )
    num_sample_per_subset: int = field(
        default=100, metadata={"help": "number of training samples per subset"}
    )
    image_dir: str = field(
        default=None, metadata={"help": "Image directory path"}
    )
    encode_output_path: str = field(
        default=None, metadata={"help": "encode output path"}
    )
    max_len: int = field(
        default=128, metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    embedding_type: str = field(
        default="", metadata={"help": "embedding type"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    image_encoder_freeze: bool = field(
        default=False, metadata={"help": "huggingface model name"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "directory for saving trained models"}
    )
    project_name: str = field(
        default=None, metadata={"help": "project name"}
    )

    logging_steps: int = field(
        default=1, metadata={"help": "logging steps"}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "number of training epochs"}
    )
    grad_cache: bool = field(
        default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(
        default=2, metadata={"help": "query side subset size"})
    gc_p_chunk_size: int = field(
        default=2, metadata={"help": "target side subset size"})


@dataclass
class MTEBArguments:
    task_types: List[str] = field(
        default=None, metadata={"help": ""}
    )
    tasks: List[str] = field(
        default=None, metadata={"help": ""}
    )
