# Copyright (2026) Tsinghua University, Tencent Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/bytedance/video-SALMONN-2. The original license is located at 'third-party-license/video-salmonn-2.txt'.
# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl25.txt'.
# Adopted from https://github.com/huggingface/transformers. The original license is located at 'third-party-license/transformers.txt'.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import random
import copy
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers import AutoConfig
from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
from qwenvl.train.trainer import QwenVLTrainer

from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from tqdm import tqdm
import torch.distributed as dist
import yaml

from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from qwenvl.train.utils import KeywordsStoppingCriteria

import time

local_rank = None

def collate_fn(batch):
    if batch[0]["input_raw_wav"] is not None:
        batch[0]["input_raw_wav"] = [batch[0]["input_raw_wav"]]
    return batch[0]

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    print("Applying Liger kernels to Qwen2.5 model...")

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from qwenvl.model.qwen2_5_omni import modeling_qwen2_5_omni

    if rope:
        modeling_qwen2_5_omni.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
    if rms_norm:
        modeling_qwen2_5_omni.Qwen2RMSNorm = LigerRMSNorm
    if swiglu:
        modeling_qwen2_5_omni.Qwen2MLP = LigerSwiGLUMLP


def set_model(model_args, model):
    if model_args.train_classify:
        model.classify_linear.requires_grad_(True)

    if model_args.use_beats:
        if model_args.tune_beats_proj:
            model.beats_ln.requires_grad_(True)
            model.beats_proj.requires_grad_(True)
        else:
            model.beats_ln.requires_grad_(False)
            model.beats_proj.requires_grad_(False)

    if model_args.tune_mm_vision:
        model.visual.requires_grad_(True)
    else:
        model.visual.requires_grad_(False)

    if model_args.tune_mm_mlp:
        model.visual.merger.requires_grad_(True)
    else:
        model.visual.merger.requires_grad_(False)

    if model_args.tune_mm_audio:
        model.audio_tower.requires_grad_(True)
    else:
        model.audio_tower.requires_grad_(False)

    if model_args.tune_mm_qformer:
        model.audio_tower.ln_post.requires_grad_(True)
        model.audio_tower.proj.requires_grad_(True)
    else:
        model.audio_tower.ln_post.requires_grad_(False)
        model.audio_tower.proj.requires_grad_(False)

    if model_args.tune_mm_llm:
        if model_args.use_lora:
            raise Exception("tune_mm_llm is not supported when use_lora is True")
        model.model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    else:
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)

def train(attn_implementation="flash_attention_2"):
    global local_rank

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.train_classify = model_args.train_classify
    model_args.temperature = training_args.temperature
    data_args.use_beats = model_args.use_beats
    data_args.beats_only = model_args.beats_only

    assert model_args.classify_type in ["last_layer", "all_layer"], f"classify_type {model_args.classify_type} is not supported"

    training_args.remove_unused_columns = False

    apply_liger_kernel_to_qwen2_5_vl()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    processor = Qwen2_5OmniProcessor.from_pretrained(model_args.model_base)
    data_args.omni_processor = processor
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = training_args.model_max_length

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if not data_args.run_test:
        model_config = Qwen2_5OmniThinkerConfig.from_pretrained(model_args.model_base, cache_dir=training_args.cache_dir,)
        if model_args.train_classify:
            model_config.train_classify = model_args.train_classify
            model_config.classify_type = model_args.classify_type
            model_config.sim_temperature = model_args.temperature
        
        if model_args.use_beats:
            model_config.audio_config.beats_path = "/mnt/sh/mmvision/home/changlitang/models/BEATs/BEATs_iter3_plus.pt"
            model_config.audio_config.beats_only = model_args.beats_only

        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=model_config, cache_dir=training_args.cache_dir, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), attn_implementation=attn_implementation)
        
        if model_args.use_beats:
            beats_ckpt = torch.load(model_config.audio_config.beats_path, map_location='cpu')
            model.beats.load_state_dict(beats_ckpt['model'])

        model.requires_grad_(False)

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            if training_args.gradient_checkpointing_kwargs is None:
                training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
            else:
                training_args.gradient_checkpointing_kwargs["use_reentrant"] = False

        if model_args.lora_ckpt != "No":
            model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
            
        set_model(model_args, model)

        if model_args.use_lora:
            module_to_save = []
            if model_args.tune_mm_vision:
                module_to_save.append("visual")
            if model_args.tune_mm_mlp:
                module_to_save.append("visual.merger")
            if model_args.tune_mm_audio:
                module_to_save.append("audio")
            if model_args.tune_mm_qformer:
                module_to_save.append("audio_tower.ln_post")
                module_to_save.append("audio_tower.proj")
            if model_args.train_classify:
                module_to_save.append("classify_linear")
            if model_args.use_beats:
                if model_args.tune_beats_proj:
                    module_to_save.append("beats_ln")
                    module_to_save.append("beats_proj")
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=r"model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj",
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=module_to_save,
            )
            model = get_peft_model(model, lora_config)

        for k, v in model.named_parameters():
            if "lora" in k:
                v.requires_grad_(True)
        
        cnt, total = 0, 0
        if dist.get_rank() == 0:
            for k, v in model.named_parameters():
                if v.requires_grad:
                    print(k, v.shape)
                    cnt += 1
                total += 1
            print(cnt, total)
        
        trainer = QwenVLTrainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
        
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
            
    else:
        model_config = Qwen2_5OmniThinkerConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir,)
        if model_args.train_classify:
            model_config.train_classify = model_args.train_classify
            model_config.classify_type = model_args.classify_type
            model_config.sim_temperature = model_args.temperature
        if model_args.use_beats:
            model_config.audio_config.beats_path = "/mnt/sh/mmvision/home/changlitang/models/BEATs/BEATs_iter3_plus.pt"
            model_config.audio_config.beats_only = model_args.beats_only

        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=model_config, cache_dir=training_args.cache_dir, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), attn_implementation=attn_implementation)

        if model_args.use_beats:
            beats_ckpt = torch.load(model_config.audio_config.beats_path, map_location='cpu')
            model.beats.load_state_dict(beats_ckpt['model'])
            
        if model_args.lora_ckpt != "No":
            model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
            model = model.merge_and_unload()

            # model.save_pretrained("/mnt/sh/mmvision/home/changlitang/models/WAVE-7B-0211", safe_serialization=False)

        model.eval()
        model.cuda()

        pred_rank = training_args.pred_rank

        os.makedirs(os.path.join(training_args.output_dir, training_args.run_name), exist_ok=True)
        result = []
        test_data = data_module["train_dataset"]
        loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=collate_fn,
            in_order=True,
        )
        all_time = 0.0
        cnt = 0
        for inputs in tqdm(loader, desc=f"RANK {pred_rank}"):
            if inputs:
                res_i = {
                    "video": inputs.pop("video", None),
                    "image": inputs.pop("image", None),
                    "prompt": inputs.pop("prompt", None),
                    "ref": inputs.pop("ref", None),
                    "audio": inputs.pop("audio", None),
                    "use_audio": inputs.pop("use_audio", False),
                }

                new_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        new_inputs[k] = v.to(model.device)
                    elif (k == 'video_second_per_grid' or k == 'pos_video_second_per_grid') and v is not None:
                        new_inputs[k] = torch.tensor([v], device=model.device)
                    elif v is None:
                        continue
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                        new_inputs[k] = [it.to(model.device) for it in v]
                    else:
                        new_inputs[k] = v

                inputs = new_inputs
                inputs["types"] = inputs.pop("type")

                if data_args.pred_embeds:
                    inputs["pred_embeds"] = True
                    if "seg_video" not in inputs or inputs["seg_video"] is None:
                        with torch.no_grad():
                            start = time.time()
                            outputs = model(**inputs)
                            end = time.time()
                            print("time: ", end - start)
                            all_time += end - start
                            cnt += 1
                        
                        mllm_embeds = outputs.mllm_embeds
                        text_embeds = outputs.text_embeds

                        res_i["mllm_embeds"] = mllm_embeds.cpu() if mllm_embeds is not None else None
                        res_i["text_embeds"] = text_embeds.cpu() if text_embeds is not None else None
                    
                    else:
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        mllm_embeds = outputs.mllm_embeds
                        text_embeds = outputs.text_embeds

                        seg_video = inputs["seg_video"]
                        seg_video_grid_thw = inputs["seg_video_grid_thw"]
                        ori_input_ids = inputs["input_ids"].tolist()
                        start_idx = torch.where(inputs["input_ids"][0] == 151652)[0]
                        end_idx = torch.where(inputs["input_ids"][0] == 151653)[0]

                        all_mllm_embeds = [mllm_embeds]
                        for i in range(seg_video.size(0)):
                            new_ids = ori_input_ids[0][:start_idx + 1] + [151656] * (seg_video[i].size(0) // 4) + ori_input_ids[0][end_idx:]
                            inputs["input_ids"] = torch.tensor([new_ids], dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
                            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                            inputs["pixel_values_videos"] = seg_video[i]
                            inputs["video_grid_thw"] = seg_video_grid_thw[i]

                            with torch.no_grad():
                                outputs = model(**inputs)
                            
                            all_mllm_embeds.append(outputs.mllm_embeds)
                        
                        all_mllm_embeds = F.normalize(torch.cat(all_mllm_embeds, dim=0), dim=-1)
                        text_embeds = F.normalize(text_embeds, dim=-1)
                        t2v_sims = text_embeds @ all_mllm_embeds.t()
                        breakpoint()

                else:
                    with torch.no_grad():
                        outputs = model(**inputs)

                    v2t_sims = outputs.v2t_sims
                    pred_idx = torch.argmax(v2t_sims, dim=1)
                    pred = inputs["all_names"][pred_idx]
                    res_i["pred"] = pred

                result.append(res_i)

        if data_args.pred_embeds:
            torch.save(result, os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.bin"))
            print(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.bin"))
            print(f"Total Time: {all_time}, Samples: {cnt}, Avg: {all_time / cnt}")
        else:
            with open(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"), "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"))

        return

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
