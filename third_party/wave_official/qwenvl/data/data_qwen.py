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
# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.

import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
# from torchcodec.decoders import VideoDecoder, AudioDecoder
from decord import VideoReader, cpu
import soundfile as sf
import ffmpeg
import transformers

import sys
if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))

from qwenvl.train.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX, PAD_TOKEN_ID, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_AUDIO_TOKEN

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def split_into_groups(counts, groups):
    result = []
    for count, g in zip(counts, groups):
        base = count // g
        remainder = count % g
        group_list = [base + 1] * remainder + [base] * (g - remainder)
        result.append(group_list)
    return result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = dataset
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.data_args = data_args
        
        self.omni_processor = data_args.omni_processor
        self.data_args.image_processor = self.omni_processor.image_processor
        self.data_args.audio_processor = self.omni_processor.feature_extractor

        list_data_dict = []

        for data in dataset_list:
            file_format = data.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data)
            else:
                annotations = json.load(open(data, "r"))
            list_data_dict += annotations

        for d in list_data_dict:
            if d["conversations"][0]["from"] == "system":
                idx = 1
            else:
                idx = 0
            if "<image>" in d["conversations"][idx]["value"] and not "image" in d and ("video" in d or "frame_dir" in d):
                d["conversations"][idx]["value"] = d["conversations"][idx]["value"].replace(
                    "<image>", "<video>"
                )
            if "<image>" in d["conversations"][idx]["value"] and not "image" in d and not ("video" in d or "frame_dir" in d) and ("audio" in d):
                d["conversations"][idx]["value"] = d["conversations"][idx]["value"].replace(
                    "<image>", "<audio>"
                )

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        # random.shuffle(list_data_dict, seed=2025)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        # self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        self.type_list = [it.get("type", "retrieval") for it in list_data_dict]
        self.type_dict = {}
        for k, it in enumerate(list_data_dict):
            tp = it.get("type", "retrieval")
            if tp not in self.type_dict:
                self.type_dict[tp] = [k]
            else:
                self.type_dict[tp].append(k)

    def __len__(self):
        return len(self.list_data_dict)

    def process_audio(self, audio_file=None, audio_wav=None, timestamps=None):
        try:
            audio_kwargs = {'sampling_rate': 16000, 'padding': 'max_length', 'return_attention_mask': True, 'return_tensors': 'pt'}

            processor = self.data_args.audio_processor

            mel = []
            if audio_wav is None:
                if isinstance(audio_file, list):
                    audio_data = []
                    for file in audio_file:
                        audio, sr = sf.read(audio_file)
                        if len(audio.shape) == 2:
                            audio = audio[:, 0]
                        assert sr == 16000
                        audio_data.append(audio)
                else:
                    audio, sr = sf.read(audio_file)
                    if len(audio.shape) == 2:
                        audio = audio[:, 0]
                    assert sr == 16000
                    if timestamps is not None:
                        audio = audio[timestamps[0] * sr: timestamps[1] * sr]
                    audio_data = [audio]
            else:
                sr = 16000
                audio_data = [audio_wav]

            audio_inputs = []
            audio_lengths = []
            for idx in range(len(audio_data)):
                feature_attention_mask_idx = []
                input_features_idx = []
                audio_lst = [audio_data[idx][k: k + 300 * audio_kwargs["sampling_rate"]] for k in range(0, len(audio_data[idx]), 300 * audio_kwargs["sampling_rate"])]
                audio_lengths_seg = 0
                for audio_seg in audio_lst:
                    if audio_seg.shape[0] < audio_kwargs["sampling_rate"]:
                        padding = audio_kwargs["sampling_rate"] - audio_seg.shape[0]
                        audio_seg = np.pad(audio_seg, (0, padding), mode="constant", constant_values=0)
                    audio_inputs_seg = self.data_args.audio_processor(audio_seg, **audio_kwargs)
                    attn_seg = audio_inputs_seg.pop("attention_mask")
                    feature_attention_mask_idx.append(attn_seg)
                    input_features_idx.append(audio_inputs_seg.pop("input_features"))
                    input_lengths_seg = (attn_seg.sum(-1) - 1) // 2 + 1
                    audio_lengths_seg += (input_lengths_seg - 2) // 2 + 1
                
                if audio_lengths_seg <= 0:
                    return None, None, None

                feature_attention_mask_idx = torch.cat(feature_attention_mask_idx, dim=0)
                input_features_idx = torch.cat(input_features_idx, dim=0)

                audio_inputs.append({
                    "feature_attention_mask": feature_attention_mask_idx,
                    "input_features": input_features_idx
                })
                audio_lengths.append(audio_lengths_seg)

            return audio_inputs, audio_lengths, audio_data
        
        except Exception as e:
            print(f"Process Audio Error: {e},  file: {audio_file}, line: {e.__traceback__.tb_lineno}")
            raise e
            

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.image_max_frame_pixels
        processor.min_pixels = self.data_args.image_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels

        image = Image.open(image_file).convert("RGB")
        width, height = image.size

        if width >= 28 and height >= 28:
            visual_processed = processor.preprocess(image, return_tensors="pt")
        else:
            pad_width = max(0, 28 - width)
            pad_height = max(0, 28 - height)
            
            if pad_width == 0 and pad_height == 0:
                pass
            else:
                left = pad_width // 2
                right = pad_width - left
                top = pad_height // 2
                bottom = pad_height - top
                fill_color=(0, 0, 0)
                padded_image = ImageOps.expand(image, border=(left, top, right, bottom), fill=fill_color)

            visual_processed = processor.preprocess(padded_image, return_tensors="pt")

        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file, timestamps=None, max_frame_num=-1):
        video = self.video_decord(video_file, timestamps=timestamps, max_frame_num=max_frame_num)
        return video

    def video_decord(self, video_file, timestamps=None, max_frame_num=-1):
        vr = VideoReader(video_file, num_threads=1) # , ctx=cpu(0)
        total_frame_num = len(vr)

        video_length = total_frame_num / vr.get_avg_fps()
        video_min_frames = getattr(self.data_args, "video_min_frames", 1)
        video_max_frames = getattr(self.data_args, "video_max_frames", 600) if max_frame_num <= 0 else max_frame_num

        interval = getattr(self.data_args, "base_interval", 0.5)

        if timestamps is None:
            num_frames_to_sample = round(video_length / interval)
            target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
            frame_idx = np.linspace(0, total_frame_num - 1, target_frames, dtype=int)
        else:
            video_length = min(timestamps[1], video_length) - timestamps[0]
            num_frames_to_sample = round(video_length / interval)
            target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
            
            start_idx = round(timestamps[0] * vr.get_avg_fps())
            end_idx = min(round(timestamps[1] * vr.get_avg_fps()), total_frame_num - 1)
            frame_idx = np.linspace(start_idx, end_idx, target_frames, dtype=int)

        video = vr.get_batch(frame_idx).asnumpy() # video: (F, H, W, C)
        
        video = np.array(video)

        video = torch.from_numpy(video)
        video_proc = self.data_args.image_processor(images=None, videos=video, return_tensors="pt")
        fps = len(frame_idx) / video_length     # 1 / interval
        fps = [fps] * 1
        video_proc["video_second_per_grid"] = [self.data_args.image_processor.temporal_patch_size / fps[i] for i in range(len(fps))]
        return video_proc["pixel_values_videos"], video_proc['video_grid_thw'], video_proc["video_second_per_grid"]

    def process_omni_conversations(self, conversations, type, gpt="gpt"):
        label = None
        omni_conversations = []
        for conv in conversations:
            if conv["from"] == "human":
                if "<video>\n" in conv["value"]:
                    omni_conversations.append({
                        "role": "user",
                        "content": [{"type": "video"}, {"type": "text", "text": conv["value"].replace("<video>\n", "")}]
                    })
                elif "<audio>\n" in conv["value"]:
                    omni_conversations.append({
                        "role": "user",
                        "content": [{"type": "audio"}, {"type": "text", "text": conv["value"].replace("<audio>\n", "")}]
                    })
                elif "<image>\n" in conv["value"]:
                    omni_conversations.append({
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": conv["value"].replace("<image>\n", "")}]
                    })
                else:
                    omni_conversations.append({
                        "role": "user",
                        "content": [{"type": "text", "text": conv["value"]}]
                    })
            elif conv["from"] == gpt:
                if "sft" not in type and self.data_args.train_classify:
                    label = conv["value"]
                else:
                    omni_conversations.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": conv["value"]}]
                    })
        
        text = self.omni_processor.apply_chat_template(omni_conversations, add_generation_prompt=False, tokenize=False)

        if "sft" not in type and self.data_args.train_classify:
            text = [text[0].split("<|im_start|>user\n")[-1].strip()]
        else:
            text = [text[0].strip()]

        return text, label

    def gen_omni_labels(self, text):
        all_convs = text.split("<|im_start|>")
        im_start = self.tokenizer("<|im_start|>")["input_ids"]
        labels = []
        for conv in all_convs:
            if conv:
                if conv.startswith("system\n") or conv.startswith("user\n"):
                    labels += [IGNORE_INDEX] * (self.tokenizer(conv, padding=True, padding_side="left", return_tensors="pt")["input_ids"].size(1) + 1)
                elif conv.startswith("assistant\n"):
                    labels += [IGNORE_INDEX] * 3    # <|im_start|>assistant\n
                    labels += self.tokenizer(conv[len("assistant\n"):], padding=True, padding_side="left", return_tensors="pt")["input_ids"].tolist()[0]
                else:
                    raise NotImplementedError
        
        labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0)
        return labels

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self._get_item(i)
        return sample 

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

            # define some variables
            image = None
            grid_thw_merged = None
            video_grid_thw_merged = None
            grid_thw = None
            video = None
            video_grid_thw = None
            second_per_grid_ts = None
            audio = None
            audio_lengths = None
            raw_wav = None

            if "image" in sources[0]:
                image_file = self.list_data_dict[i]["image"]
                image, grid_thw = self.process_image_unified(image_file)
                grid_thw = grid_thw.unsqueeze(0)
                image = [image]
                grid_thw_merged = copy.deepcopy(grid_thw)
                if not isinstance(grid_thw, Sequence):
                    grid_thw_merged = [grid_thw_merged]
                    grid_thw = [grid_thw]
                grid_thw_merged = [
                    merged_thw.prod() // self.data_args.image_processor.merge_size**2
                    for merged_thw in grid_thw_merged
                ]

            if "frame_dir" in sources[0]:
                video_max_frames = getattr(self.data_args, "video_max_frames", 600)
                video_file = sources[0]["frame_dir"]
                frame_files = os.listdir(video_file)
                frame_files = sorted(frame_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                frame_files = [os.path.join(video_file, fr) for fr in frame_files]
                image_list = []
                for f_file in frame_files:
                    image_list.append(np.array(Image.open(f_file)))
                
                if len(image_list) > video_max_frames:
                    i_indices = np.linspace(0, len(image_list) - 1, video_max_frames, dtype=int).tolist()
                    image_list = [image_list[k] for k in i_indices]
                    
                video = np.stack(image_list)

                video = torch.from_numpy(video)
                video_proc = self.data_args.image_processor(images=None, videos=video, return_tensors="pt")
                fps = [len(image_list) / sources[0]["duration"] if "duration" in sources[0] and sources[0]["duration"] > 1 else 1] * 1
                video_proc["video_second_per_grid"] = [self.data_args.image_processor.temporal_patch_size / fps[i] for i in range(len(fps))]
                video, video_grid_thw, second_per_grid_ts = video_proc["pixel_values_videos"], video_proc['video_grid_thw'], video_proc["video_second_per_grid"]

                video = [video]

                video_grid_thw_merged = copy.deepcopy(video_grid_thw)
                if not isinstance(video_grid_thw, Sequence):
                    video_grid_thw_merged = [video_grid_thw_merged]
                    video_grid_thw = [video_grid_thw]
                else:
                    raise NotImplementedError

            elif "video" in sources[0]:
                video_file = sources[0]["video"]
                timestamps = sources[0].get("timestamps", None)
                if isinstance(video_file, List):
                    if len(video_file) > 1:
                        video_file = [
                            file for file in video_file
                        ]
                        results = [self.process_video(file) for file in video_file]
                        video, video_grid_thw, second_per_grid_ts = zip(*results)
                    else:
                        video_file = video_file[0]
                        video, video_grid_thw, second_per_grid_ts = self.process_video(video_file)
                        video = [video]
                else:
                    video, video_grid_thw, second_per_grid_ts = self.process_video(video_file, timestamps=timestamps)
                    video = [video]

                video_grid_thw_merged = copy.deepcopy(video_grid_thw)
                if not isinstance(video_grid_thw, Sequence):
                    video_grid_thw_merged = [video_grid_thw_merged]
                    video_grid_thw = [video_grid_thw]

            if "audio" in sources[0]:
                audio_file = sources[0]["audio"]
                timestamps = sources[0].get("timestamps", None)
                audio, audio_lengths, raw_wav = self.process_audio(audio_file, timestamps=timestamps)
            
            if raw_wav is not None and len(raw_wav[0]) < 16000: # pad audio to at least 1s
                sil = np.zeros(16000 - len(raw_wav[0]), dtype=float)
                raw_wav[0] = np.concatenate((raw_wav[0], sil), axis=0)
            
            chat_sources = copy.deepcopy([e["conversations"] for e in sources])

            text, label = self.process_omni_conversations(sources[0]["conversations"], sources[0].get("type", "retrieval"), gpt="gpt")
            text = self.omni_processor.replace_multimodal_special_tokens(
                text,
                iter(audio_lengths[0]) if audio is not None else iter([]),
                iter(grid_thw[0]) if image is not None else iter([]),
                iter(video_grid_thw[0]) if video_grid_thw is not None else iter([]),
                video_second_per_grid=iter(second_per_grid_ts) if video_grid_thw is not None else iter([]),
                use_audio_in_video=audio is not None,
                position_id_per_seconds=25,
                seconds_per_chunk=2.0 * second_per_grid_ts[0] if second_per_grid_ts is not None else None,
            )
            assert len(text) == 1
            if (self.data_args.use_beats and not self.data_args.beats_only):
                text[0] = text[0].replace("<|AUDIO|>", "<|AUDIO|><|AUDIO|>")
            
            labels = None
            if "sft" in sources[0].get("type", "retrieval") or not self.data_args.train_classify:
                labels = self.gen_omni_labels(text[0])

            token_res = self.tokenizer(text, padding=True, padding_side="left", return_tensors="pt")
            input_ids = token_res["input_ids"]
            attention_mask = token_res["attention_mask"]

            label_ids = None
            if sources[0].get("type", "retrieval") != "sft":
                if label:
                    if not label.endswith("<|im_end|>"):
                        label += "<|im_end|>"
                    label_ids = self.tokenizer(label, padding=True, padding_side="left", return_tensors="pt")["input_ids"]

            
            if "pos_image" in sources[0]:
                if "image" in sources[0]["pos_image"]:
                    pos_image_file = sources[0]["pos_image"]["image"]
                    pos_image, pos_grid_thw = self.process_image_unified(pos_image_file)
                    pos_image = [pos_image]
                    pos_grid_thw = [pos_grid_thw.unsqueeze(0)]
                else:
                    pos_image, pos_grid_thw = None, None

                if "video" in sources[0]["pos_image"]:
                    pos_video_file = sources[0]["pos_image"]["video"]
                    pos_video, pos_video_grid_thw, pos_second_per_grid_ts = self.process_video(pos_video_file)
                    pos_video = [pos_video]
                    pos_video_grid_thw = [pos_video_grid_thw]
                else:
                    pos_video, pos_video_grid_thw, pos_second_per_grid_ts = None, None, None

                pos_text, _ = self.process_omni_conversations(sources[0]["pos_image"]["conversations"], sources[0].get("type", "retrieval"), gpt="gpt")
                pos_text = self.omni_processor.replace_multimodal_special_tokens(
                    pos_text,
                    iter([]),
                    iter(pos_grid_thw[0]) if pos_grid_thw is not None else iter([]),
                    iter(pos_video_grid_thw[0]) if pos_video_grid_thw is not None else iter([]),
                    video_second_per_grid=iter(pos_second_per_grid_ts) if pos_second_per_grid_ts is not None else iter([]),
                    use_audio_in_video=False,
                    position_id_per_seconds=25,
                    seconds_per_chunk=None,
                )
                pos_token_res = self.tokenizer(pos_text, padding=True, padding_side="left", return_tensors="pt")
                pos_input_ids = pos_token_res["input_ids"]
                pos_attention_mask = pos_token_res["attention_mask"]

            else:
                pos_video, pos_video_grid_thw, pos_second_per_grid_ts = None, None, None
                pos_image, pos_grid_thw, pos_input_ids, pos_attention_mask = None, None, None, None


            data_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": image[0] if image is not None else None,
                "image_grid_thw": grid_thw[0] if grid_thw is not None else None,
                "pixel_values_videos": video[0] if video is not None else None,
                "video_grid_thw": video_grid_thw[0] if video_grid_thw is not None else None,
                "video_second_per_grid": second_per_grid_ts[0] if second_per_grid_ts is not None else None,
                "input_features": audio[0]["input_features"] if audio is not None else None,
                "feature_attention_mask": audio[0]["feature_attention_mask"] if audio is not None else None,
                "label_ids": label_ids,
                "type": sources[0].get("type", "retrieval"),
                "labels": labels,
                "pos_pixel_values": pos_image[0] if pos_image is not None else None,
                "pos_image_grid_thw": pos_grid_thw[0] if pos_grid_thw is not None else None,
                "pos_input_ids": pos_input_ids,
                "pos_attention_mask": pos_attention_mask,
                "pos_pixel_values_videos": pos_video[0] if pos_video is not None else None,
                "pos_video_grid_thw": pos_video_grid_thw[0] if pos_video_grid_thw is not None else None,
                "pos_video_second_per_grid": pos_second_per_grid_ts,
                "input_raw_wav": torch.from_numpy(raw_wav[0]) if raw_wav is not None else None,
            }

            if "neg_text" in sources[0]: # and sources[0].get("type", "retrieval") == "give_neg":
                neg_text = sources[0]["neg_text"]
                neg_text = [it + "<|im_end|>" for it in neg_text]
                all_text = [label] + neg_text
                all_label_res = self.tokenizer(all_text, padding=True, padding_side="left", return_tensors="pt")
                all_ids = all_label_res["input_ids"]
                all_attention_mask = all_label_res["attention_mask"]
                data_dict["all_ids"] = all_ids
                data_dict["all_attention_mask"] = all_attention_mask
                # data_dict["all_names"] = all_text
            else:
                data_dict["all_ids"] = None
                data_dict["all_attention_mask"] = None

            if self.data_args.run_test:
                if sources[0].get("type", "retrieval") != "sft":
                    if "neg_text" in sources[0]:
                        neg_text = sources[0]["neg_text"]
                        neg_text = [it + "<|im_end|>" for it in neg_text]
                        all_text = [label] + neg_text
                        all_label_res = self.tokenizer(all_text, padding=True, padding_side="left", return_tensors="pt")
                        all_ids = all_label_res["input_ids"]
                        all_attention_mask = all_label_res["attention_mask"]
                        data_dict["all_ids"] = all_ids
                        data_dict["all_attention_mask"] = all_attention_mask
                        data_dict["all_names"] = all_text
                
                if "sft" in sources[0].get("type", "retrieval"):
                    labels = data_dict.pop("labels", None)
                    len_input = sum(labels[0] == IGNORE_INDEX)
                    data_dict["input_ids"] = data_dict["input_ids"][:, :len_input]
                    data_dict["attention_mask"] = torch.ones_like(data_dict["input_ids"])
                    labels = None

                if "video" in sources[0]:
                    data_dict["video"] = sources[0]["video"]
                elif "frame_dir" in sources[0]:
                    data_dict["video"] = sources[0]["frame_dir"]
                else:
                    data_dict["video"] = None

                data_dict["image"] = sources[0].get("image", None)

                if "audio" in sources[0]:
                    data_dict["audio"] = sources[0]["audio"]
                else:
                    data_dict["audio"] = None

                data_dict["prompt"] = sources[0]["conversations"][:-1]
                data_dict["ref"] = sources[0]["conversations"][-1]["value"]

            return data_dict

        except Exception as e:
            print(f"Error: {e}, line: {e.__traceback__.tb_lineno}")
            # raise e
            if self.data_args.run_test:
                print(f"Error loading {sources[0]}")
                return None
            else:
                randidx = random.choice(self.type_dict[sources[0].get("type", "retrieval")])
                return self.__getitem__(randidx)

@dataclass
class DataCollatorForOmniDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert all([it["type"] == instances[0]["type"] for it in instances])

        if instances[0]["type"] != "sft":
            input_ids, label_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "label_ids"))
            input_ids = [ids.squeeze(0) for ids in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID, padding_side='left')
            attention_mask = input_ids.ne(PAD_TOKEN_ID).to(torch.int64)

            if label_ids[0] is not None:
                label_ids = [ids.squeeze(0) for ids in label_ids]
                label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=PAD_TOKEN_ID, padding_side='left')
                label_attention_mask = label_ids.ne(PAD_TOKEN_ID).to(torch.int64)
            else:
                label_ids, label_attention_mask = None, None

            if instances[0]["labels"] is not None:
                labels = [instance["labels"] for instance in instances]
                labels = [ids.squeeze(0) for ids in labels]
                labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side='left')
            else:
                labels = None
        else:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = [ids.squeeze(0) for ids in input_ids]
            labels = [ids.squeeze(0) for ids in labels]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID, padding_side='right')
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side='right')
            attention_mask = input_ids.ne(PAD_TOKEN_ID).to(torch.int64)
            label_ids, label_attention_mask = None, None

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance and instance["pixel_values"] is not None]
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            concat_grid_thw = torch.cat([instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance and instance["image_grid_thw"] is not None], dim=0)
        else:
            concat_images = None
            concat_grid_thw = None

        videos = [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance and instance["pixel_values_videos"] is not None]
        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            concat_video_grid_thw = torch.cat([instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance and instance["video_grid_thw"] is not None], dim=0)
            video_second_per_grid = torch.tensor([instance["video_second_per_grid"] for instance in instances if "video_second_per_grid" in instance and instance["video_second_per_grid"] is not None], dtype=torch.float32)
        else:
            concat_videos = None
            concat_video_grid_thw = None
            video_second_per_grid = None

        audios = [instance["input_features"] for instance in instances if "input_features" in instance and instance["input_features"] is not None]
        if len(audios) != 0:
            concat_audios = torch.cat([audio for audio in audios], dim=0)
            concat_feature_attention_mask = torch.cat([instance["feature_attention_mask"] for instance in instances if "feature_attention_mask" in instance and instance["feature_attention_mask"] is not None], dim=0)
        else:
            concat_audios = None
            concat_feature_attention_mask = None

        if instances[0]["input_raw_wav"] is not None:
            input_raw_wav = [instance["input_raw_wav"] for instance in instances]
        else:
            input_raw_wav = None

        types = [instance["type"] for instance in instances]
        if any([it["all_ids"] is not None for it in instances]):
            all_ids = [instance["all_ids"] for instance in instances]
            all_attention_mask = [instance["all_attention_mask"] for instance in instances]
        else:
            all_ids = None
            all_attention_mask = None

        if instances[0]["pos_input_ids"] is not None:
            pos_input_ids = [instance["pos_input_ids"] for instance in instances]
            pos_input_ids = [ids.squeeze(0) for ids in pos_input_ids]
            pos_input_ids = torch.nn.utils.rnn.pad_sequence(pos_input_ids, batch_first=True, padding_value=PAD_TOKEN_ID, padding_side='left')
            pos_attention_mask = pos_input_ids.ne(PAD_TOKEN_ID).to(torch.int64)
        else:
            pos_input_ids = None
            pos_attention_mask = None

        pos_images = [instance["pos_pixel_values"] for instance in instances]
        if pos_images[0] is not None:
            concat_pos_images = torch.cat([image for image in pos_images], dim=0)
            concat_pos_grid_thw = torch.cat([instance["pos_image_grid_thw"] for instance in instances], dim=0)
        else:
            concat_pos_images = None
            concat_pos_grid_thw = None

        pos_videos = [instance["pos_pixel_values_videos"] for instance in instances]
        if pos_videos[0] is not None:
            concat_pos_videos = torch.cat([video for video in pos_videos], dim=0)
            concat_pos_video_grid_thw = torch.cat([instance["pos_video_grid_thw"] for instance in instances], dim=0)
            pos_video_second_per_grid = torch.tensor([instance["pos_video_second_per_grid"] for instance in instances], dtype=torch.float32)
        else:
            concat_pos_videos = None
            concat_pos_video_grid_thw = None
            pos_video_second_per_grid = None


        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": concat_images,
            "image_grid_thw": concat_grid_thw,
            "pixel_values_videos": concat_videos,
            "video_grid_thw": concat_video_grid_thw,
            "video_second_per_grid": video_second_per_grid,
            "input_features": concat_audios,
            "feature_attention_mask": concat_feature_attention_mask,
            "label_ids": label_ids,
            "label_attention_mask": label_attention_mask,
            "use_audio_in_video": concat_video_grid_thw is not None and concat_audios is not None,
            "types": types,
            "all_ids": all_ids,
            "all_attention_mask": all_attention_mask,
            "labels": labels,
            "pos_pixel_values":  concat_pos_images,
            "pos_image_grid_thw": concat_pos_grid_thw,
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_attention_mask,
            "pos_pixel_values_videos": concat_pos_videos,
            "pos_video_grid_thw": concat_pos_video_grid_thw,
            "pos_video_second_per_grid": pos_video_second_per_grid,
            "input_raw_wav": input_raw_wav,
        }

        return batch

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForOmniDataset()
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    from qwenvl.train.argument import DataArguments
    from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    
    random.seed(2025)
    data_args = DataArguments()
    data_args.dataset_use = "/mnt/sh/mmvision/home/changlitang/preprocess_dataset/retAv_panda100w_mcOe10w_Shot154k_retUps.json" # /mnt/sh/mmvision/home/changlitang/preprocess_dataset/mmebv2_retrieval_msrvtt_av.json"
    data_args.video_max_frames = 128
    data_args.run_test = False # True # 
    data_args.use_beats = True # 
    data_args.beats_only = False
    data_args.train_classify = True # 

    data_args.video_min_frames = 1

    processor = Qwen2_5OmniProcessor.from_pretrained("/mnt/sh/mmvision/home/changlitang/models/Qwen2.5-Omni-7B")
    data_args.omni_processor = processor
    tokenizer = processor.tokenizer

    dataset = LazySupervisedDataset(tokenizer, data_args=data_args)
    # from tqdm import tqdm
    # for i in tqdm(range(len(dataset))):
    #     item0 = dataset._get_item(i)
    # item0 = dataset._get_item(42041+19180+84723+7481)
    item0 = dataset._get_item(0)
    # item1 = dataset._get_item(1)

    collate_fn = DataCollatorForOmniDataset()

    batch = collate_fn([item0])
    breakpoint()