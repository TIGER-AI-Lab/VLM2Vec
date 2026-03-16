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

import torch
from transformers import StoppingCriteria

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
PAD_TOKEN_ID = 151643
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        batch_size = output_ids.shape[0]
        all_matched = True
        for batch_idx in range(batch_size):
            sample_output_ids = output_ids[batch_idx:batch_idx + 1]
            offset = sample_output_ids.shape[1] - self.start_len # min(sample_output_ids.shape[1] - self.start_len, 3)
            self.keyword_ids = [keyword_id.to(sample_output_ids.device) for keyword_id in self.keyword_ids]
            keyword_matched = False
            for keyword_id in self.keyword_ids:
                if torch.all(sample_output_ids[0, -keyword_id.shape[0]:] == keyword_id):
                    keyword_matched = True
                    break
            if not keyword_matched:
                outputs = self.tokenizer.batch_decode(sample_output_ids[:, -offset:], skip_special_tokens=False)[0]
                for keyword in self.keywords:
                    if keyword in outputs:
                        keyword_matched = True
                        break
            if not keyword_matched:
                all_matched = False
                break
        return all_matched