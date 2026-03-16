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
from torch.utils.data import Sampler
import random
from collections import defaultdict
from tqdm import tqdm

class TypeAwareBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, world_size, shuffle=True, seed=2025):
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        self.type_indices = defaultdict(list)
        for idx in range(len(dataset)):
            item_type = dataset.type_list[idx]
            self.type_indices[item_type].append(idx)
        
        if torch.distributed.get_rank() == 0:
            print("=" * 50)
            print(f"All Types")
            for key, values in self.type_indices.items():
                print(f"{key}: {len(values)}")
            print("=" * 50)

        self._generate_batches()

    def _generate_batches(self):
        global_batches = []
        global_batch_size = self.batch_size * self.world_size
        
        for item_type, indices in self.type_indices.items():
            assert len(indices) >= global_batch_size
            remainder = len(indices) % global_batch_size
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                rand_indices = torch.randperm(len(indices) - remainder, generator=g).tolist()
                if remainder != 0:
                    rand_indices = rand_indices + list(range(len(indices) - remainder, len(indices))) + rand_indices[:global_batch_size - remainder]
                self.type_indices[item_type] = [indices[i] for i in rand_indices]
        
        for item_type, indices in self.type_indices.items():
            for i in range(0, len(indices), global_batch_size):
                global_batch = indices[i:i+global_batch_size]
                global_batches.append(global_batch)
                
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 42)
            global_indices = torch.randperm(len(global_batches), generator=g).tolist()
            global_batches = [global_batches[i] for i in global_indices]
        
        self.process_batches = []
        for i, global_batch in enumerate(global_batches):
            self.process_batches += global_batch
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        self._generate_batches()
    
    def __iter__(self):
        for batch in self.process_batches:
            yield batch
    
    def __len__(self):
        return len(self.process_batches)    