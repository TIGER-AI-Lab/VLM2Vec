import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Iterator

@dataclass
class HomogeneousSampler(torch.utils.data.sampler.RandomSampler):
    """
    Sampler used when training on multiple datasets to ensure each 
    batch only contains samples from one dataset for the majority of cases.
    """
    total_batch_size: int = 8
    ds_lens: List[int] = None
    _num_samples: int = None
    data_source: Dataset = None
    replacement: bool = False
    ordered_datasets: bool = False
    same_datasets: bool = False
    random_datasets: bool = False
    num_repeats: int = 1
    chunk_size: int = None

    def __len__(self):
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        
        if not hasattr(self, "generator") or self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # We have multiple datasets each with a different number of samples
        # e.g. [100, 150, 50]
        # We would like to sample from them such that as much as possible each batch
        # only has samples from the same dataset.
        # For example if our batch size is 4 then
        # indices might be [0,1,2,3,100,101,102,103,150,151,152,153,50,51,52,53]
        # To do so:
        # 1. Shuffle the indices of each dataset separately
        # 2. Create batches with only samples from one dataset
        # 3. Keep the remaining samples which do not fit into a batch separate
        # 4. Then create mixed batches from the remaining samples
        # 5. Then yield randomly from all the batches
        # Testing:
        # ds_lens = [100, 150, 50]
        # batch_size = 8
        # Create random indices for each dataset

        if(self.ordered_datasets):
            print(">>>>>>>>>>>>>>>>>>>>>>Asking for ordered datasets", flush=True)
            ds_indices = []
            for n in self.ds_lens:
                assert n % self.chunk_size == 0, f"Dataset length {n} is not divisible by chunk_size"
                chunks = torch.arange(n).view(-1, self.chunk_size)  # shape (n // 16, 16)
                perm = torch.randperm(chunks.size(0))  # randomize chunks
                shuffled_chunks = chunks[perm]
                ds_indices.append(shuffled_chunks.flatten().tolist())

            ds_indices = [[i + sum(self.ds_lens[:j]) for i in ds_indices[j]] for j in range(len(self.ds_lens))]
            ds_batches = [list(torch.split(torch.tensor(ds_indices[j]), self.total_batch_size)) for j in range(len(self.ds_lens))]
        elif(self.same_datasets):
            ds_indices = [torch.randperm(n, generator=generator).tolist() for n in self.ds_lens]
            ds_indices = [[i + sum(self.ds_lens[:j]) for i in ds_indices[j]] for j in range(len(self.ds_lens))]
            ds_batches = [list(torch.split(torch.tensor(ds_indices[j]), self.total_batch_size)) for j in range(len(self.ds_lens))]
        elif(self.random_datasets):
            total_len = sum(self.ds_lens)
            shuffled_indices = torch.randperm(total_len, generator=generator)
            ds_batches = [list(torch.split(shuffled_indices, self.total_batch_size))]
        else:
            assert False, "none of odibn,sdibn,rdibn found"
        incomplete_indices = []

        for b in ds_batches:
            if len(b[-1]) < self.total_batch_size:
                incomplete_indices.append(b.pop())

        # import ipdb; ipdb.set_trace()
        # assert len(incomplete_indices)==0
        # if incomplete_indices:
        if False:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=generator).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(torch.tensor(incomplete_indices), self.total_batch_size))
            if len(mixed_batches[-1]) < self.total_batch_size:
                mixed_batches.pop()
            # Merge all batches to look like [...tensor([259, 273, 284, 289]), tensor([262, 280, 295, 258]), ...]
            ds_batches = sum(ds_batches, []) + mixed_batches
            logger.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches) - len(mixed_batches)} single-dataset batches & {len(mixed_batches)} mixed dataset batches.")
        else:
            ds_batches = sum(ds_batches, [])
            logger.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches)} single-dataset batches.")


        all_batches = []  # Start empty

        for _ in range(self.num_repeats):  # Iterate K times to include the first epoch as well
            epoch_order = torch.randperm(len(ds_batches), generator=generator).tolist()
            shuffled_batches = [ds_batches[i] for i in epoch_order]  # Keep batch structure
            all_batches.append(shuffled_batches)

        
        # Flatten across all epochs
        ds_batches = [batch for epoch_batches in all_batches for batch in epoch_batches]

        # Convert to list of indices
        ds_batches = [int(i) for i in torch.cat(ds_batches).tolist()]

        # Yield the indices
        yield from ds_batches