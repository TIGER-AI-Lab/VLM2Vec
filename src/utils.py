import torch
import numpy as np
from src.logging import get_logger
logger = get_logger(__name__)

# Implement Union-Find operator for constructing ui patches
class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parent[py] = px

def get_select_mask(tensor, skip_ratio=0, rand=False):
    # Use tensor operations for efficiency
    retain_mask = (tensor == -1).clone()
    unique_vals, counts = torch.unique(tensor, return_counts=True)

    for i, (val, count) in enumerate(zip(unique_vals, counts)):
        if val == -1:
            continue
        positions = (tensor == val).nonzero(as_tuple=True)[0]
        num_positions = len(positions)
        
        if num_positions == 1:
            retain_mask[positions] = True
        else:
            num_to_skip = int(round(num_positions * skip_ratio))
            num_to_retain = max(1, num_positions - num_to_skip)
            if rand:
                # rand means random select subset of selective tokens for layer-wise
                perm = torch.randperm(num_positions, device=tensor.device)
                positions_to_retain = positions[perm[:num_to_retain]]
            else:
                indices = torch.linspace(0, num_positions - 1, steps=num_to_retain).long()
                positions_to_retain = positions[indices]
                
            retain_mask[positions_to_retain] = True
    return retain_mask

def print_rank(message):
    """If distributed is initialized, print the rank."""
    if torch.distributed.is_initialized():
        logger.info(f'rank{torch.distributed.get_rank()}: ' + message)
    else:
        logger.info(message)


def print_master(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)
