import math

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class BlockShuffleDistributedSampler(Sampler):
    """Shuffle local index blocks while keeping each rank's disk reads local.

    A fully random DistributedSampler turns millions of small PNG reads on a
    hard disk into random I/O. This sampler still changes sample order every
    epoch, but assigns whole shuffled blocks to ranks and optionally shuffles
    samples inside each block. Each rank gets the same number of samples.
    """

    def __init__(
        self,
        dataset,
        block_size=256,
        num_replicas=None,
        rank=None,
        seed=0,
        shuffle_within_block=True,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} must be in [0, {num_replicas})")

        self.dataset = dataset
        self.block_size = int(block_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle_within_block = bool(shuffle_within_block)
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        blocks = [
            list(range(start, min(start + self.block_size, len(self.dataset))))
            for start in range(0, len(self.dataset), self.block_size)
        ]
        order = torch.randperm(len(blocks), generator=generator).tolist()
        rank_blocks = order[self.rank :: self.num_replicas]

        indices = []
        for block_id in rank_blocks:
            block = blocks[block_id]
            if self.shuffle_within_block and len(block) > 1:
                permutation = torch.randperm(len(block), generator=generator).tolist()
                block = [block[i] for i in permutation]
            indices.extend(block)

        if len(indices) < self.num_samples:
            indices.extend(indices[: self.num_samples - len(indices)])
        return iter(indices[: self.num_samples])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
