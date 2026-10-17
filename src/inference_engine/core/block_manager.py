import math
import time
from collections.abc import Sequence

import torch

from inference_engine.core.block import Block, BlockTable


class BlockManager:
    """Simple block manager for allocating/freeing blocks and caching by hash.

    This implementation is intentionally lightweight and deterministic so it
    can be used by other modules that need block bookkeeping.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 0 or block_size <= 0:
            raise ValueError("num_blocks must be >=0 and block_size > 0")

        self.num_blocks: int = num_blocks
        self.block_size: int = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block: list[int] = list(range(num_blocks))
        self.block_table: BlockTable = BlockTable()

    def allocate(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """Allocate `n` free blocks and return a tensor of block ids (int32).

        Raises RuntimeError if insufficient free blocks.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        if n > len(self.free_block):
            raise RuntimeError("not enough free blocks to allocate")

        ids = [self.free_block.pop() for _ in range(n)]
        # mark as referenced
        for i in ids:
            self.blocks[i].ref_count = 1
            self.blocks[i].last_accessed = time.time()

        tensor = torch.tensor(ids, dtype=torch.int32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def free(self, block_ids: Sequence[int]) -> None:
        """Free the provided block ids and remove them from the block table."""
        for bid in block_ids:
            if 0 <= int(bid) < self.num_blocks:
                b = self.blocks[int(bid)]
                if b.block_hash is not None:
                    self.block_table.remove(b.block_hash)
                b.reset_block()
                if int(bid) not in self.free_block:
                    self.free_block.append(int(bid))

    def num_required_block(self, seq_len: int) -> int:
        return math.ceil(seq_len / self.block_size)

    def can_allocate(self, seq_len_or_num: int) -> bool:
        """Return True if there are enough free blocks for the given length or count."""
        # Accept either a sequence length or an explicit block count.
        if seq_len_or_num <= 0:
            return True
        # Heuristic: if value > block_size assume it's a seq_len, otherwise treat as count
        if seq_len_or_num > self.block_size:
            needed = self.num_required_block(seq_len_or_num)
        else:
            needed = seq_len_or_num
        return needed <= len(self.free_block)

    def get_cached_block(self, block_hash: int) -> int | None:
        return self.block_table.get(block_hash)

    def cache_full_blocks(self, token_ids: Sequence[int]) -> int | None:
        """Cache a single full block represented by `token_ids`.

        Returns the block id used or None if no free block available.
        """
        if len(token_ids) != self.block_size:
            raise ValueError("token_ids length must equal block_size to cache a full block")

        bh = hash(tuple(token_ids))
        existing = self.block_table.get(bh)
        if existing is not None:
            # touch existing
            self.blocks[existing].last_accessed = time.time()
            return existing

        if not self.free_block:
            return None

        bid = self.free_block.pop()
        self.blocks[bid].update_block(token_ids, bh)
        self.blocks[bid].inc_ref = 1
        self.block_table.add(bh, bid)
        return bid

    def try_get_prefix_blocks(self, token_ids: Sequence[int], num_cached_tokens: int) -> list[int]:
        """Try to retrieve cached block ids covering the first `num_cached_tokens` tokens.

        Returns list of block ids for consecutive blocks starting at token 0 that are cached.
        """
        if num_cached_tokens <= 0:
            return []
        result: list[int] = []
        num_blocks = self.num_required_block(num_cached_tokens)
        for b in range(num_blocks):
            start = b * self.block_size
            end = start + self.block_size
            slice_tokens = token_ids[start:end]
            if len(slice_tokens) < self.block_size:
                break
            bh = hash(tuple(slice_tokens))
            cached = self.block_table.get(bh)
            if cached is None:
                break
            result.append(cached)
        return result

    def swap_in_out(self, block_ids_in: Sequence[int], block_ids_out: Sequence[int]) -> None:
        """Placeholder for swapping blocks in/out of device memory.

        In this simple manager we just update timestamps and ref counts.
        """
        for bid in block_ids_out:
            if 0 <= int(bid) < self.num_blocks:
                self.blocks[int(bid)].ref_count = max(0, self.blocks[int(bid)].ref_count - 1)
        for bid in block_ids_in:
            if 0 <= int(bid) < self.num_blocks:
                self.blocks[int(bid)].ref_count += 1
                self.blocks[int(bid)].last_accessed = time.time()

    def num_free(self) -> int:
        return len(self.free_block)

    @staticmethod
    def static_estimate_num_gpu_blocks(
        model_config: dict[str, int], block_size: int, gpu_mem_util: float
    ) -> int:
        """Return a very rough estimate of how many blocks fit given a util value.

        This is a heuristic helper for sizing tests and benchmarks. It expects
        `model_config` to contain an approximate `kv_bytes_per_token` entry if
        available; otherwise falls back to a conservative default.
        """
        kv_bytes = model_config.get("kv_bytes_per_token", 64)
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        # available bytes approximated by util * 1GiB
        available_bytes = int(gpu_mem_util * 1024 * 1024 * 1024)
        bytes_per_block = kv_bytes * block_size
        return max(0, available_bytes // bytes_per_block)
