import math
from collections.abc import Sequence

import torch

from inference_engine.core.block import Block, BlockTable


class BlockManager:
    """Paged KV block manager with refcounted lifetime and prefix-cache lookup.

    This version adds the missing design semantics that are easy to overlook:
    - live block tables carry positive refcounts
    - shared blocks use copy-on-write when a write requires mutation
    - free-list reclamation happens only once the block is fully unreferenced
    - full-block prefix cache entries are hash-based and contiguous
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 0 or block_size <= 0:
            raise ValueError("num_blocks must be >=0 and block_size > 0")

        self.num_blocks: int = int(num_blocks)
        self.block_size: int = int(block_size)
        self.blocks: list[Block] = [Block(i) for i in range(self.num_blocks)]
        self.free_block: list[int] = list(range(self.num_blocks))
        self.block_table: BlockTable = BlockTable()

    @staticmethod
    def _hash_tokens(token_ids: Sequence[int]) -> int:
        return hash(tuple(int(token_id) for token_id in token_ids))

    def _reclaim_if_unreferenced(self, block_id: int) -> None:
        """Return a block to the free list once its refcount drops to zero."""
        if not 0 <= block_id < self.num_blocks:
            return

        block = self.blocks[block_id]
        if block.ref_count() != 0:
            return

        block_hash = block.block_hash()
        if block_hash is not None and self.block_table.get(block_hash) == block_id:
            self.block_table.remove(block_hash)

        block.reset()
        if block_id not in self.free_block:
            self.free_block.append(block_id)

    def allocate(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        """Allocate `n` free blocks and return a tensor of block ids (int32)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n > len(self.free_block):
            raise RuntimeError("not enough free blocks to allocate")

        ids = [self.free_block.pop() for _ in range(n)]
        for block_id in ids:
            self.blocks[block_id].inc_ref()
            self.blocks[block_id].touch()

        tensor = torch.tensor(ids, dtype=torch.int32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def fork(self, parent_block_table: Sequence[int]) -> list[int]:
        """Fork a block table with shared physical blocks, preserving refcounts."""
        shared = list(parent_block_table)
        for block_id in shared:
            if 0 <= int(block_id) < self.num_blocks:
                self.blocks[int(block_id)].inc_ref()
                self.blocks[int(block_id)].touch()
        return shared

    def write(self, seq_block_table: list[int], logical_idx: int, new_block_id: int) -> int:
        """Write a block id into a sequence table with copy-on-write semantics."""
        if logical_idx < 0 or logical_idx >= len(seq_block_table):
            raise IndexError("logical block index out of range")

        current_block_id = int(seq_block_table[logical_idx])
        if not 0 <= current_block_id < self.num_blocks:
            raise ValueError("invalid current block id")

        if self.blocks[current_block_id].ref_count() > 1:
            old_block_id = current_block_id
            self.blocks[old_block_id].dec_ref()
            self._reclaim_if_unreferenced(old_block_id)

            if new_block_id < 0 or new_block_id >= self.num_blocks:
                raise ValueError("new block id out of range")

            self.blocks[new_block_id].inc_ref()
            self.blocks[new_block_id].touch()
            seq_block_table[logical_idx] = int(new_block_id)
            return int(new_block_id)

        seq_block_table[logical_idx] = int(new_block_id)
        return int(new_block_id)

    def free(self, block_ids: Sequence[int]) -> None:
        """Release reference counts and reclaim blocks when they become unreferenced."""
        for raw_id in block_ids:
            block_id = int(raw_id)
            if not 0 <= block_id < self.num_blocks:
                continue

            block = self.blocks[block_id]
            if block.ref_count() > 0:
                block.dec_ref()
            else:
                block.reset()

            self._reclaim_if_unreferenced(block_id)

    def num_required_block(self, seq_len: int) -> int:
        return math.ceil(seq_len / self.block_size)

    def can_allocate(self, seq_len_or_num: int) -> bool:
        """Return True if there are enough free blocks for the given length or count."""
        if seq_len_or_num <= 0:
            return True
        if seq_len_or_num > self.block_size:
            needed = self.num_required_block(seq_len_or_num)
        else:
            needed = seq_len_or_num
        return needed <= len(self.free_block)

    def get_cached_block(self, block_hash: int) -> int | None:
        return self.block_table.get(block_hash)

    def cache_full_blocks(self, token_ids: Sequence[int]) -> int | None:
        """Cache a single full block represented by `token_ids`.

        Returns the block id used or None if no free block is available.
        """
        if len(token_ids) != self.block_size:
            raise ValueError("token_ids length must equal block_size to cache a full block")

        block_hash = self._hash_tokens(token_ids)
        existing = self.block_table.get(block_hash)
        if existing is not None:
            self.blocks[existing].touch()
            return existing

        if not self.free_block:
            return None

        block_id = self.free_block.pop()
        block = self.blocks[block_id]
        block.update_block(list(token_ids), block_hash=block_hash)
        block.inc_ref()
        self.block_table.add(block_hash, block_id)
        return block_id

    def try_get_prefix_blocks(self, token_ids: Sequence[int], num_cached_tokens: int) -> list[int]:
        """Return all cached contiguous full blocks for the requested prefix length."""
        if num_cached_tokens <= 0:
            return []

        max_tokens = min(len(token_ids), num_cached_tokens)
        result: list[int] = []
        num_blocks = self.num_required_block(max_tokens)
        for block_index in range(num_blocks):
            start = block_index * self.block_size
            end = start + self.block_size
            slice_tokens = token_ids[start:end]
            if len(slice_tokens) < self.block_size:
                break
            block_hash = self._hash_tokens(slice_tokens)
            cached = self.block_table.get(block_hash)
            if cached is None:
                break
            result.append(cached)
        return result

    def match_prefix(self, token_ids: Sequence[int]) -> int:
        """Return the number of tokens matched by the cached prefix."""
        matched_tokens = 0
        for block_index in range(self.num_required_block(len(token_ids))):
            start = block_index * self.block_size
            end = start + self.block_size
            slice_tokens = token_ids[start:end]
            if len(slice_tokens) < self.block_size:
                break
            block_hash = self._hash_tokens(slice_tokens)
            cached = self.block_table.get(block_hash)
            if cached is None:
                break
            matched_tokens += self.block_size
        return matched_tokens

    def swap_in_out(self, block_ids_in: Sequence[int], block_ids_out: Sequence[int]) -> None:
        """Simple swap hook that updates refcounts and timestamps without altering the pool."""
        for block_id in block_ids_out:
            block_id = int(block_id)
            if 0 <= block_id < self.num_blocks:
                self.blocks[block_id].dec_ref()
                self._reclaim_if_unreferenced(block_id)

        for block_id in block_ids_in:
            block_id = int(block_id)
            if 0 <= block_id < self.num_blocks:
                self.blocks[block_id].inc_ref()
                self.blocks[block_id].touch()

    def num_free(self) -> int:
        return len(self.free_block)

    def evict_unreferenced(self) -> list[int]:
        """Return unreferenced blocks sorted by last access for simple eviction."""
        candidates = [
            block_id
            for block_id in range(self.num_blocks)
            if self.blocks[block_id].ref_count() == 0 and block_id in self.free_block
        ]
        return sorted(candidates, key=lambda block_id: self.blocks[block_id].last_accessed())

    def validate_invariants(self) -> bool:
        """Cheap invariant checks for the common aliasing and bookkeeping mistakes."""
        live_block_ids = {
            block.block_id()
            for block in self.blocks
            if block.ref_count() > 0 or block.block_hash() is not None
        }
        for block_id in live_block_ids:
            block = self.blocks[block_id]
            if block.ref_count() < 0:
                return False
            if block.block_hash() is not None:
                canonical = self.block_table.get(block.block_hash())
                if canonical != block_id:
                    return False
        if len(self.free_block) != len(set(self.free_block)):
            return False
        if sum(1 for block in self.blocks if block.ref_count() > 0) + len(self.free_block) != self.num_blocks:
            return False
        return True

    @staticmethod
    def static_estimate_num_gpu_blocks(
        model_config: dict[str, int], block_size: int, gpu_mem_util: float
    ) -> int:
        """Return a rough estimate of how many blocks fit given a util value."""
        kv_bytes = model_config.get("kv_bytes_per_token", 64)
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        available_bytes = int(max(0.0, gpu_mem_util) * 1024 * 1024 * 1024)
        bytes_per_block = kv_bytes * block_size
        return max(0, available_bytes // bytes_per_block)
