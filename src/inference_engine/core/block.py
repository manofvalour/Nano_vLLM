import time

import torch


class Block:
    """Lightweight container representing a cached block of tokens/kv data."""

    def __init__(self, block_id: int):
        self._block_id: int = block_id
        self._ref_count: int = 0
        self._last_accessed: float = time.time()
        self._block_hash: int | None = None
        self._data: torch.Tensor | None = None

    def block_id(self) -> int:
        return self._block_id

    def ref_count(self) -> int:
        return self._ref_count

    def touch(self) -> None:
        """updating the last time the block was accessed time"""
        self._last_accessed = time.time()

    def last_accessed(self) -> float:
        """retrieving when the block was last accessed"""
        return self._last_accessed

    def block_hash(self) -> int | None:
        return self._block_hash

    def data(self) -> torch.Tensor | None:
        return self._data

    def update_block(self, tensor: torch.Tensor, block_hash: int | None = None) -> None:
        self._data = tensor
        if block_hash is not None:
            self._block_hash = block_hash
        self.touch()  # update the timestamp

    def inc_ref(self) -> None:
        self._ref_count += 1

    def dec_ref(self) -> None:
        if self._ref_count > 0:
            self._ref_count -= 1

    def reset(self) -> None:
        self._ref_count = 0
        self._block_hash = None
        self._data = None
        self.touch()


class BlockTable:
    """Simple mapping from block_hash -> block id.
    (Mapping Logical Block to Physical memory Block)"""

    def __init__(self):
        self._hash_to_id: dict[int, int] = {}  # dict(block_hash, block_id)

    def add(self, block_hash: int, block_id: int) -> None:
        """Add a block to the block table"""
        self._hash_to_id[block_hash] = block_id

    def get(self, block_hash: int) -> int | None:
        """using the block_hash to retrieve the block"""
        return self._hash_to_id.get(block_hash)

    def remove(self, block_hash: int) -> None:
        """removing a block from the block table"""
        self._hash_to_id.pop(block_hash, None)

    def contains(self, block_hash: int) -> bool:
        """check if a block is in the block table"""
        return block_hash in self._hash_to_id


class BlockPool:
    """Pool of reusable blocks for KV-cache allocation.

    The pool tracks all known blocks and a free list for quick allocation.
    """

    def __init__(self, blocks: list[Block] | None = None):
        self.blocks: list[Block] = []
        self.free_list: list[int] = []

        if blocks is not None:
            for block in blocks:
                self.add_block(block)

    def _get_block_by_id(self, block_id: int) -> Block | None:
        for block in self.blocks:
            if block.block_id() == block_id:
                return block
        return None

    def add_block(self, block: Block) -> None:
        """Register a block in the pool and make it available for reuse."""
        if any(existing.block_id() == block.block_id() for existing in self.blocks):
            raise ValueError(f"block_id {block.block_id()} already exists in the pool")

        self.blocks.append(block)
        self.free_list.append(block.block_id())

    def allocate(self) -> Block | None:
        """Allocate the next free block from the pool."""
        if not self.free_list:
            return None

        block_id = self.free_list.pop(0)
        block = self._get_block_by_id(block_id)
        if block is None:
            return None

        block.inc_ref()
        return block

    def free(self, block: Block | int) -> None:
        """Return a block to the free list."""
        block_id = block.block_id() if isinstance(block, Block) else int(block)
        block_obj = self._get_block_by_id(block_id)

        if block_obj is not None:
            block_obj.dec_ref()

        if block_id not in self.free_list:
            self.free_list.append(block_id)

    def get_free_count(self) -> int:
        return len(self.free_list)

    def __len__(self) -> int:
        return len(self.blocks)
