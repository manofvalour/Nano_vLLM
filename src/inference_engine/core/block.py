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
        self._last_accessed = time.time()

    def last_accessed(self) -> float:
        return self._last_accessed

    def block_hash(self) -> int | None:
        return self._block_hash

    def data(self) -> torch.Tensor | None:
        return self._data

    def update_block(self, tensor: torch.Tensor, block_hash: int | None = None) -> None:
        self._data = tensor
        if block_hash is not None:
            self._block_hash = block_hash
        self.touch()

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
    """Simple mapping from block_hash -> block id."""

    def __init__(self):
        self._hash_to_id: dict[int, int] = {}

    def add(self, block_hash: int, block_id: int) -> None:
        self._hash_to_id[block_hash] = block_id

    def get(self, block_hash: int) -> int | None:
        return self._hash_to_id.get(block_hash)

    def remove(self, block_hash: int) -> None:
        self._hash_to_id.pop(block_hash, None)

    def contains(self, block_hash: int) -> bool:
        return block_hash in self._hash_to_id
