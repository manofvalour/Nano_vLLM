import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference_engine.core.block_manager import BlockManager


def test_block_manager_initial_state_and_validation():
    manager = BlockManager(num_blocks=8, block_size=4)

    assert manager.num_blocks == 8
    assert manager.block_size == 4
    assert len(manager.blocks) == 8
    assert [block.block_id() for block in manager.blocks] == list(range(8))
    assert manager.free_block == list(range(8))
    assert manager.num_free() == 8
    assert manager.block_table.get(1) is None

    with pytest.raises(ValueError):
        BlockManager(num_blocks=-1, block_size=4)
    with pytest.raises(ValueError):
        BlockManager(num_blocks=4, block_size=0)

    assert BlockManager(num_blocks=0, block_size=4).num_free() == 0


def test_num_required_block():
    manager = BlockManager(num_blocks=8, block_size=4)

    assert manager.num_required_block(0) == 0
    assert manager.num_required_block(1) == 1
    assert manager.num_required_block(4) == 1
    assert manager.num_required_block(5) == 2
    assert manager.num_required_block(16) == 4


def test_can_allocate_accepts_counts_and_lengths():
    manager = BlockManager(num_blocks=4, block_size=4)

    assert manager.can_allocate(0) is True
    assert manager.can_allocate(-1) is True
    assert manager.can_allocate(4) is True
    assert manager.can_allocate(5) is True
    assert manager.can_allocate(16) is True
    assert manager.can_allocate(17) is False

    manager.allocate(3)

    assert manager.can_allocate(1) is True
    assert manager.can_allocate(2) is False
    assert manager.can_allocate(5) is False


def test_allocate_returns_int32_tensor_and_updates_state():
    manager = BlockManager(num_blocks=8, block_size=4)

    block_ids = manager.allocate(3)

    assert isinstance(block_ids, torch.Tensor)
    assert block_ids.dtype == torch.int32
    assert block_ids.tolist() == [7, 6, 5]
    assert manager.num_free() == 5
    assert manager.free_block == [0, 1, 2, 3, 4]
    for block_id in block_ids.tolist():
        assert manager.blocks[block_id].ref_count() == 1


def test_allocate_zero_and_allocation_errors():
    manager = BlockManager(num_blocks=4, block_size=2)

    empty = manager.allocate(0)

    assert empty.numel() == 0
    assert empty.dtype == torch.int32
    assert manager.num_free() == 4

    with pytest.raises(ValueError):
        manager.allocate(-1)
    with pytest.raises(RuntimeError):
        manager.allocate(5)


def test_allocate_honors_device_argument():
    manager = BlockManager(num_blocks=2, block_size=2)
    device = torch.device("cpu")

    block_ids = manager.allocate(2, device=device)

    assert block_ids.device == device


def test_free_returns_blocks_and_resets_them():
    manager = BlockManager(num_blocks=4, block_size=2)
    block_ids = manager.allocate(2).tolist()

    manager.free(block_ids)

    assert manager.num_free() == 4
    assert manager.free_block == [0, 1, 3, 2]
    for block_id in block_ids:
        assert manager.blocks[block_id].ref_count() == 0


def test_free_is_idempotent_and_ignores_invalid_ids():
    manager = BlockManager(num_blocks=4, block_size=2)

    manager.free([0])
    manager.free([0])

    assert manager.free_block.count(0) == 1
    assert manager.num_free() == 4

    manager.free([99])

    assert manager.num_free() == 4


def test_freed_blocks_can_be_allocated_again():
    manager = BlockManager(num_blocks=2, block_size=4)

    first = manager.allocate(2).tolist()
    manager.free(first)
    second = manager.allocate(2).tolist()

    assert sorted(second) == sorted(first)


def test_cache_full_blocks_requires_full_block():
    manager = BlockManager(num_blocks=4, block_size=4)

    with pytest.raises(ValueError):
        manager.cache_full_blocks([1, 2, 3])


def test_cache_full_blocks_stores_block_and_hash():
    manager = BlockManager(num_blocks=4, block_size=4)
    tokens = [1, 2, 3, 4]
    expected_hash = hash(tuple(tokens))

    block_id = manager.cache_full_blocks(tokens)

    assert block_id == 3
    assert manager.num_free() == 3
    assert manager.get_cached_block(expected_hash) == block_id
    assert manager.block_table.contains(expected_hash) is True
    assert manager.blocks[block_id].block_hash() == expected_hash
    assert manager.blocks[block_id].data() == tokens
    assert manager.blocks[block_id].ref_count() == 1


def test_cache_full_blocks_deduplicates_existing_content():
    manager = BlockManager(num_blocks=4, block_size=4)
    tokens = [1, 2, 3, 4]
    block_id = manager.cache_full_blocks(tokens)

    cached_again = manager.cache_full_blocks(tokens)

    assert cached_again == block_id
    assert manager.num_free() == 3
    assert manager.blocks[block_id].ref_count() == 1


def test_cache_full_blocks_returns_none_when_exhausted():
    manager = BlockManager(num_blocks=1, block_size=4)

    assert manager.cache_full_blocks([1, 2, 3, 4]) is not None
    assert manager.cache_full_blocks([5, 6, 7, 8]) is None


def test_get_cached_block_returns_none_for_unknown_hash():
    manager = BlockManager(num_blocks=4, block_size=4)

    assert manager.get_cached_block(12345) is None


def test_free_removes_cached_hash_from_block_table():
    manager = BlockManager(num_blocks=4, block_size=4)
    tokens = [1, 2, 3, 4]
    block_hash = hash(tuple(tokens))
    block_id = manager.cache_full_blocks(tokens)

    manager.free([block_id])

    assert manager.get_cached_block(block_hash) is None
    assert manager.block_table.contains(block_hash) is False
    assert manager.blocks[block_id].block_hash() is None
    assert manager.blocks[block_id].data() is None
    assert manager.num_free() == 4


def test_try_get_prefix_blocks_returns_cached_prefix():
    manager = BlockManager(num_blocks=8, block_size=4)
    first_tokens = [1, 2, 3, 4]
    second_tokens = [5, 6, 7, 8]
    first_id = manager.cache_full_blocks(first_tokens)
    second_id = manager.cache_full_blocks(second_tokens)

    assert manager.try_get_prefix_blocks(first_tokens + second_tokens, 0) == []
    assert manager.try_get_prefix_blocks(first_tokens + second_tokens, 4) == [first_id]
    assert manager.try_get_prefix_blocks(first_tokens + second_tokens, 8) == [
        first_id,
        second_id,
    ]


def test_try_get_prefix_blocks_stops_at_partial_or_missing_block():
    manager = BlockManager(num_blocks=8, block_size=4)
    first_id = manager.cache_full_blocks([1, 2, 3, 4])

    assert manager.try_get_prefix_blocks([1, 2, 3, 4, 5, 6], 6) == [first_id]
    assert manager.try_get_prefix_blocks([9, 10, 11, 12], 4) == []
    assert manager.try_get_prefix_blocks([1, 2, 3], 3) == []


def test_swap_in_out_updates_ref_counts():
    manager = BlockManager(num_blocks=4, block_size=2)
    manager.allocate(2)

    manager.swap_in_out([0], [3])

    assert manager.blocks[0].ref_count() == 1
    assert manager.blocks[3].ref_count() == 0

    manager.swap_in_out([0], [3])

    assert manager.blocks[0].ref_count() == 2
    assert manager.blocks[3].ref_count() == 0


def test_swap_in_out_touches_blocks_and_ignores_invalid_ids():
    manager = BlockManager(num_blocks=4, block_size=2)
    manager.blocks[0]._last_accessed = 0.0

    manager.swap_in_out([0, 99], [-1])

    assert manager.blocks[0].last_accessed() > 0.0


def test_static_estimate_num_gpu_blocks():
    config = {"kv_bytes_per_token": 32}
    expected = (int(0.01 * 1024**3)) // (32 * 4)

    assert BlockManager.static_estimate_num_gpu_blocks(config, 4, 0.01) == expected
    assert BlockManager.static_estimate_num_gpu_blocks({}, 16, 1.0) == (1024**3) // (64 * 16)
    assert BlockManager.static_estimate_num_gpu_blocks({}, 16, 0.0) == 0
    assert BlockManager.static_estimate_num_gpu_blocks({}, 16, -1.0) == 0

    with pytest.raises(ValueError):
        BlockManager.static_estimate_num_gpu_blocks(config, 0, 0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
