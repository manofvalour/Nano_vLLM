import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference_engine.core.block import Block, BlockPool, BlockTable


def test_block_basic_lifecycle():
    block = Block(5)

    assert block.block_id() == 5
    assert block.ref_count() == 0
    assert block.block_hash() is None
    assert block.data() is None
    assert isinstance(block.last_accessed(), float)

    block.inc_ref()
    assert block.ref_count() == 1
    block.inc_ref()
    assert block.ref_count() == 2
    block.dec_ref()
    assert block.ref_count() == 1

    data = torch.randn(2, 3)
    block.update_block(data, block_hash=42)

    assert block.block_hash() == 42
    assert torch.equal(block.data(), data)

    block.reset()

    assert block.ref_count() == 0
    assert block.block_hash() is None
    assert block.data() is None


def test_block_dec_ref_never_goes_negative():
    block = Block(0)

    block.dec_ref()
    assert block.ref_count() == 0

    block.inc_ref()
    block.dec_ref()
    block.dec_ref()
    assert block.ref_count() == 0


def test_block_touch_updates_last_accessed():
    block = Block(0)
    block._last_accessed = 0.0

    block.touch()

    assert block.last_accessed() > 0.0


def test_block_update_block_preserves_hash_when_not_provided():
    block = Block(0)
    block.update_block(torch.zeros(1), block_hash=7)

    block.update_block(torch.ones(2))

    assert block.block_hash() == 7
    assert torch.equal(block.data(), torch.ones(2))


def test_block_reset_keeps_id_and_updates_timestamp():
    block = Block(3)
    block.inc_ref()
    block.update_block(torch.zeros(1), block_hash=1)
    block._last_accessed = 0.0

    block.reset()

    assert block.block_id() == 3
    assert block.last_accessed() > 0.0


def test_blocks_track_independent_state():
    first = Block(0)
    second = Block(1)

    first.inc_ref()
    first.update_block(torch.ones(1), block_hash=1)

    assert first.ref_count() == 1
    assert second.ref_count() == 0
    assert second.data() is None
    assert second.block_hash() is None


def test_block_table_add_get_contains_and_remove():
    table = BlockTable()

    assert table.get(1) is None
    assert table.contains(1) is False

    table.add(1, 10)

    assert table.contains(1) is True
    assert table.get(1) == 10

    table.remove(1)

    assert table.get(1) is None
    assert table.contains(1) is False


def test_block_table_remove_missing_hash_is_noop():
    table = BlockTable()
    table.add(1, 10)

    table.remove(999)

    assert table.get(1) == 10


def test_block_table_overwrites_existing_hash():
    table = BlockTable()
    table.add(1, 10)
    table.add(1, 11)

    assert table.get(1) == 11


def test_block_pool_starts_empty():
    pool = BlockPool()

    assert len(pool) == 0
    assert pool.get_free_count() == 0
    assert pool.allocate() is None


def test_block_pool_registers_initial_blocks():
    blocks = [Block(0), Block(1)]
    pool = BlockPool(blocks)

    assert len(pool) == 2
    assert pool.get_free_count() == 2
    assert pool.free_list == [0, 1]
    assert pool.blocks == blocks


def test_block_pool_add_block_rejects_duplicate_ids():
    pool = BlockPool([Block(0)])

    with pytest.raises(ValueError):
        pool.add_block(Block(0))

    pool.add_block(Block(1))

    assert len(pool) == 2
    assert pool.get_free_count() == 2


def test_block_pool_allocates_fifo_and_increments_refs():
    blocks = [Block(0), Block(1), Block(2)]
    pool = BlockPool(blocks)

    first = pool.allocate()
    second = pool.allocate()

    assert first is blocks[0]
    assert second is blocks[1]
    assert first.ref_count() == 1
    assert second.ref_count() == 1
    assert pool.free_list == [2]
    assert pool.get_free_count() == 1


def test_block_pool_returns_none_when_exhausted():
    pool = BlockPool([Block(0)])

    assert pool.allocate() is not None
    assert pool.allocate() is None


def test_block_pool_free_by_block_and_by_id():
    pool = BlockPool([Block(0), Block(1)])
    block = pool.allocate()

    pool.free(block)

    assert block.ref_count() == 0
    assert pool.get_free_count() == 2
    assert 0 in pool.free_list

    pool.free(1)

    assert pool.get_free_count() == 2
    assert pool.free_list.count(1) == 1


def test_block_pool_free_is_idempotent():
    pool = BlockPool([Block(0), Block(1)])
    pool.allocate()

    pool.free(0)
    pool.free(0)

    assert pool.free_list.count(0) == 1
    assert pool.get_free_count() == 2


def test_block_pool_reuses_freed_blocks():
    pool = BlockPool([Block(0)])

    block = pool.allocate()
    pool.free(block)
    reused = pool.allocate()

    assert reused is block
    assert reused.ref_count() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
