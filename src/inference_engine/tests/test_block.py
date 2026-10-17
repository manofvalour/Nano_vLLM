import sys,os

from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(Path(__file__).parent, "..", ".."))
from inference_engine.core.block import Block, BlockTable


def test_block_basic_lifecycle():
    b = Block(5)
    assert b.block_id() == 5
    assert b.ref_count() == 0

    # increment and decrement refs
    b.inc_ref()
    assert b.ref_count() == 1
    b.inc_ref()

    assert b.ref_count() == 2
    b.dec_ref()
    assert b.ref_count() == 1

    # update data and hash
    t = torch.randn(2, 3)
    old_time = b.last_accessed()
    b.update_block(t, block_hash=42)
    assert b.block_hash() == 42
    assert torch.equal(b.data(), t)
    assert b.last_accessed() >= old_time

    # reset clears
    b.reset()
    assert b.ref_count() == 0
    assert b.block_hash() is None
    assert b.data() is None


def test_block_table_basic():
    bt = BlockTable()
    assert not bt.contains(1)
    bt.add(1, 10)
    assert bt.contains(1)
    assert bt.get(1) == 10
    bt.remove(1)
    assert bt.get(1) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
