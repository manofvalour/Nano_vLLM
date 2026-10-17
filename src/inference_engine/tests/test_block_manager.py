import torch
from pathlib import Path
import os, sys
import pytest

sys.path.insert(0, os.path.join(Path(__file__).parent, "..", ".."))
from inference_engine.core.block_manager import BlockManager


def test_block_manager_basic_allocation_and_query():
    bm = BlockManager(num_blocks=8, block_size=4)

    # num_required_block
    assert bm.num_required_block(1) == 1
    assert bm.num_required_block(4) == 1
    assert bm.num_required_block(5) == 2

    # can_allocate by length and by count
    assert bm.can_allocate(1)
    assert bm.can_allocate(8) is True
    assert bm.can_allocate(100) is False

    # num_free initially equals total
    assert bm.num_free() == 8

    # allocate a few blocks
    ids = bm.allocate(3)
    assert ids.numel() == 3
    assert bm.num_free() == 5

    # static estimate uses model config
    est = BlockManager.static_estimate_num_gpu_blocks({"kv_bytes_per_token": 32}, 4, 0.01)
    assert isinstance(est, int)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])