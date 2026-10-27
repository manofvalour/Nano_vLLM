import sys
from pathlib import Path

import pytest
import torch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference_engine.attention.paged_attn import (
    BLOCK_SIZE,
    compute_slot_mapping,
    get_block_ids,
    paged_attention,
    reshape_and_cache,
    store_kvcache,
)


class TestPagedAttention:
    """Unit tests for paged attention kernel correctness."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block_size = BLOCK_SIZE
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 64
        self.max_context_len = 256

    def test_store_kvcache_basic(self):
        """Test basic KV cache storage functionality."""
        num_tokens = 10
        num_blocks = 5

        # Create test tensors
        key = torch.randn(num_tokens, self.num_kv_heads, self.head_dim, device=self.device)
        value = torch.randn(num_tokens, self.num_kv_heads, self.head_dim, device=self.device)
        key_cache = torch.zeros(
            num_blocks, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )
        value_cache = torch.zeros(
            num_blocks, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )

        # Create slot mapping (each token in a different block)
        slot_mapping = torch.arange(num_tokens, device=self.device)

        # Store in cache
        store_kvcache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            num_tokens,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )

        # Verify cache contents
        for i in range(num_tokens):
            block_id = i // self.block_size
            block_offset = i % self.block_size

            assert torch.allclose(key_cache[block_id, :, block_offset, :], key[i])
            assert torch.allclose(value_cache[block_id, :, block_offset, :], value[i])

    def test_store_kvcache_multiple_tokens_per_block(self):
        """Test storing multiple tokens in the same block."""
        num_tokens = 20  # 2 blocks worth of tokens
        num_blocks = 3

        # Create test tensors
        key = torch.randn(num_tokens, self.num_kv_heads, self.head_dim, device=self.device)
        value = torch.randn(num_tokens, self.num_kv_heads, self.head_dim, device=self.device)
        key_cache = torch.zeros(
            num_blocks, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )
        value_cache = torch.zeros(
            num_blocks, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )

        # Create slot mapping (first 16 tokens in block 0, next 4 in block 1)
        slot_mapping = torch.cat(
            [torch.arange(0, 16, device=self.device), torch.arange(16, 20, device=self.device)]
        )

        # Store in cache
        store_kvcache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            num_tokens,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )

        # Verify cache contents for first block
        for i in range(16):
            block_id = 0
            block_offset = i
            assert torch.allclose(key_cache[block_id, :, block_offset, :], key[i])
            assert torch.allclose(value_cache[block_id, :, block_offset, :], value[i])

        # Verify cache contents for second block
        for i in range(16, 20):
            block_id = 1
            block_offset = i - 16
            assert torch.allclose(key_cache[block_id, :, block_offset, :], key[i])
            assert torch.allclose(value_cache[block_id, :, block_offset, :], value[i])

    def test_paged_attention_basic(self):
        """Test basic paged attention computation."""
        num_seqs = 2
        seq_lens = torch.tensor([8, 12], device=self.device)

        # Create test tensors
        query = torch.randn(num_seqs, self.num_heads, self.head_dim, device=self.device)
        key_cache = torch.zeros(
            3, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )
        value_cache = torch.zeros(
            3, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )

        # Create block tables (each sequence uses 2 blocks)
        block_tables = torch.tensor(
            [
                [0, 1],  # Sequence 0: blocks 0 and 1
                [1, 2],  # Sequence 1: blocks 1 and 2
            ],
            device=self.device,
        )

        # Fill cache with test data
        for seq_id in range(num_seqs):
            seq_len = seq_lens[seq_id].item()
            for pos in range(seq_len):
                block_id = block_tables[seq_id, pos // self.block_size].item()
                block_offset = pos % self.block_size

                # Create key and value for this position
                key = torch.randn(self.num_kv_heads, self.head_dim, device=self.device)
                value = torch.randn(self.num_kv_heads, self.head_dim, device=self.device)

                key_cache[block_id, :, block_offset, :] = key
                value_cache[block_id, :, block_offset, :] = value

        # Compute attention
        output = paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            self.max_context_len,
        )

        # Verify output shape
        assert output.shape == query.shape

        # Verify output is not all zeros (attention should produce non-zero output)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_paged_attention_single_token(self):
        """Test paged attention with single token sequences."""
        num_seqs = 3
        seq_lens = torch.ones(num_seqs, dtype=torch.int32, device=self.device)

        # Create test tensors
        query = torch.randn(num_seqs, self.num_heads, self.head_dim, device=self.device)
        key_cache = torch.zeros(
            3, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )
        value_cache = torch.zeros(
            3, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )

        # Create block tables (each sequence uses block 0)
        block_tables = torch.zeros(num_seqs, 1, dtype=torch.int32, device=self.device)

        # Fill cache with test data
        for seq_id in range(num_seqs):
            block_id = 0
            block_offset = 0

            key = torch.randn(self.num_kv_heads, self.head_dim, device=self.device)
            value = torch.randn(self.num_kv_heads, self.head_dim, device=self.device)

            key_cache[block_id, :, block_offset, :] = key
            value_cache[block_id, :, block_offset, :] = value

        # Compute attention
        output = paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            self.max_context_len,
        )

        # Verify output shape
        assert output.shape == query.shape

    def test_compute_slot_mapping_basic(self):
        """Test basic slot mapping computation."""
        num_seqs = 2
        seq_lens = torch.tensor([8, 12], device=self.device)

        # Create block tables
        block_tables = torch.tensor(
            [
                [0, 1],  # Sequence 0: blocks 0 and 1
                [1, 2],  # Sequence 1: blocks 1 and 2
            ],
            device=self.device,
        )

        # Compute slot mapping
        slot_mapping = compute_slot_mapping(block_tables, seq_lens, num_seqs, self.block_size)

        # Verify shape
        assert slot_mapping.shape == (20,)  # 8 + 12 tokens

        # Verify slot mapping values
        expected_slots = []
        for seq_id in range(num_seqs):
            for pos in range(seq_lens[seq_id].item()):
                block_id = block_tables[seq_id, pos // self.block_size].item()
                block_offset = pos % self.block_size
                expected_slots.append(block_id * self.block_size + block_offset)

        assert torch.allclose(
            slot_mapping, torch.tensor(expected_slots, dtype=slot_mapping.dtype, device=self.device)
        )

    def test_get_block_ids_basic(self):
        """Test basic block ID retrieval."""
        num_seqs = 2
        seq_lens = torch.tensor([8, 12], device=self.device)

        # Create block tables
        block_tables = torch.tensor(
            [
                [0, 1],  # Sequence 0: blocks 0 and 1
                [1, 2],  # Sequence 1: blocks 1 and 2
            ],
            device=self.device,
        )

        # Get block IDs
        block_ids = get_block_ids(block_tables, seq_lens, num_seqs, self.block_size)

        # Verify shape
        assert block_ids.shape == (20,)  # 8 + 12 tokens

        # Verify block IDs
        expected_block_ids = []
        for seq_id in range(num_seqs):
            for pos in range(seq_lens[seq_id].item()):
                expected_block_ids.append(block_tables[seq_id, pos // self.block_size].item())

        assert torch.allclose(
            block_ids, torch.tensor(expected_block_ids, dtype=block_ids.dtype, device=self.device)
        )

    def test_reshape_and_cache(self):
        """Test reshape and cache functionality."""
        num_tokens = 10
        num_blocks = 3

        # Create test tensors
        key = torch.randn(num_tokens, self.num_kv_heads, self.head_dim, device=self.device)
        value = torch.randn(num_tokens, self.num_kv_heads, self.head_dim, device=self.device)
        key_cache = torch.zeros(
            num_blocks, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )
        value_cache = torch.zeros(
            num_blocks, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )
        slot_mapping = torch.arange(num_tokens, device=self.device)

        # Reshape and cache
        reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            num_tokens,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )

        # Verify cache contents
        for i in range(num_tokens):
            block_id = i // self.block_size
            block_offset = i % self.block_size

            assert torch.allclose(key_cache[block_id, :, block_offset, :], key[i])
            assert torch.allclose(value_cache[block_id, :, block_offset, :], value[i])

    def test_attention_consistency(self):
        """Test that attention produces consistent results with different inputs."""
        num_seqs = 2
        seq_lens = torch.tensor([4, 6], device=self.device)

        # Create test tensors
        query1 = torch.randn(num_seqs, self.num_heads, self.head_dim, device=self.device)
        query2 = torch.randn(num_seqs, self.num_heads, self.head_dim, device=self.device)

        key_cache = torch.zeros(
            3, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )
        value_cache = torch.zeros(
            3, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )

        # Create block tables
        block_tables = torch.tensor([[0, 1], [1, 2]], device=self.device)

        # Fill cache with test data
        for seq_id in range(num_seqs):
            seq_len = seq_lens[seq_id].item()
            for pos in range(seq_len):
                block_id = block_tables[seq_id, pos // self.block_size].item()
                block_offset = pos % self.block_size

                key = torch.randn(self.num_kv_heads, self.head_dim, device=self.device)
                value = torch.randn(self.num_kv_heads, self.head_dim, device=self.device)

                key_cache[block_id, :, block_offset, :] = key
                value_cache[block_id, :, block_offset, :] = value

        # Compute attention with different queries
        output1 = paged_attention(
            query1,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            self.max_context_len,
        )

        output2 = paged_attention(
            query2,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            self.max_context_len,
        )

        # Verify outputs are different (different queries should produce different outputs)
        assert not torch.allclose(output1, output2)

    def test_edge_case_empty_sequence(self):
        """Test edge case with empty sequences."""
        num_seqs = 2
        seq_lens = torch.tensor([0, 0], device=self.device)

        # Create test tensors
        query = torch.randn(num_seqs, self.num_heads, self.head_dim, device=self.device)
        key_cache = torch.zeros(
            1, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )
        value_cache = torch.zeros(
            1, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )

        # Create empty block tables
        block_tables = torch.zeros(num_seqs, 1, dtype=torch.int32, device=self.device)

        # Compute attention (should handle empty sequences gracefully)
        output = paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            self.max_context_len,
        )

        # Verify output shape
        assert output.shape == query.shape

        # Verify output is zeros (no attention computation for empty sequences)
        assert torch.allclose(output, torch.zeros_like(output))

    def test_block_size_variations(self):
        """Test with different block sizes."""
        for block_size in [2, 4, 8, 16, 32]:
            num_seqs = 2
            seq_lens = torch.tensor([block_size * 2, block_size * 3], device=self.device)

            # Create test tensors
            query = torch.randn(num_seqs, self.num_heads, self.head_dim, device=self.device)
            num_blocks = 5
            key_cache = torch.zeros(
                num_blocks, self.num_kv_heads, block_size, self.head_dim, device=self.device
            )
            value_cache = torch.zeros(
                num_blocks, self.num_kv_heads, block_size, self.head_dim, device=self.device
            )

            # Create block tables
            block_tables = torch.tensor([[0, 1, 1], [1, 2, 0]], device=self.device)

            # Fill cache with test data
            for seq_id in range(num_seqs):
                seq_len = seq_lens[seq_id].item()  # 4
                for pos in range(seq_len):
                    block_id = block_tables[seq_id, pos // block_size].item()
                    block_offset = pos % block_size

                    key = torch.randn(self.num_kv_heads, self.head_dim, device=self.device)
                    value = torch.randn(self.num_kv_heads, self.head_dim, device=self.device)

                    key_cache[block_id, :, block_offset, :] = key
                    value_cache[block_id, :, block_offset, :] = value

            # Compute attention
            output = paged_attention(
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                num_seqs,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                block_size,
                self.max_context_len,
            )

            # Verify output shape
            assert output.shape == query.shape
            assert not torch.allclose(output, torch.zeros_like(output))

    def test_reshape_and_cache_rewrites_duplicate_slots(self):
        """reshape_and_cache should behave like store_kvcache when slots repeat."""
        num_tokens = 4
        num_blocks = 2
        slot_mapping = torch.tensor([0, 0, 1, 1], device=self.device, dtype=torch.long)
        key = torch.randn(num_tokens, self.num_kv_heads, self.head_dim, device=self.device)
        value = torch.randn(num_tokens, self.num_kv_heads, self.head_dim, device=self.device)
        key_cache = torch.zeros(
            num_blocks, self.num_kv_heads, self.block_size, self.head_dim, device=self.device
        )
        value_cache = torch.zeros_like(key_cache)

        reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            num_tokens,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )

        assert torch.allclose(key_cache[0, :, 0, :], key[1])
        assert torch.allclose(value_cache[0, :, 0, :], value[1])
        assert torch.allclose(key_cache[0, :, 1, :], key[3])
        assert torch.allclose(value_cache[0, :, 1, :], value[3])

    def test_compute_slot_mapping_with_sparse_non_contiguous_blocks(self):
        """Slot mapping should respect arbitrary block tables and sequence lengths."""
        num_seqs = 2
        seq_lens = torch.tensor([5, 3], device=self.device, dtype=torch.int32)
        block_tables = torch.tensor(
            [
                [2, 7, 4],  # seq 0 -> blocks 2, 7, 4
                [1, 1, 5],  # seq 1 -> blocks 1, 1, 5
            ],
            device=self.device,
            dtype=torch.int32,
        )

        slot_mapping = compute_slot_mapping(block_tables, seq_lens, num_seqs, self.block_size)

        expected = []
        for seq_id in range(num_seqs):
            for pos in range(seq_lens[seq_id].item()):
                block_id = block_tables[seq_id, pos // self.block_size].item()
                block_offset = pos % self.block_size
                expected.append(block_id * self.block_size + block_offset)

        assert slot_mapping.shape == (seq_lens.sum().item(),)
        assert torch.equal(
            slot_mapping, torch.tensor(expected, device=self.device, dtype=slot_mapping.dtype)
        )

    def test_get_block_ids_with_sparse_non_contiguous_blocks(self):
        """get_block_ids should return one block per token in order, regardless of block IDs."""
        num_seqs = 2
        seq_lens = torch.tensor([5, 3], device=self.device, dtype=torch.int32)
        block_tables = torch.tensor(
            [
                [2, 7, 4],
                [1, 1, 5],
            ],
            device=self.device,
            dtype=torch.int32,
        )

        block_ids = get_block_ids(block_tables, seq_lens, num_seqs, self.block_size)

        expected = []
        for seq_id in range(num_seqs):
            for pos in range(seq_lens[seq_id].item()):
                expected.append(block_tables[seq_id, pos // self.block_size].item())

        assert block_ids.shape == (seq_lens.sum().item(),)
        assert torch.equal(
            block_ids, torch.tensor(expected, device=self.device, dtype=block_ids.dtype)
        )

    def test_paged_attention_multiple_sequences_are_independent(self):
        """Each sequence should attend only to its own context length."""
        num_seqs = 2
        seq_lens = torch.tensor([2, 4], device=self.device, dtype=torch.int32)
        num_heads = 2
        num_kv_heads = 2
        head_dim = 4
        block_size = 4

        query = torch.randn(num_seqs, num_heads, head_dim, device=self.device)
        key_cache = torch.zeros((3, num_kv_heads, block_size, head_dim), device=self.device)
        value_cache = torch.zeros_like(key_cache)

        block_tables = torch.tensor([[0], [1]], device=self.device, dtype=torch.int32)
        seq0_key = torch.randn(seq_lens[0].item(), num_kv_heads, head_dim, device=self.device)
        seq0_val = torch.randn(seq_lens[0].item(), num_kv_heads, head_dim, device=self.device)
        seq1_key = torch.randn(seq_lens[1].item(), num_kv_heads, head_dim, device=self.device)
        seq1_val = torch.randn(seq_lens[1].item(), num_kv_heads, head_dim, device=self.device)

        key_cache[0, :, : seq_lens[0].item(), :] = seq0_key.transpose(0, 1)
        value_cache[0, :, : seq_lens[0].item(), :] = seq0_val.transpose(0, 1)
        key_cache[1, :, : seq_lens[1].item(), :] = seq1_key.transpose(0, 1)
        value_cache[1, :, : seq_lens[1].item(), :] = seq1_val.transpose(0, 1)

        output = paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            self.max_context_len,
        )

        assert output.shape == query.shape
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_compute_slot_mapping_handles_repeated_blocks(self):
        """Repeated block IDs inside a block table should still produce correct flattened slots."""
        num_seqs = 1
        seq_lens = torch.tensor([8], device=self.device, dtype=torch.int32)
        block_tables = torch.tensor([[1, 1, 1, 1]], device=self.device, dtype=torch.int32)

        slot_mapping = compute_slot_mapping(block_tables, seq_lens, num_seqs, self.block_size)

        expected = []
        for pos in range(seq_lens[0].item()):
            block_id = block_tables[0, pos // self.block_size].item()
            expected.append(block_id * self.block_size + (pos % self.block_size))

        assert torch.equal(
            slot_mapping, torch.tensor(expected, device=self.device, dtype=slot_mapping.dtype)
        )

    def test_get_block_ids_handles_repeated_blocks(self):
        """get_block_ids should return repeated block IDs when the block table repeats blocks."""
        num_seqs = 1
        seq_lens = torch.tensor([8], device=self.device, dtype=torch.int32)
        block_tables = torch.tensor([[3, 3, 3, 3]], device=self.device, dtype=torch.int32)

        block_ids = get_block_ids(block_tables, seq_lens, num_seqs, self.block_size)

        expected = [3] * seq_lens[0].item()
        assert torch.equal(
            block_ids, torch.tensor(expected, device=self.device, dtype=block_ids.dtype)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
