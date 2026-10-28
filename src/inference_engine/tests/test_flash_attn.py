import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference_engine.attention.flash_attn import (
    FlashAttentionWrapper,
    flash_attn_decode,
    flash_attn_paged_decode,
    flash_attn_prefill,
)

DTYPE = torch.float32
ATOL = 1e-5


def reference_prefill(q, k, v, cu_seqlens_q, cu_seqlens_k, causal, scale):
    group_size = q.shape[1] // k.shape[1]
    batch_size = len(cu_seqlens_q) - 1
    outputs = []

    for index in range(batch_size):
        q_start, q_end = cu_seqlens_q[index], cu_seqlens_q[index + 1]
        k_start, k_end = cu_seqlens_k[index], cu_seqlens_k[index + 1]
        query = q[q_start:q_end].transpose(0, 1)
        q_len, k_len = query.shape[-2], k_end - k_start
        if q_len == 0:
            outputs.append(q.new_zeros(0, q.shape[1], q.shape[2]))
            continue

        key = k[k_start:k_end].repeat_interleave(group_size, dim=1).transpose(0, 1)
        value = v[k_start:k_end].repeat_interleave(group_size, dim=1).transpose(0, 1)

        scores = query @ key.transpose(-1, -2) * scale
        if causal:
            query_positions = torch.arange(q_len).unsqueeze(1) + (k_len - q_len)
            key_positions = torch.arange(k_len).unsqueeze(0)
            mask = key_positions <= query_positions
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        outputs.append((weights @ value).transpose(0, 1))

    return torch.cat(outputs, dim=0)


def reference_decode(q, keys, values, context_lens, scale):
    group_size = q.shape[1] // keys.shape[2]
    outputs = []

    for index in range(q.shape[0]):
        context_len = int(context_lens[index])
        if context_len == 0:
            outputs.append(q.new_zeros(q.shape[1], q.shape[2]))
            continue

        query = q[index]
        key = keys[index][:context_len].repeat_interleave(group_size, dim=1)
        value = values[index][:context_len].repeat_interleave(group_size, dim=1)
        scores = torch.einsum("hd,lhd->hl", query, key) * scale
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hl,lhd->hd", weights, value))

    return torch.stack(outputs) if outputs else q.new_zeros(q.shape)


def build_paged_cache(keys, values, block_ids, block_size, num_blocks):
    num_kv_heads = keys.shape[1]
    head_dim = keys.shape[2]
    k_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim, dtype=keys.dtype)
    v_cache = torch.zeros_like(k_cache)

    for position in range(keys.shape[0]):
        block_id = block_ids[position // block_size]
        offset = position % block_size
        k_cache[block_id, :, offset, :] = keys[position]
        v_cache[block_id, :, offset, :] = values[position]

    return k_cache, v_cache


@pytest.mark.parametrize("causal", [False, True])
def test_prefill_single_sequence_matches_reference(causal):
    torch.manual_seed(0)
    q = torch.randn(6, 4, 8, dtype=DTYPE)
    k = torch.randn(6, 2, 8, dtype=DTYPE)
    v = torch.randn(6, 2, 8, dtype=DTYPE)
    cu_seqlens = [0, 6]
    scale = 8**-0.5

    output = flash_attn_prefill(q, k, v, cu_seqlens, cu_seqlens, causal=causal)
    expected = reference_prefill(q, k, v, cu_seqlens, cu_seqlens, causal, scale)

    assert output.shape == (6, 4, 8)
    assert output.dtype == DTYPE
    assert torch.allclose(output, expected, atol=ATOL)


def test_prefill_ragged_batch_matches_reference():
    torch.manual_seed(1)
    q = torch.randn(8, 4, 8, dtype=DTYPE)
    k = torch.randn(8, 2, 8, dtype=DTYPE)
    v = torch.randn(8, 2, 8, dtype=DTYPE)
    cu_seqlens = [0, 3, 8]
    scale = 8**-0.5

    output = flash_attn_prefill(q, k, v, cu_seqlens, cu_seqlens, causal=True)
    expected = reference_prefill(q, k, v, cu_seqlens, cu_seqlens, True, scale)

    assert torch.allclose(output, expected, atol=ATOL)


def test_prefill_chunked_queries_attend_to_key_prefix():
    torch.manual_seed(2)
    q = torch.randn(5, 4, 8, dtype=DTYPE)
    k = torch.randn(10, 2, 8, dtype=DTYPE)
    v = torch.randn(10, 2, 8, dtype=DTYPE)
    cu_seqlens_q = [0, 2, 5]
    cu_seqlens_k = [0, 6, 10]
    scale = 8**-0.5

    output = flash_attn_prefill(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=3,
        max_seqlen_k=6,
        causal=True,
    )
    expected = reference_prefill(q, k, v, cu_seqlens_q, cu_seqlens_k, True, scale)

    assert torch.allclose(output, expected, atol=ATOL)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_prefill_supports_grouped_query_attention(num_kv_heads):
    torch.manual_seed(3)
    q = torch.randn(7, 4, 8, dtype=DTYPE)
    k = torch.randn(7, num_kv_heads, 8, dtype=DTYPE)
    v = torch.randn(7, num_kv_heads, 8, dtype=DTYPE)
    cu_seqlens = [0, 7]
    scale = 8**-0.5

    output = flash_attn_prefill(q, k, v, cu_seqlens, cu_seqlens, causal=True)
    expected = reference_prefill(q, k, v, cu_seqlens, cu_seqlens, True, scale)

    assert torch.allclose(output, expected, atol=ATOL)


def test_prefill_uses_custom_softmax_scale():
    torch.manual_seed(4)
    q = torch.randn(5, 4, 8, dtype=DTYPE)
    k = torch.randn(5, 2, 8, dtype=DTYPE)
    v = torch.randn(5, 2, 8, dtype=DTYPE)
    cu_seqlens = [0, 5]

    output = flash_attn_prefill(q, k, v, cu_seqlens, cu_seqlens, softmax_scale=0.3)
    expected = reference_prefill(q, k, v, cu_seqlens, cu_seqlens, False, 0.3)

    assert torch.allclose(output, expected, atol=ATOL)


def test_prefill_default_scale_is_inverse_sqrt_head_dim():
    torch.manual_seed(5)
    q = torch.randn(4, 2, 16, dtype=DTYPE)
    k = torch.randn(4, 2, 16, dtype=DTYPE)
    v = torch.randn(4, 2, 16, dtype=DTYPE)
    cu_seqlens = [0, 4]

    default = flash_attn_prefill(q, k, v, cu_seqlens, cu_seqlens)
    explicit = flash_attn_prefill(q, k, v, cu_seqlens, cu_seqlens, softmax_scale=16**-0.5)

    assert torch.allclose(default, explicit, atol=ATOL)


def test_prefill_handles_empty_sequences():
    torch.manual_seed(6)
    q = torch.randn(3, 4, 8, dtype=DTYPE)
    k = torch.randn(3, 2, 8, dtype=DTYPE)
    v = torch.randn(3, 2, 8, dtype=DTYPE)
    cu_seqlens = [0, 0, 3]
    scale = 8**-0.5

    output = flash_attn_prefill(q, k, v, cu_seqlens, cu_seqlens)
    expected = reference_prefill(q, k, v, cu_seqlens, cu_seqlens, False, scale)

    assert output.shape == (3, 4, 8)
    assert torch.allclose(output, expected, atol=ATOL)


def test_prefill_with_no_queries_returns_empty_tensor():
    q = torch.zeros(0, 4, 8, dtype=DTYPE)
    k = torch.zeros(0, 2, 8, dtype=DTYPE)
    v = torch.zeros(0, 2, 8, dtype=DTYPE)

    output = flash_attn_prefill(q, k, v, [0, 0], [0, 0])

    assert output.shape == (0, 4, 8)
    assert output.dtype == DTYPE


@pytest.mark.parametrize(
    "cu_seqlens_q,cu_seqlens_k",
    [
        ([1, 4], [1, 4]),
        ([0, 4, 2], [0, 4, 2]),
        ([0, 3], [0, 3]),
        ([0], [0]),
        ([0, 2, 4], [0, 4]),
        ([0, 2.0, 4], [0, 2.0, 4]),
    ],
)
def test_prefill_rejects_invalid_cu_seqlens(cu_seqlens_q, cu_seqlens_k):
    q = torch.zeros(4, 4, 8, dtype=DTYPE)
    k = torch.zeros(4, 2, 8, dtype=DTYPE)
    v = torch.zeros(4, 2, 8, dtype=DTYPE)

    with pytest.raises(ValueError):
        flash_attn_prefill(q, k, v, cu_seqlens_q, cu_seqlens_k)


def test_prefill_validates_max_seqlen_hints():
    q = torch.zeros(4, 4, 8, dtype=DTYPE)
    k = torch.zeros(4, 2, 8, dtype=DTYPE)
    v = torch.zeros(4, 2, 8, dtype=DTYPE)

    with pytest.raises(ValueError):
        flash_attn_prefill(q, k, v, [0, 4], [0, 4], max_seqlen_q=3)
    with pytest.raises(ValueError):
        flash_attn_prefill(q, k, v, [0, 4], [0, 4], max_seqlen_k=3)

    flash_attn_prefill(q, k, v, [0, 4], [0, 4], max_seqlen_q=4, max_seqlen_k=4)


def test_prefill_rejects_causal_when_keys_are_shorter_than_queries():
    q = torch.zeros(4, 4, 8, dtype=DTYPE)
    k = torch.zeros(2, 2, 8, dtype=DTYPE)
    v = torch.zeros(2, 2, 8, dtype=DTYPE)

    with pytest.raises(ValueError):
        flash_attn_prefill(q, k, v, [0, 4], [0, 2], causal=True)


@pytest.mark.parametrize(
    "q_shape,k_shape",
    [
        ((4, 4, 8), (4, 2, 4)),
        ((4, 5, 8), (4, 2, 8)),
        ((4, 8), (4, 2, 8)),
        ((4, 4, 8), (4, 2, 8, 1)),
    ],
)
def test_prefill_rejects_invalid_shapes(q_shape, k_shape):
    q = torch.zeros(q_shape, dtype=DTYPE)
    k = torch.zeros(k_shape, dtype=DTYPE)
    v = torch.zeros(k_shape, dtype=DTYPE)

    with pytest.raises((ValueError, TypeError)):
        flash_attn_prefill(q, k, v, [0, 4], [0, 4])


def test_prefill_rejects_dtype_mismatch_and_non_float_inputs():
    q = torch.zeros(4, 4, 8, dtype=DTYPE)
    k = torch.zeros(4, 2, 8, dtype=torch.float64)
    v = torch.zeros(4, 2, 8, dtype=torch.float64)

    with pytest.raises(ValueError):
        flash_attn_prefill(q, k, v, [0, 4], [0, 4])

    integer_q = torch.zeros(4, 4, 8, dtype=torch.long)
    integer_k = torch.zeros(4, 2, 8, dtype=torch.long)
    with pytest.raises(ValueError):
        flash_attn_prefill(integer_q, integer_k, integer_k, [0, 4], [0, 4])


def test_prefill_rejects_non_tensor_inputs():
    with pytest.raises(TypeError):
        flash_attn_prefill([[0.0]], torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), [0, 1], [0, 1])


def test_wrapper_prefill_matches_function():
    torch.manual_seed(7)
    q = torch.randn(5, 4, 8, dtype=DTYPE)
    k = torch.randn(5, 2, 8, dtype=DTYPE)
    v = torch.randn(5, 2, 8, dtype=DTYPE)
    cu_seqlens = [0, 5]
    wrapper = FlashAttentionWrapper(softmax_scale=0.25)

    output = wrapper.prefill(q, k, v, cu_seqlens, cu_seqlens, causal=True)
    expected = flash_attn_prefill(q, k, v, cu_seqlens, cu_seqlens, softmax_scale=0.25, causal=True)

    assert torch.allclose(output, expected, atol=ATOL)


def test_paged_decode_single_sequence_matches_reference():
    torch.manual_seed(8)
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    block_size, num_blocks, context_len = 4, 4, 7
    block_ids = [2, 0]
    scale = head_dim**-0.5

    q = torch.randn(1, num_heads, head_dim, dtype=DTYPE)
    keys = torch.randn(1, context_len, num_kv_heads, head_dim, dtype=DTYPE)
    values = torch.randn(1, context_len, num_kv_heads, head_dim, dtype=DTYPE)
    k_cache, v_cache = build_paged_cache(keys[0], values[0], block_ids, block_size, num_blocks)

    output = flash_attn_paged_decode(
        q,
        k_cache,
        v_cache,
        block_tables=torch.tensor([block_ids + [0]], dtype=torch.long),
        context_lens=torch.tensor([context_len]),
    )
    expected = reference_decode(q, keys, values, [context_len], scale)

    assert output.shape == (1, num_heads, head_dim)
    assert output.dtype == DTYPE
    assert torch.allclose(output, expected, atol=ATOL)


def test_paged_decode_multiple_sequences_matches_reference():
    torch.manual_seed(9)
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    block_size, num_blocks = 4, 6
    context_lens = [3, 7]
    block_tables = [[1, 0, 2], [3, 5, 4]]
    scale = head_dim**-0.5

    q = torch.randn(2, num_heads, head_dim, dtype=DTYPE)
    keys = torch.randn(2, 7, num_kv_heads, head_dim, dtype=DTYPE)
    values = torch.randn(2, 7, num_kv_heads, head_dim, dtype=DTYPE)

    k_cache = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim, dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)
    for index, context_len in enumerate(context_lens):
        seq_k, seq_v = build_paged_cache(
            keys[index, :context_len],
            values[index, :context_len],
            block_tables[index],
            block_size,
            num_blocks,
        )
        k_cache += seq_k
        v_cache += seq_v

    output = flash_attn_paged_decode(
        q,
        k_cache,
        v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
    )
    expected = reference_decode(q, keys, values, context_lens, scale)

    assert torch.allclose(output, expected, atol=ATOL)


def test_paged_decode_matches_existing_paged_attention():
    torch.manual_seed(10)
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    block_size, num_blocks, context_len = 4, 4, 6
    scale = head_dim**-0.5
    block_ids = [1, 3, 0]
    block_tables = torch.tensor([block_ids + [0]], dtype=torch.long)
    context_lens = torch.tensor([context_len])

    q = torch.randn(1, num_heads, head_dim, dtype=DTYPE)
    keys = torch.randn(1, context_len, num_kv_heads, head_dim, dtype=DTYPE)
    values = torch.randn(1, context_len, num_kv_heads, head_dim, dtype=DTYPE)
    k_cache, v_cache = build_paged_cache(keys[0], values[0], block_ids, block_size, num_blocks)
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()

    from inference_engine.attention.paged_attn import paged_attention

    expected = paged_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        num_seqs=1,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_context_len=16,
        softmax_scale=scale,
    )
    output = flash_attn_paged_decode(q, k_cache, v_cache, block_tables, context_lens)

    assert torch.allclose(output, expected, atol=ATOL)


def test_decode_alias_matches_paged_decode_and_ignores_causal_flag():
    torch.manual_seed(11)
    num_kv_heads, head_dim, block_size, num_blocks, context_len = 2, 8, 4, 2, 5
    block_ids = [1, 0]

    q = torch.randn(1, 4, head_dim, dtype=DTYPE)
    keys = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    values = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    k_cache, v_cache = build_paged_cache(keys, values, block_ids, block_size, num_blocks)
    block_table = torch.tensor([block_ids], dtype=torch.long)
    context_lens = torch.tensor([context_len])

    paged = flash_attn_paged_decode(q, k_cache, v_cache, block_table, context_lens)
    decoded = flash_attn_decode(q, k_cache, v_cache, context_lens, block_table)
    decoded_causal = flash_attn_decode(q, k_cache, v_cache, context_lens, block_table, causal=True)

    assert torch.allclose(decoded, paged, atol=ATOL)
    assert torch.allclose(decoded_causal, paged, atol=ATOL)


def test_paged_decode_handles_empty_context():
    torch.manual_seed(12)
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    block_size, num_blocks = 4, 2
    block_ids = [0]
    context_len = 2

    q = torch.randn(2, num_heads, head_dim, dtype=DTYPE)
    keys = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    values = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    k_cache, v_cache = build_paged_cache(keys, values, block_ids, block_size, num_blocks)

    output = flash_attn_paged_decode(
        q,
        k_cache,
        v_cache,
        block_tables=[[0], [0]],
        context_lens=[0, context_len],
    )

    assert torch.count_nonzero(output[0]) == 0
    expected = reference_decode(
        q[1:],
        keys[None, ...],
        values[None, ...],
        [context_len],
        head_dim**-0.5,
    )[0]
    assert torch.allclose(output[1], expected, atol=ATOL)


def test_paged_decode_rejects_context_longer_than_block_table():
    q = torch.zeros(1, 4, 8, dtype=DTYPE)
    k_cache = torch.zeros(2, 2, 4, 8, dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)

    with pytest.raises(ValueError):
        flash_attn_paged_decode(q, k_cache, v_cache, [[0]], [5])


def test_paged_decode_rejects_out_of_range_block_ids():
    q = torch.zeros(1, 4, 8, dtype=DTYPE)
    k_cache = torch.zeros(2, 2, 4, 8, dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)

    with pytest.raises(ValueError):
        flash_attn_paged_decode(q, k_cache, v_cache, [[5]], [1])
    with pytest.raises(ValueError):
        flash_attn_paged_decode(q, k_cache, v_cache, [[-1]], [1])


def test_paged_decode_rejects_invalid_metadata_shapes():
    q = torch.zeros(2, 4, 8, dtype=DTYPE)
    k_cache = torch.zeros(2, 2, 4, 8, dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)

    with pytest.raises(ValueError):
        flash_attn_paged_decode(q, k_cache, v_cache, [[0]], [1, 1])
    with pytest.raises(ValueError):
        flash_attn_paged_decode(q, k_cache, v_cache, [[0, 1]], [1])
    with pytest.raises(ValueError):
        flash_attn_paged_decode(q, k_cache, v_cache, [[0, 1], [0, 1]], [[1, 1]])


def test_paged_decode_rejects_invalid_q_and_cache_shapes():
    q = torch.zeros(4, 8, dtype=DTYPE)
    k_cache = torch.zeros(2, 2, 4, 8, dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)

    with pytest.raises(ValueError):
        flash_attn_paged_decode(q, k_cache, v_cache, [[0]], [1])

    with pytest.raises(ValueError):
        flash_attn_paged_decode(torch.zeros(1, 5, 8, dtype=DTYPE), k_cache, v_cache, [[0]], [1])
    with pytest.raises(ValueError):
        flash_attn_paged_decode(torch.zeros(1, 4, 16, dtype=DTYPE), k_cache, v_cache, [[0]], [1])
    with pytest.raises(ValueError):
        flash_attn_paged_decode(
            torch.zeros(1, 4, 8, dtype=torch.float64), k_cache, v_cache, [[0]], [1]
        )


def test_wrapper_decode_matches_function():
    torch.manual_seed(13)
    num_kv_heads, head_dim, block_size, num_blocks, context_len = 2, 8, 4, 2, 4
    block_ids = [1, 0]

    q = torch.randn(1, 4, head_dim, dtype=DTYPE)
    keys = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    values = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    k_cache, v_cache = build_paged_cache(keys, values, block_ids, block_size, num_blocks)
    block_table = torch.tensor([block_ids], dtype=torch.long)
    context_lens = torch.tensor([context_len])
    wrapper = FlashAttentionWrapper(softmax_scale=0.2)

    output = wrapper.decode(q, k_cache, v_cache, context_lens, block_table)
    expected = flash_attn_decode(q, k_cache, v_cache, context_lens, block_table, softmax_scale=0.2)
    paged = wrapper.paged_decode(q, k_cache, v_cache, block_table, context_lens)

    assert torch.allclose(output, expected, atol=ATOL)
    assert torch.allclose(paged, expected, atol=ATOL)


def test_paged_decode_custom_scale_changes_result():
    torch.manual_seed(14)
    num_kv_heads, head_dim, block_size, num_blocks, context_len = 2, 8, 4, 2, 4
    block_ids = [1, 0]

    q = torch.randn(1, 4, head_dim, dtype=DTYPE)
    keys = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    values = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    k_cache, v_cache = build_paged_cache(keys, values, block_ids, block_size, num_blocks)
    block_table = torch.tensor([block_ids], dtype=torch.long)
    context_lens = torch.tensor([context_len])

    default = flash_attn_paged_decode(q, k_cache, v_cache, block_table, context_lens)
    scaled = flash_attn_paged_decode(
        q, k_cache, v_cache, block_table, context_lens, softmax_scale=0.1
    )
    expected = reference_decode(q, keys[None, ...], values[None, ...], [context_len], 0.1)

    assert torch.allclose(scaled, expected, atol=ATOL)
    assert not torch.allclose(default, scaled, atol=ATOL)


def test_paged_decode_default_scale_matches_explicit_inverse_sqrt():
    torch.manual_seed(15)
    num_kv_heads, head_dim, block_size, num_blocks, context_len = 2, 16, 4, 2, 4
    block_ids = [1, 0]

    q = torch.randn(1, 4, head_dim, dtype=DTYPE)
    keys = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    values = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    k_cache, v_cache = build_paged_cache(keys, values, block_ids, block_size, num_blocks)
    block_table = torch.tensor([block_ids], dtype=torch.long)
    context_lens = torch.tensor([context_len])

    default = flash_attn_paged_decode(q, k_cache, v_cache, block_table, context_lens)
    explicit = flash_attn_paged_decode(
        q, k_cache, v_cache, block_table, context_lens, softmax_scale=16**-0.5
    )

    assert torch.allclose(default, explicit, atol=ATOL)


def test_chunked_prefill_matches_sequential_prefill():
    torch.manual_seed(16)
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    total_len = 6
    scale = head_dim**-0.5

    q = torch.randn(total_len, num_heads, head_dim, dtype=DTYPE)
    k = torch.randn(total_len, num_kv_heads, head_dim, dtype=DTYPE)
    v = torch.randn(total_len, num_kv_heads, head_dim, dtype=DTYPE)

    full = flash_attn_prefill(q, k, v, [0, total_len], [0, total_len], causal=True)
    prefix = flash_attn_prefill(q[:2], k[:2], v[:2], [0, 2], [0, 2], causal=True)
    chunk = flash_attn_prefill(q[2:], k, v, [0, 4], [0, total_len], causal=True)

    expected = reference_prefill(q, k, v, [0, total_len], [0, total_len], True, scale)

    assert torch.allclose(full, expected, atol=ATOL)
    assert torch.allclose(torch.cat([prefix, chunk], dim=0), expected, atol=ATOL)


def test_prefill_output_is_independent_of_max_seqlen_hints():
    torch.manual_seed(17)
    q = torch.randn(4, 4, 8, dtype=DTYPE)
    k = torch.randn(4, 2, 8, dtype=DTYPE)
    v = torch.randn(4, 2, 8, dtype=DTYPE)

    without_hints = flash_attn_prefill(q, k, v, [0, 4], [0, 4], causal=True)
    with_hints = flash_attn_prefill(
        q, k, v, [0, 4], [0, 4], max_seqlen_q=4, max_seqlen_k=8, causal=True
    )

    assert torch.allclose(without_hints, with_hints, atol=ATOL)


def test_paged_decode_scales_context_beyond_single_block():
    torch.manual_seed(18)
    num_kv_heads, head_dim, block_size, num_blocks = 2, 8, 4, 3
    context_len = 10
    block_ids = [2, 0, 1]

    q = torch.randn(1, 4, head_dim, dtype=DTYPE)
    keys = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    values = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    k_cache, v_cache = build_paged_cache(keys, values, block_ids, block_size, num_blocks)

    output = flash_attn_paged_decode(
        q,
        k_cache,
        v_cache,
        block_tables=[block_ids],
        context_lens=[context_len],
    )
    expected = reference_decode(
        q, keys[None, ...], values[None, ...], [context_len], head_dim**-0.5
    )

    assert torch.allclose(output, expected, atol=ATOL)


def test_paged_decode_rejects_dtype_and_device_mismatches():
    q = torch.zeros(1, 4, 8, dtype=DTYPE)
    k_cache = torch.zeros(2, 2, 4, 8, dtype=torch.float64)
    v_cache = torch.zeros_like(k_cache)

    with pytest.raises(ValueError):
        flash_attn_paged_decode(q, k_cache, v_cache, [[0]], [1])


def test_paged_decode_block_size_not_power_of_two():
    torch.manual_seed(19)
    num_kv_heads, head_dim, block_size, num_blocks = 2, 8, 3, 2
    context_len = 5
    block_ids = [1, 0]

    q = torch.randn(1, 4, head_dim, dtype=DTYPE)
    keys = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    values = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE)
    k_cache, v_cache = build_paged_cache(keys, values, block_ids, block_size, num_blocks)

    output = flash_attn_paged_decode(q, k_cache, v_cache, [block_ids], [context_len])
    expected = reference_decode(
        q, keys[None, ...], values[None, ...], [context_len], head_dim**-0.5
    )

    assert torch.allclose(output, expected, atol=ATOL)
    assert math.ceil(context_len / block_size) == 2
