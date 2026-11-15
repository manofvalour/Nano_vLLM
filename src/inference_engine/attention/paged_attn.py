import torch
import torch.nn.functional as F

BLOCK_SIZE = 16


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    block_size: int = BLOCK_SIZE,
) -> None:
    """Store keys/values into a paged KV cache.

    The cache layout is `[num_blocks, num_kv_heads, block_size, head_dim]` and each
    `slot_mapping[i]` gives the physical location of token `i` in the cache.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0")
    if key.shape[0] != num_tokens or value.shape[0] != num_tokens:
        raise ValueError("key/value token count must match num_tokens")
    if key.shape[1] != num_heads or value.shape[1] != num_heads:
        raise ValueError("key/value head count must match num_heads")
    if key.shape[2] != head_dim or value.shape[2] != head_dim:
        raise ValueError("head_dim mismatch")

    for i in range(num_tokens):
        slot = int(slot_mapping[i].item())
        block_id = slot // block_size
        block_offset = slot % block_size

        if block_id < 0 or block_id >= key_cache.shape[0]:
            raise ValueError(f"slot {slot} maps outside the key cache")
        if block_offset < 0 or block_offset >= block_size:
            raise ValueError(f"slot {slot} has invalid offset in block")

        key_cache[block_id, :, block_offset, :] = key[i]
        value_cache[block_id, :, block_offset, :] = value[i]


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int = BLOCK_SIZE,
    max_context_len: int = 4096,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Reference paged attention implementation.

    The implementation remains intentionally straightforward to preserve the current
    repo API and tests while matching the design's block-table lookup semantics.
    """
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    output = torch.zeros(num_seqs, num_heads, head_dim, device=query.device, dtype=query.dtype)
    group_size = num_heads // num_kv_heads

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)

    for seq_id in range(num_seqs):
        seq_len = int(context_lens[seq_id].item())
        if seq_len == 0:
            continue
        if seq_len > max_context_len:
            raise ValueError("Context exceed max_context_len")

        seq_block_table = block_tables[seq_id]
        keys = []
        values = []

        for pos in range(seq_len):
            block_index = pos // block_size
            block_id = int(seq_block_table[block_index].item())
            block_offset = pos % block_size

            keys.append(key_cache[block_id, :, block_offset, :])
            values.append(value_cache[block_id, :, block_offset, :])

        K = torch.stack(keys, dim=0)
        V = torch.stack(values, dim=0)
        K = K.repeat_interleave(group_size, dim=1)
        V = V.repeat_interleave(group_size, dim=1)

        Q = query[seq_id]
        attn_scores = torch.einsum("hd,thd->ht", Q, K) * softmax_scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        output[seq_id] = torch.einsum("ht,thd->hd", attn_weights, V)

    return output


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int = BLOCK_SIZE,
) -> None:
    """Compatibility wrapper that reshapes and stores keys/values into the paged cache."""
    store_kvcache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_tokens,
        num_kv_heads,
        head_dim,
        block_size,
    )


def compute_slot_mapping(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_seqs: int,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Compute the flattened slot mapping for a paged KV cache."""
    total_tokens = int(context_lens.sum().item())
    slot_mapping = torch.zeros(total_tokens, dtype=torch.int32, device=block_tables.device)

    current_pos = 0
    for seq_id in range(num_seqs):
        seq_len = int(context_lens[seq_id].item())
        seq_block_table = block_tables[seq_id]

        for pos in range(seq_len):
            block_id = int(seq_block_table[pos // block_size].item())
            block_offset = pos % block_size
            slot_mapping[current_pos] = block_id * block_size + block_offset
            current_pos += 1

    return slot_mapping


def get_block_ids(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_seqs: int,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Get the physical block id for each token position."""
    total_tokens = int(context_lens.sum().item())
    block_ids = torch.zeros(total_tokens, dtype=torch.int32, device=block_tables.device)

    current_pos = 0
    for seq_id in range(num_seqs):
        seq_len = int(context_lens[seq_id].item())
        seq_block_table = block_tables[seq_id]

        for pos in range(seq_len):
            block_ids[current_pos] = int(seq_block_table[pos // block_size].item())
            current_pos += 1

    return block_ids
