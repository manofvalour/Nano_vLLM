
import torch
import torch.nn.functional as F
#from attention.naive_attn import MHA

# PagedAttention kernel implementation using Triton
# Based on vLLM's paged attention implementation

BLOCK_SIZE = 16  # tokens per block


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
    """
    Store key-value pairs into paged cache using block_based storage.

    Args:
        key: Key tensor of shape [num_tokens, num_heads, head_dim]
        value: Value tensor of shape [num_tokens, num_heads, head_dim]

        key_cache: Key cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Value cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim]

        slot_mapping: Slot mapping tensor of shape [num_tokens]
        num_tokens: Number of tokens to store
        num_heads: Number of attention heads
        head_dim: Head dimension
        block_size: Size of each block in tokens

    Returns:
        None
    """
    # Triton kernel implementation would go here
    # For now, implement a reference implementation
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_id = slot // block_size
        block_offset = slot % block_size

        # Store key
        key_cache[block_id, :, block_offset, :] = key[i]  # store key
        value_cache[block_id, :, block_offset, :] = value[i]  # Store value


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
    softmax_scale: float|None = None,
) -> torch.Tensor:
    """
    Paged attention computation with block table lookups.

    Args:
        query: Query tensor of shape [num_seqs, num_heads, head_dim]
        key_cache: Key cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Value cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim]
        block_tables: Block tables tensor of shape [num_seqs, max_blocks_per_seq]
        context_lens: Context lengths tensor of shape [num_seqs]
        num_seqs: Number of sequences
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        head_dim: Head dimension
        block_size: Size of each block in tokens
        max_context_len: Maximum context length
        softmax_scale: Scale factor for softmax

    Returns:
        Output tensor of shape [num_seqs, num_heads, head_dim]
    """
    output = torch.zeros(num_seqs, num_heads, head_dim)  # num_seq, num_head, head_dim
    group_size = num_heads//num_kv_heads
    assert num_heads % num_kv_heads == 0

    if softmax_scale is None:
        softmax_scale = head_dim * -0.5

    for seq_id in range(num_seqs):
        seq_len = int(context_lens[seq_id].item())

        if seq_len == 0:
            continue

        if seq_len >max_context_len:
            raise ValueError("Context Exceed Max Context Len")

        # Get block table for this sequence
        seq_block_table = block_tables[seq_id]
        keys = []
        values = []

        # Compute attention for each position in the sequence
        for pos in range(seq_len):
            # Find which block contains this position
            block_id = seq_block_table[pos // block_size].item()
            block_offset = pos % block_size

            # Get key and value from cache
            keys.append(key_cache[block_id, :, block_offset, :])
            values.append(value_cache[block_id, :, block_offset, :])

        K = torch.stack(keys, dim=0)
        V = torch.stack(values, dim=0)

        K = K.repeat_interleave(group_size,dim=1)
        V = V.repeat_interleave(group_size, dim=1)

        Q = query[seq_id]
        # Compute attention scores
        attn_scores = torch.einsum("hd,thd->ht", Q,K) * softmax_scale
        #attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute output
        output[seq_id] = torch.einsum("ht, thd->hd", attn_weights,V)
        #output[seq_id] = torch.matmul(attn_weights, V)

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
    """
    Reshape key-value pairs and store them in paged cache.

    Args:
        key: Key tensor of shape [num_tokens, num_kv_heads, head_dim]
        value: Value tensor of shape [num_tokens, num_kv_heads, head_dim]
        key_cache: Key cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Value cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim]
        slot_mapping: Slot mapping tensor of shape [num_tokens]
        num_tokens: Number of tokens to store
        num_kv_heads: Number of key-value heads
        head_dim: Head dimension
        block_size: Size of each block in tokens
    """
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
    """
    Compute slot mapping from block tables and context lengths.

    Args:
        block_tables: Block tables tensor of shape [num_seqs, max_blocks_per_seq]
        context_lens: Context lengths tensor of shape [num_seqs]
        num_seqs: Number of sequences
        block_size: Size of each block in tokens

    Returns:
        Slot mapping tensor of shape [total_tokens]
    """
    total_tokens = int(context_lens.sum().item())
    slot_mapping = torch.zeros(total_tokens, dtype=torch.int32)

    current_pos = 0
    for seq_id in range(num_seqs):
        seq_len = context_lens[seq_id].item()
        seq_block_table = block_tables[seq_id]

        for pos in range(seq_len):
            block_id = seq_block_table[pos // block_size].item()
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
    """
    Get block IDs for each sequence position.

    Args:
        block_tables: Block tables tensor of shape [num_seqs, max_blocks_per_seq]
        context_lens: Context lengths tensor of shape [num_seqs]
        num_seqs: Number of sequences
        block_size: Size of each block in tokens

    Returns:
        Block IDs tensor of shape [total_tokens]
    """
    total_tokens = int(context_lens.sum().item())
    block_ids = torch.zeros(total_tokens, dtype=torch.int32)

    current_pos = 0
    for seq_id in range(num_seqs):
        seq_len = context_lens[seq_id].item()
        seq_block_table = block_tables[seq_id]

        for pos in range(seq_len):
            block_ids[current_pos] = seq_block_table[pos // block_size].item()
            current_pos += 1

    return block_ids
