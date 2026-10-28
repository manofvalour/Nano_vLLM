"""FlashAttention-style prefill and paged-decode helpers.

The kernels are implemented with :func:`torch.nn.functional.scaled_dot_product_attention`
so the module is dependency-free and correct on CPU, while PyTorch can dispatch
to fused FlashAttention kernels on CUDA. The public signatures mirror the
``flash-attn`` varlen/paged APIs used by the rest of the engine.
"""

import math
from itertools import pairwise

import torch
import torch.nn.functional as F


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:  #shh_dim
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dim() != 3:
            raise ValueError(f"{name} must have shape [tokens, heads, head_dim]")
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")

    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.shape[2] != k.shape[2]:
        raise ValueError("q and k must have the same head_dim")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must share a dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")


def _as_int_list(values, name: str) -> list[int]:
    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{name} must contain integers")
    return [int(value) for value in tensor.tolist()]


def _validate_cu_seqlens(
    cu_seqlens,
    total_tokens: int,
    name: str,
) -> list[int]:
    offsets = _as_int_list(cu_seqlens, name)
    if len(offsets) < 2:
        raise ValueError(f"{name} must contain at least two offsets")
    if offsets[0] != 0:
        raise ValueError(f"{name} must start at 0")
    if any(later < earlier for earlier, later in pairwise(offsets)):
        raise ValueError(f"{name} must be non-decreasing")
    if offsets[-1] != total_tokens:
        raise ValueError(f"{name}[-1] must equal the number of tokens")
    return offsets


def _check_max_seqlen(value: int | None, actual: int, name: str) -> None:
    if value is not None and int(value) < actual:
        raise ValueError(f"{name} must be >= the longest sequence ({actual})")


def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    if k_len < q_len:
        raise ValueError("causal prefill requires k_len >= q_len")
    query_positions = torch.arange(q_len, device=device).unsqueeze(1) + (k_len - q_len)
    key_positions = torch.arange(k_len, device=device).unsqueeze(0)
    return key_positions <= query_positions


def _expand_kv(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    if group_size == 1:
        return tensor
    return tensor.repeat_interleave(group_size, dim=1)


def flash_attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Variable-length attention over packed sequences.

    Args:
        q: ``[total_q, num_heads, head_dim]``.
        k: ``[total_k, num_kv_heads, head_dim]``.
        v: ``[total_k, num_kv_heads, head_dim]``.
        cu_seqlens_q: cumulative query lengths, shape ``[batch + 1]``.
        cu_seqlens_k: cumulative key/value lengths, shape ``[batch + 1]``.
        max_seqlen_q: optional longest query length (validated when provided).
        max_seqlen_k: optional longest key/value length (validated when provided).
        softmax_scale: defaults to ``head_dim ** -0.5``.
        causal: apply a causal mask with the query block aligned to the end of
            the keys, i.e. query ``i`` may attend keys ``<= k_len - q_len + i``.

    Returns:
        ``[total_q, num_heads, head_dim]``.
    """
    _validate_qkv(q, k, v)
    q_offsets = _validate_cu_seqlens(cu_seqlens_q, q.shape[0], "cu_seqlens_q") #check if it meet the offseting requirement
    k_offsets = _validate_cu_seqlens(cu_seqlens_k, k.shape[0], "cu_seqlens_k")
    if len(q_offsets) != len(k_offsets):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch")

    batch_size = len(q_offsets) - 1
    #computing the length of each query and key
    q_lengths = [q_offsets[i + 1] - q_offsets[i] for i in range(batch_size)]
    k_lengths = [k_offsets[i + 1] - k_offsets[i] for i in range(batch_size)]
    #max validation
    _check_max_seqlen(max_seqlen_q, max(q_lengths, default=0), "max_seqlen_q")
    _check_max_seqlen(max_seqlen_k, max(k_lengths, default=0), "max_seqlen_k")

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    group_size = q.shape[1] // k.shape[1] #groupsize for GQA/MQA/MHA
    outputs: list[torch.Tensor] = []

    for index in range(batch_size):
        q_start, q_end = q_offsets[index], q_offsets[index + 1]
        k_start, k_end = k_offsets[index], k_offsets[index + 1]
        q_len = q_end - q_start
        k_len = k_end - k_start
        if q_len == 0:
            continue

        query = q[q_start:q_end].transpose(0, 1).unsqueeze(0) #qnh -> bnqh
        key = _expand_kv(k[k_start:k_end], group_size).transpose(0, 1).unsqueeze(0) #knh -> bnkh
        value = _expand_kv(v[k_start:k_end], group_size).transpose(0, 1).unsqueeze(0) #vnh -> bnvh

        attn_mask = _causal_mask(q_len, k_len, q.device) if causal else None
        #using pytrch sdpa
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            scale=softmax_scale,
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))

    if not outputs:
        return q.new_zeros(q.shape)
    return torch.cat(outputs, dim=0)


def flash_attn_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables,
    context_lens,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Paged attention for single-token decode queries.

    Args:
        q: ``[num_seqs, num_heads, head_dim]``.
        k_cache: ``[num_blocks, num_kv_heads, block_size, head_dim]``.
        v_cache: same shape as ``k_cache``.
        block_tables: ``[num_seqs, max_blocks]`` block ids per sequence.
        context_lens: ``[num_seqs]`` number of cached tokens per sequence.
        softmax_scale: defaults to ``head_dim ** -0.5``.

    Returns:
        ``[num_seqs, num_heads, head_dim]``.
    """
    if not q.is_floating_point():
        raise ValueError("q must be a floating-point tensor")
    if q.dim() != 3:
        raise ValueError("q must have shape [num_seqs, num_heads, head_dim]")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("caches must have shape [blocks, kv_heads(headsize),block_size, head_dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if q.shape[1] % k_cache.shape[1] != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if q.shape[2] != k_cache.shape[3]:
        raise ValueError("q and caches must have the same head_dim")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q and caches must share a dtype")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("q and caches must be on the same device")

    num_seqs = q.shape[0]
    block_tables = (
        block_tables
        if isinstance(block_tables, torch.Tensor)
        else torch.as_tensor(block_tables, dtype=torch.long)
    )
    context_lens = (
        context_lens
        if isinstance(context_lens, torch.Tensor)
        else torch.as_tensor(context_lens, dtype=torch.long)
    )
    if block_tables.dim() != 2:
        raise ValueError("block_tables must be two-dimensional")
    if context_lens.dim() != 1:
        raise ValueError("context_lens must be one-dimensional")
    if block_tables.shape[0] != num_seqs or context_lens.shape[0] != num_seqs:
        raise ValueError("block_tables and context_lens must match the number of queries")
    if block_tables.is_floating_point() or context_lens.is_floating_point():
        raise ValueError("block_tables and context_lens must contain integers")

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    num_blocks, num_kv_heads, block_size = k_cache.shape[:3]
    group_size = q.shape[1] // num_kv_heads
    output = q.new_zeros(q.shape)

    for index in range(num_seqs):
        context_len = int(context_lens[index].item())
        if context_len < 0:
            raise ValueError("context lengths must be non-negative")
        if context_len == 0:
            continue

        needed_blocks = math.ceil(context_len / block_size)
        if needed_blocks > block_tables.shape[1]:
            raise ValueError("context length exceeds the block table capacity")

        block_ids = [int(block_id) for block_id in block_tables[index, :needed_blocks].tolist()]
        for block_id in block_ids:
            if block_id < 0 or block_id >= num_blocks:
                raise ValueError(f"block id {block_id} is out of range")

        keys = torch.cat([k_cache[block_id] for block_id in block_ids], dim=1)
        values = torch.cat([v_cache[block_id] for block_id in block_ids], dim=1)
        keys = keys[:, :context_len, :].repeat_interleave(group_size, dim=0)
        values = values[:, :context_len, :].repeat_interleave(group_size, dim=0)

        query = q[index].unsqueeze(0).unsqueeze(2)
        attended = F.scaled_dot_product_attention(
            query,
            keys.unsqueeze(0),
            values.unsqueeze(0),
            scale=softmax_scale,
        )
        output[index] = attended[0, :, 0, :]

    return output


def flash_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens,
    block_table,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Decode helper matching the ``flash-attn`` paged-cache signature.

    Each query token is the next position in its sequence, so it can attend to
    the full cached context; ``causal`` is accepted for API parity and does not
    mask any keys.
    """
    del causal
    return flash_attn_paged_decode(
        q,
        k_cache,
        v_cache,
        block_tables=block_table,
        context_lens=cache_seqlens,
        softmax_scale=softmax_scale,
    )


class FlashAttentionWrapper:
    """Convenience wrapper that applies a shared ``softmax_scale``."""

    def __init__(self, softmax_scale: float | None = None) -> None:
        self.softmax_scale = softmax_scale

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        return flash_attn_prefill(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )

    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens,
        block_table,
        causal: bool = False,
    ) -> torch.Tensor:
        return flash_attn_decode(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            block_table,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )

    def paged_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables,
        context_lens,
    ) -> torch.Tensor:
        return flash_attn_paged_decode(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            softmax_scale=self.softmax_scale,
        )


__all__ = [
    "FlashAttentionWrapper",
    "flash_attn_decode",
    "flash_attn_paged_decode",
    "flash_attn_prefill",
]
