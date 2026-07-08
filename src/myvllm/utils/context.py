from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence as TypingSequence

if TYPE_CHECKING:
    import torch


@dataclass
class MixedAttentionSplit:
    decode_token_indices: list[int]
    decode_seq_indices: list[int]
    prefill_token_indices: list[int]
    prefill_seq_indices: list[int]
    prefill_cu_seqlens_q: list[int]
    decode_token_indices_tensor: torch.Tensor | None = None
    decode_seq_indices_tensor: torch.Tensor | None = None
    prefill_token_indices_tensor: torch.Tensor | None = None
    prefill_seq_indices_tensor: torch.Tensor | None = None
    prefill_cu_seqlens_q_tensor: torch.Tensor | None = None


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    query_lens: torch.Tensor | None = None
    is_decode: torch.Tensor | None = None
    mixed_attention_split: MixedAttentionSplit | None = None
    # True 表示本轮 batch 中所有 seq 都是从 position 0 开始、q_len 覆盖完整 context 的 pure full prefill。
    # mixed/chunked/decode-extend batch 即使 is_prefill=True，也必须保持 False，attention 才会走 paged prefill/extend。
    is_full_prefill: bool = False
    # KIVI 量化路径需要用到的额外字段：
    # - residual_slots: (num_tokens,) 每个 token 在 fp16 residual 缓冲中的行号
    #   prefill 时 token 顺序展开；decode 时每个序列一个 slot
    # - residual_lens:  (num_seqs,) decode 阶段每条序列已经在 residual 中的 token 数 (取 paged 写入前的状态)
    # 这两个字段只在 kv_cache_dtype 以 "kivi" 开头时使用，其它路径保持 None。
    residual_slots: torch.Tensor | None = None
    residual_lens: torch.Tensor | None = None

def _to_list(values: TypingSequence[int] | object) -> list[int]:
    if hasattr(values, "tolist"):
        return list(values.tolist())
    return list(values)  # type: ignore[arg-type]


def split_mixed_decode_prefill_metadata(
    cu_seqlens_q: TypingSequence[int] | object,
    query_lens: TypingSequence[int] | object,
    is_decode: TypingSequence[bool] | object,
) -> MixedAttentionSplit:
    cu_q = _to_list(cu_seqlens_q)
    q_lens = _to_list(query_lens)
    decode_flags = [bool(value) for value in _to_list(is_decode)]

    decode_token_indices: list[int] = []
    decode_seq_indices: list[int] = []
    prefill_token_indices: list[int] = []
    prefill_seq_indices: list[int] = []
    prefill_cu_seqlens_q = [0]

    for seq_idx, (q_len, decode_flag) in enumerate(zip(q_lens, decode_flags)):
        start = cu_q[seq_idx]
        end = cu_q[seq_idx + 1]
        token_indices = list(range(start, end))
        if decode_flag:
            decode_token_indices.extend(token_indices)
            decode_seq_indices.append(seq_idx)
        else:
            prefill_token_indices.extend(token_indices)
            prefill_seq_indices.append(seq_idx)
            prefill_cu_seqlens_q.append(prefill_cu_seqlens_q[-1] + q_len)

    return MixedAttentionSplit(
        decode_token_indices=decode_token_indices,
        decode_seq_indices=decode_seq_indices,
        prefill_token_indices=prefill_token_indices,
        prefill_seq_indices=prefill_seq_indices,
        prefill_cu_seqlens_q=prefill_cu_seqlens_q,
    )


_context = Context()


def get_context() -> Context:
    return _context


def reset_context():
    global _context
    _context = Context()


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None, positions=None,
                query_lens=None, is_decode=None, mixed_attention_split=None,
                is_full_prefill=False, residual_slots=None, residual_lens=None):
    global _context
    _context = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        positions=positions,
        query_lens=query_lens,
        is_decode=is_decode,
        mixed_attention_split=mixed_attention_split,
        is_full_prefill=is_full_prefill,
        residual_slots=residual_slots,
        residual_lens=residual_lens,
    )
