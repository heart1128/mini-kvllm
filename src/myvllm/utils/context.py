from dataclasses import dataclass 
import torch 


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
    # KIVI 量化路径需要用到的额外字段：
    # - residual_slots: (num_tokens,) 每个 token 在 fp16 residual 缓冲中的行号
    #   prefill 时 token 顺序展开；decode 时每个序列一个 slot
    # - residual_lens:  (num_seqs,) decode 阶段每条序列已经在 residual 中的 token 数 (取 paged 写入前的状态)
    # 这两个字段只在 kv_cache_dtype 以 "kivi" 开头时使用，其它路径保持 None。
    residual_slots: torch.Tensor | None = None
    residual_lens: torch.Tensor | None = None

_context = Context()

def get_context() -> Context:
    return _context

def reset_context():
    global _context
    _context = Context()

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None, positions=None,
                residual_slots=None, residual_lens=None):
    global _context
    _context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables, positions,
                       residual_slots, residual_lens)
