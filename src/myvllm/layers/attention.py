import triton 
import triton.language as tl
from myvllm.utils import get_context
import torch
import torch.nn as nn

@triton.jit
def store_kvcache_kernel(
    key_ptr, # pointer to what we want to store
    value_ptr,
    k_cache_ptr, # pointer to where we want to store
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr
):
    """
    Store keys and values into paged KV cache.
    Each token is mapped to a slot via slot_mapping.
    Grid layout: (num_tokens, num_kv_heads)
    Cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    """
    # thread ID, in dimension 0
    token_idx = tl.program_id(0) # each GPU thread processes one token
    # slot ID, where in cache to store this token
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    if slot_idx == -1:
        return
    
    # Calculate which block and position within block
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    
    # Process each head
    # program_id(0) = which token
    # program_id(1) = which head
    head_idx = tl.program_id(1)
    
    # it creates a vector [0, 1, ..., head_dim-1]
    # Load key and value for this token and head
    head_offsets = tl.arange(0, head_dim)
    # Input: (num_tokens, num_kv_heads, head_dim)
    # example: input_offset = 5 * (8 * 128) + 3 * 128 + [0, 1, 2, ..., 127]
    #         = 5120 + 384 + [0, 1, 2, ..., 127]
    #         = [5504, 5505, 5506, ..., 5631]
    input_offset = (token_idx * num_kv_heads * head_dim + # skip previous tokens
                    head_idx * head_dim + # skip previous heads
                    head_offsets)

    # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim + # skip previous blocks
                   block_offset * num_kv_heads * head_dim + # skip previous positions in block
                   head_idx * head_dim + # skip previous heads
                   head_offsets) 
    
    # load key and value value floats from··· the pointers's memory
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)
    
    # store into cache
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor,
    block_size: int
):
    """
    Store key-value pairs into paged cache.
    
    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - maps each token to a cache slot
        block_size: number of tokens per block
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    
    # Make contiguous if needed
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"
    
    grid = (num_tokens, num_kv_heads)
    # launch num_tokens x num_kv_heads threads
    store_kvcache_kernel[grid](
        key, # tensors are automatically converted to pointers by triton
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size
    )


# ===================== FP8 KV Cache 量化相关 =====================
# FP8 e4m3fn 的最大可表示绝对值，用于计算量化 scale (amax / FP8_E4M3_MAX)
FP8_E4M3_MAX = 448.0


@triton.jit
def store_kvcache_fp8_kernel(
    key_ptr,            # 待写入的 key 指针, 形状 (num_tokens, num_kv_heads, head_dim)
    value_ptr,          # 待写入的 value 指针, 同上
    k_cache_ptr,        # FP8 K cache 指针, 形状 (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache_ptr,        # FP8 V cache 指针, 同上
    k_scale_ptr,        # K 的 scale 张量指针
    v_scale_ptr,        # V 的 scale 张量指针
    slot_mapping_ptr,   # token -> cache slot 的映射
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    PER_TOKEN_HEAD: tl.constexpr,  # 1=per_token_head 动态计算 scale; 0=per_tensor 使用传入的标量 scale
    FP8_MAX: tl.constexpr,         # FP8 数值上限 (448.0)
):
    """
    将 key/value 量化为 FP8 并写入 paged KV cache。
    Grid 布局: (num_tokens, num_kv_heads)，每个 program 处理一个 token 的一个 KV head。

    量化数学:
        per_token_head: scale = max(|x|) / FP8_MAX  (对该 token+head 的 head_dim 个元素动态求)
        per_tensor:     scale 由外部预先算好, 直接读取标量
        量化:   x_fp8 = clamp(x / scale, -FP8_MAX, FP8_MAX) 后转为 float8
    """
    token_idx = tl.program_id(0)   # 当前处理的 token
    head_idx = tl.program_id(1)    # 当前处理的 KV head

    # 取该 token 对应的 cache slot；-1 表示无需写入 (例如 prefix cache 命中部分)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return

    # 由 slot 计算所在 block 与 block 内偏移
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    head_offsets = tl.arange(0, head_dim)
    # 输入张量 (num_tokens, num_kv_heads, head_dim) 的偏移
    input_offset = (token_idx * num_kv_heads * head_dim +
                    head_idx * head_dim +
                    head_offsets)
    # cache 张量 (num_blocks, block_size, num_kv_heads, head_dim) 的偏移
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim +
                    block_offset * num_kv_heads * head_dim +
                    head_idx * head_dim +
                    head_offsets)

    # 以 float32 载入原始 key/value，保证量化计算精度
    key = tl.load(key_ptr + input_offset).to(tl.float32)
    value = tl.load(value_ptr + input_offset).to(tl.float32)

    if PER_TOKEN_HEAD:
        # ---- per_token_head: 针对当前 (token, head) 的 head_dim 个元素分别求 amax ----
        k_amax = tl.max(tl.abs(key))
        v_amax = tl.max(tl.abs(value))
        # 防止除零: 加一个极小值下限
        k_scale = tl.where(k_amax > 0, k_amax / FP8_MAX, 1.0)
        v_scale = tl.where(v_amax > 0, v_amax / FP8_MAX, 1.0)
        # scale 张量形状 (num_blocks, block_size, num_kv_heads)
        scale_offset = (block_idx * block_size * num_kv_heads +
                        block_offset * num_kv_heads +
                        head_idx)
        tl.store(k_scale_ptr + scale_offset, k_scale)
        tl.store(v_scale_ptr + scale_offset, v_scale)
    else:
        # ---- per_tensor: scale 为预先算好的全局标量 (k_scale_ptr/v_scale_ptr 指向单元素) ----
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)

    # 量化: 除以 scale 后裁剪到 FP8 范围，再写入 (store 时自动转为 float8 cache dtype)
    k_q = tl.minimum(tl.maximum(key / k_scale, -FP8_MAX), FP8_MAX)
    v_q = tl.minimum(tl.maximum(value / v_scale, -FP8_MAX), FP8_MAX)
    tl.store(k_cache_ptr + cache_offset, k_q.to(k_cache_ptr.dtype.element_ty))
    tl.store(v_cache_ptr + cache_offset, v_q.to(v_cache_ptr.dtype.element_ty))


def store_kvcache_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    kv_cache_dtype: str,
):
    """
    将 key/value 量化为 FP8 写入 paged cache 的 Python 封装。

    Args:
        key/value:     (num_tokens, num_kv_heads, head_dim) 原始精度 (fp16/bf16)
        k_cache/v_cache: (num_blocks, block_size, num_kv_heads, head_dim) FP8 cache
        k_scale/v_scale: scale 张量 (per_tensor: 形状(1,); per_token_head: (num_blocks,block_size,num_kv_heads))
        slot_mapping:  (num_tokens,) token->slot 映射
        kv_cache_dtype: "fp8_per_tensor" 或 "fp8_per_token_head"
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    key = key.contiguous()
    value = value.contiguous()

    per_token_head = 1 if kv_cache_dtype == 'fp8_per_token_head' else 0

    # per_tensor 模式: scale 必须在整个生命周期内【固定】。
    # 原因: cache 里历史 token 是用「第一次标定的 scale」量化写入的，
    #       decode 反量化时也必须用【同一个】scale，否则历史值会被错误地按
    #       新 scale 还原 -> 数值全错 -> 输出乱码。
    # 因此只在首次(scale 仍为初始值 1.0)用当前批 amax 标定一次，之后不再改动。

    # 这里就是直接算出scale，不用在kernel中计算，因为是per-tensor的，
    if per_token_head == 0:
        # 约定: scale 初始值为 1.0 (allocate_kv_cache 用 torch.ones 初始化)，
        #       一旦被标定过(!=1.0)就跳过，保持固定。
        if float(k_scale.reshape(-1)[0].item()) == 1.0:
            k_amax = key.abs().amax().clamp(min=1e-8)
            k_scale.fill_((k_amax / FP8_E4M3_MAX).item())
        if float(v_scale.reshape(-1)[0].item()) == 1.0:
            v_amax = value.abs().amax().clamp(min=1e-8)
            v_scale.fill_((v_amax / FP8_E4M3_MAX).item())

    grid = (num_tokens, num_kv_heads)
    store_kvcache_fp8_kernel[grid](
        key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        PER_TOKEN_HEAD=per_token_head,
        FP8_MAX=FP8_E4M3_MAX,
    )


@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel for variable-length sequences.
    Each program processes one block of queries for one head in one sequence.
    """
    # Program IDs
    start_m = tl.program_id(0) # block index
    off_h = tl.program_id(1) # head index
    seq_idx = tl.program_id(2) # sequence index

    # Determine which KV head to use (for GQA)
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # Load sequence boundaries
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    # Early exit if this block is beyond sequence length
    if start_m * BLOCK_M >= seq_len:
        return
    
    # Offset for this block of queries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    
    # Load Q block - shape (BLOCK_M, head_dim)
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize output accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    # Number of blocks to process
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    # Loop over K, V blocks
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Mask for valid positions
        mask_n = offs_n < seq_len
        
        # K pointers: K has shape (total_tokens, num_kv_heads, head_dim)
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        
        # Load K block - shape (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Compute QK^T - shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale
        
        # Apply causal mask: only attend to positions <= current position
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Rescale previous accumulator
        acc = acc * alpha[:, None]
        
        # Load V block - shape (BLOCK_N, head_dim)
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Accumulate weighted values
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output: O has shape (total_tokens, num_heads, head_dim)
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Optimized Flash Attention for prefill phase with variable-length sequences.
    
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: cumulative sequence lengths
        scale: attention scale factor
    
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Conservative block sizes to avoid OOM on shared memory
    # Shared memory usage ~ BLOCK_M * BLOCK_N * 4 bytes (for float32 attention scores)
    # + BLOCK_M * head_dim * 4 (for Q)
    # + BLOCK_N * head_dim * 4 (for K, V)
    # Want to keep total < 48KB for most GPUs
    
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16
    
    # Number of sequences
    num_seqs = cu_seqlens.shape[0] - 1
    
    # Find max sequence length to determine grid size
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # Calculate grid dimensions - launch all kernels at once
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)
    
    flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output


@triton.jit
def paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized paged attention kernel for decode phase.
    Processes KV cache in chunks.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Determine which KV head this query head uses (for GQA)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Load context length
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    # Load query: (batch_size, num_heads, head_dim)
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    
    # Initialize accumulators
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10
    
    # Calculate total number of chunks to process
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    
    # Process all tokens in chunks
    for chunk_idx in range(max_chunks):
        # Global token index for this chunk
        token_start = chunk_idx * BLOCK_N
        
        # Only process if within valid range
        if token_start < context_len:
            # Determine which tokens in this chunk are valid
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len
            
          
            # Compute attention scores for this chunk
            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10
            
            # Load K for each valid position and compute scores
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    
                    if block_num < max_num_blocks:
                        # Look up physical block
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        
                        if physical_block_idx != -1:
                            # Load K
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            k_vec = tl.load(k_cache_ptr + k_offset)
                            
                            # Compute score for this token
                            score = tl.sum(q * k_vec) * scale
                            
                            # Update qk array at position i using tl.where
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)
            
            # Apply mask to invalid positions
            qk = tl.where(mask_n, qk, -1e10)
            
            # Online softmax
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            
            # Rescale accumulator
            acc = acc * alpha
            l_i = l_i * alpha
            
            # Load V and accumulate
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    
                    if block_num < max_num_blocks:
                        # Look up physical block
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        
                        if physical_block_idx != -1:
                            # Load V
                            v_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            v_vec = tl.load(v_cache_ptr + v_offset)
                            
                            # Extract weight for this token from p
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            
                            acc = acc + weight * v_vec
                            l_i = l_i + weight
            
            m_i = m_i_new
    
    # Normalize
    output = acc / l_i
    
    # Store output
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_prefill_torch(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    block_size: int,
) -> torch.Tensor:
    outputs = torch.empty_like(query)
    q_ranges = cu_seqlens_q.cpu().tolist()
    group_size = num_heads // num_kv_heads

    for seq_idx in range(len(q_ranges) - 1):
        q_start, q_end = q_ranges[seq_idx], q_ranges[seq_idx + 1]
        context_len = int(context_lens[seq_idx].item())
        seq_blocks = block_tables[seq_idx]
        history_k = []
        history_v = []
        for token_idx in range(context_len):
            block_idx = token_idx // block_size
            block_offset = token_idx % block_size
            physical_block = int(seq_blocks[block_idx].item())
            history_k.append(k_cache[physical_block, block_offset])
            history_v.append(v_cache[physical_block, block_offset])
        history_k = torch.stack(history_k, dim=0)
        history_v = torch.stack(history_v, dim=0)

        for q_idx in range(q_start, q_end):
            token_pos = int(positions[q_idx].item())
            visible_k = history_k[:token_pos + 1]
            visible_v = history_v[:token_pos + 1]
            for head_idx in range(num_heads):
                kv_head_idx = head_idx // group_size
                scores = torch.sum(
                    query[q_idx, head_idx].unsqueeze(0) * visible_k[:, kv_head_idx],
                    dim=-1,
                ) * scale
                probs = torch.softmax(scores, dim=-1)
                outputs[q_idx, head_idx] = torch.sum(
                    probs.unsqueeze(-1) * visible_v[:, kv_head_idx],
                    dim=0,
                )
    return outputs



def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    """
    Compute attention in decode mode using paged KV cache.
    
    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        context_lens: (batch_size,)
        scale: attention scale factor
    
    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    
    # Make contiguous
    query = query.contiguous()
    
    output = torch.empty_like(query)
    
    # Chunk size for processing KV tokens
    BLOCK_N = 64 if head_dim <= 128 else 32
    
    grid = (batch_size, num_heads)
    
    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )
    
    return output


@triton.jit
def paged_attention_decode_fp8_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,        # FP8 K cache，布局为 (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache_ptr,        # FP8 V cache，布局同 K cache
    k_scale_ptr,        # K 反量化 scale；per_tensor 为单标量，per_token_head 为逐 token/head scale
    v_scale_ptr,        # V 反量化 scale；K/V 独立存储，不能共用
    block_tables_ptr,   # 逻辑 block 到物理 block 的映射，布局为 (batch_size, max_num_blocks)
    context_lens_ptr,   # 每条请求当前可见的历史 token 数
    scale: tl.constexpr,            # attention 缩放因子，通常是 1 / sqrt(head_dim)
    num_heads: tl.constexpr,        # Q head 数
    num_kv_heads: tl.constexpr,     # KV head 数；GQA/MQA 下会小于 num_heads
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,   # 单条请求 block table 最多能索引的逻辑 block 数
    BLOCK_N: tl.constexpr,          # 每次循环处理的历史 token 数，用于分块 online softmax
    PER_TOKEN_HEAD: tl.constexpr,   # 1=per_token_head; 0=per_tensor
):
    """
    decode 阶段的 paged attention，读取 FP8 KV cache，反量化后完成 attention。

    这个 kernel 的并行粒度是一个 (batch_idx, head_idx)：
      - 一个 Triton program 只负责一条请求的一个 Q head；
      - 该 program 会遍历这条请求所有历史 token；
      - 历史 K/V 不连续存放，而是通过 block table 从逻辑 block 找到物理 block。

    与普通 paged_attention_decode_kernel 的主要区别：
      - K/V cache 中存的是 FP8 数值，载入后必须乘以 scale 才能近似还原；
      - per_tensor 模式下，整层 K 和整层 V 各自只有一个 scale，可提前加载并复用；
      - per_token_head 模式下，每个 (physical_block, block_offset, kv_head) 都有自己的 scale，
        所以每读取一个 token/head 的 K 或 V，都要按 cache 位置读取对应 scale。

    数值计算使用 online softmax：按 BLOCK_N 分块扫描历史 token，维护全局最大值 m_i、
    softmax 分母 l_i 和加权输出 acc，避免一次性保存完整 qk 矩阵。
    Grid 布局: (batch_size, num_heads)
    """
    # 当前 Triton program 负责第 batch_idx 条请求、第 head_idx 个 Q head。
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # GQA/MQA 映射：多个 Q head 共享一个 KV head。
    # 例如 num_heads=32, num_kv_heads=8 时，每 4 个 Q head 对应同一个 KV head。
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # 当前请求真实的历史长度；block table 可能有 padding，但 attention 只能看到 context_len 内 token。
    context_len = tl.load(context_lens_ptr + batch_idx)

    # 取当前请求、当前 Q head 的 q 向量，offs_d 覆盖 head_dim 维度。
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    # per_tensor 模式下 scale 是单个标量，提前载入可以避免每个 token 重复读取。
    # per_token_head 模式不能提前载入，因为不同 token/head 位置的 scale 不同。
    if PER_TOKEN_HEAD == 0:
        k_scale_scalar = tl.load(k_scale_ptr)
        v_scale_scalar = tl.load(v_scale_ptr)

    # online softmax 的三个状态：
    # acc: 当前已经处理过的 token 的 sum(exp(score) * V)，按 head_dim 维度累加。
    # l_i: 当前 softmax 分母 sum(exp(score))。
    # m_i: 当前已经处理过的 token 的最大 score，用于数值稳定。
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    # 理论最多扫描 max_num_blocks * block_size 个 token；真实长度由 context_len 截断。
    # 用 BLOCK_N 分块，是为了每块内部先形成一小段 qk，再用 online softmax 合并到全局状态。
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N
        if token_start < context_len:
            # 当前 chunk 覆盖的逻辑 token 下标范围 [token_start, token_start + BLOCK_N)。
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len

            # qk 保存当前 chunk 内 BLOCK_N 个 token 的 attention score。
            # 先填成一个很小的值，后面对无效 token 保持 -inf，softmax 后权重约为 0。
            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    # 将逻辑 token 下标映射成逻辑 block 号和 block 内偏移。
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    if block_num < max_num_blocks:
                        # block_tables[batch_idx, block_num] 给出这个逻辑 block 实际存在哪个物理 block。
                        # physical_block_idx == -1 表示该位置无效，不能访问 cache。
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        if physical_block_idx != -1:
                            # 从 FP8 K cache 中读取当前历史 token、当前 KV head 的整条 K 向量。
                            # cache 物理布局: (physical_block, block_offset, kv_head, head_dim)。
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            k_vec = tl.load(k_cache_ptr + k_offset).to(tl.float32)

                            # 反量化 K: 原值近似为 fp8_value * scale。
                            # per_token_head 的 scale 和 cache token 对齐，索引布局为
                            # (physical_block, block_offset, kv_head)；per_tensor 直接复用标量。
                            if PER_TOKEN_HEAD:
                                ks_off = (physical_block_idx * block_size * num_kv_heads +
                                          block_offset * num_kv_heads + kv_head_idx)
                                k_s = tl.load(k_scale_ptr + ks_off)
                            else:
                                k_s = k_scale_scalar
                            k_vec = k_vec * k_s

                            # 计算当前 token 的 q·k，并乘 attention scale。
                            # qk 是长度 BLOCK_N 的向量，这里用 mask_i 把标量 score 写回第 i 个位置。
                            score = tl.sum(q * k_vec) * scale
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)

            # 对超出 context_len 的位置强制设为 -inf，确保 softmax 权重为 0。
            qk = tl.where(mask_n, qk, -1e10)

            # online softmax 更新。
            # m_ij 是当前 chunk 的最大 score；m_i_new 是截至当前 chunk 的全局最大 score。
            # 当全局最大值变大时，需要用 alpha 把历史 acc/l_i 重新缩放到新基准下。
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            acc = acc * alpha
            l_i = l_i * alpha

            # 第二遍扫描当前 chunk：读取 V，反量化，并按照刚算出的 softmax 分子 p 加权累加。
            # 这里 p 还没有除以最终分母 l_i，最终在函数末尾统一做 acc / l_i。
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        if physical_block_idx != -1:
                            # 从 FP8 V cache 中读取当前历史 token、当前 KV head 的整条 V 向量。
                            v_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            v_vec = tl.load(v_cache_ptr + v_offset).to(tl.float32)

                            # 反量化 V。注意 V 使用 v_scale_ptr，和 K 的 scale 独立，不能混用。
                            if PER_TOKEN_HEAD:
                                vs_off = (physical_block_idx * block_size * num_kv_heads +
                                          block_offset * num_kv_heads + kv_head_idx)
                                v_s = tl.load(v_scale_ptr + vs_off)
                            else:
                                v_s = v_scale_scalar
                            v_vec = v_vec * v_s

                            # 取出当前 token 在本 chunk 内的 softmax 分子 p[i]，累加到 acc 和 l_i。
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            acc = acc + weight * v_vec
                            l_i = l_i + weight

            # 更新全局最大 score，供下一个 chunk 做数值稳定的 online softmax 合并。
            m_i = m_i_new

    # 完成 softmax 归一化，并写回当前 (batch_idx, head_idx) 的输出向量。
    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output.to(output_ptr.dtype.element_ty))


def paged_attention_decode_fp8(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    kv_cache_dtype: str,
) -> torch.Tensor:
    """
    decode 阶段基于 FP8 paged cache 计算注意力 (含反量化) 的 Python 封装。
    输出 dtype 与 query 一致 (fp16/bf16)。
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    query = query.contiguous()
    output = torch.empty_like(query)

    BLOCK_N = 64 if head_dim <= 128 else 32
    per_token_head = 1 if kv_cache_dtype == 'fp8_per_token_head' else 0
    grid = (batch_size, num_heads)

    paged_attention_decode_fp8_kernel[grid](
        output, query, k_cache, v_cache, k_scale, v_scale,
        block_tables, context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
        PER_TOKEN_HEAD=per_token_head,
    )
    return output


# ===================== KIVI KV Cache 量化相关 =====================
# KIVI 设计要点 (paged 适配版):
#   - K: 块内 per-channel-per-group 非对称量化
#        scale = (max - min) / (2^bits - 1)
#        zero  = min
#        q     = clamp(round((x - zero) / scale), 0, 2^bits - 1)
#        反量化: x = q * scale + zero
#   - V: per-token-per-head 对称量化 (与 fp8_per_token_head 同思路)
#        scale = amax / qmax,  q = clamp(round(x/scale), -qmax, qmax)
#        反量化: x = q * scale
#   - 实现简化: 量化值不做 bit-pack, 直接放入 int8 容器 (每元素占 1 byte)。
#     这样 paged 访存与 fp16 / fp8 路径完全同构, kernel 索引计算 0 修改成本。
#     如需进一步省显存, 可在 store/decode kernel 里加 nibble pack。

# K: 非对称量化的 qmax (无符号)
@triton.jit
def _kivi_qmax_unsigned(bits):
    return (1 << bits) - 1


@triton.jit
def store_kvcache_kivi_kernel(
    key_ptr,            # 待写入的 key (num_tokens, num_kv_heads, head_dim)
    value_ptr,          # 待写入的 value
    k_cache_ptr,        # int8 K cache  (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache_ptr,        # int8 V cache  同上
    k_scale_ptr,        # K scale  (num_blocks, num_kv_heads, n_groups), fp32
    k_zero_ptr,         # K zero   同上, fp32
    v_scale_ptr,        # V scale  (num_blocks, block_size, num_kv_heads), fp32
    k_residual_ptr,     # K fp16 residual (num_seqs, residual_length, num_kv_heads, head_dim)
    v_residual_ptr,     # V fp16 residual  同上
    slot_mapping_ptr,   # token -> cache slot
    residual_slots_ptr, # token -> residual buffer 的行号 (seq_slot)
    residual_lens_ptr,  # (num_seqs,) 暂未使用; 留作扩展
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    group_size: tl.constexpr,
    residual_length: tl.constexpr,
    n_groups: tl.constexpr,
    BITS: tl.constexpr,
):
    """
    KIVI 写入路径:
      1) 把当前 (token, head) 的 fp16 K/V 同步写入 residual buffer (供 decode 阶段的尾部直接用)
      2) 同步写入 paged cache (量化):
         - K: 在当前 token 的 head_dim 内, 每 group_size 个元素求 (min, max) -> scale/zero -> 量化
              scale/zero 存入 (block_idx, kv_head_idx, group_idx)
         - V: 求 amax -> scale -> 对称量化, scale 存入 (block_idx, block_offset, kv_head_idx)

    注意: K 的"块内 per-channel"严格说应在整个块 (block_size 个 token) 收集完后再求 channel min/max。
          这里采用每 token 独立标定 (per-token-per-channel-group), 等价于把分组从 (head_dim/group) 扩展为
          (block_size, head_dim/group)。这是与 paging 兼容的最简实现, 避免跨 token 同步; 精度比原论文
          的"整段序列 per-channel"略差, 但与 paging 块独立性完全自洽。
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    head_offsets = tl.arange(0, head_dim)
    input_offset = (token_idx * num_kv_heads * head_dim +
                    head_idx * head_dim +
                    head_offsets)
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim +
                    block_offset * num_kv_heads * head_dim +
                    head_idx * head_dim +
                    head_offsets)

    key = tl.load(key_ptr + input_offset).to(tl.float32)
    value = tl.load(value_ptr + input_offset).to(tl.float32)

    # ---- 1) 写 residual fp16 buffer ----
    # residual 是 ring buffer: 行号 = slot, 列号 = (slot_idx % residual_length)
    seq_slot = tl.load(residual_slots_ptr + token_idx)
    res_pos = slot_idx % residual_length  # 简化: 直接按 slot 取模, decode 端用同样的 mapping
    res_offset = (seq_slot * residual_length * num_kv_heads * head_dim +
                  res_pos * num_kv_heads * head_dim +
                  head_idx * head_dim +
                  head_offsets)
    tl.store(k_residual_ptr + res_offset, key.to(k_residual_ptr.dtype.element_ty))
    tl.store(v_residual_ptr + res_offset, value.to(v_residual_ptr.dtype.element_ty))

    # ---- 2a) K 量化: 对每个 group 求 (min, max), 量化为 [0, 2^bits-1] ----
    # qmax_k 是 constexpr 计算结果 (Python int), 不要再用 float() 包 tl 标量
    qmax_k: tl.constexpr = (1 << BITS) - 1  # 例如 2-bit -> 3, 4-bit -> 15
    qmax_k_f: tl.constexpr = float(qmax_k)
    group_ids = head_offsets // group_size  # 每个 head_dim 元素属于哪个 group
    # 对每个 group 单独算 min/max: 用 mask + reduction
    for g in tl.static_range(0, n_groups):
        mask_g = group_ids == g
        # 计算 min/max: 用 where 给非本 group 设极值
        xmax = tl.max(tl.where(mask_g, key,  -1.0e30))
        xmin = tl.min(tl.where(mask_g, key,   1.0e30))
        rng = xmax - xmin
        scale = tl.where(rng > 0, rng / qmax_k_f, 1.0)
        zero = xmin
        # 量化: q = round((x - zero) / scale), 仅本 group 元素生效
        q = (key - zero) / scale
        # round to nearest, clamp 到 [0, qmax_k]
        q = tl.minimum(tl.maximum(q + 0.5 * tl.where(q >= 0, 1.0, -1.0), 0.0), qmax_k_f)
        q_int = q.to(tl.int32)
        # 写量化 K (int8 容器), 仅写本 group 的位置
        tl.store(k_cache_ptr + cache_offset, q_int.to(k_cache_ptr.dtype.element_ty), mask=mask_g)
        # 写 scale/zero
        # shape: (num_blocks, block_size, num_kv_heads, n_groups)
        # 注意必须包含 block_offset 维: 同一物理块内多个 token 都要写自己的 scale/zero,
        # 否则后写覆盖先写, decode 反量化全错
        sz_off = (block_idx * block_size * num_kv_heads * n_groups +
                  block_offset * num_kv_heads * n_groups +
                  head_idx * n_groups + g)
        tl.store(k_scale_ptr + sz_off, scale)
        tl.store(k_zero_ptr + sz_off, zero)

    # ---- 2b) V 量化: per-token-per-head 对称, qmax = 2^(bits-1) - 1 (有符号空间) ----
    qmax_v: tl.constexpr = (1 << (BITS - 1)) - 1 if BITS > 1 else 1
    qmax_v_f: tl.constexpr = float(qmax_v)
    neg_qmax_v_f: tl.constexpr = float(-qmax_v)
    v_amax = tl.max(tl.abs(value))
    v_scale = tl.where(v_amax > 0, v_amax / qmax_v_f, 1.0)
    vq = value / v_scale
    # round + clamp 到 [-qmax_v, qmax_v]
    vq = tl.minimum(tl.maximum(vq + 0.5 * tl.where(vq >= 0, 1.0, -1.0),
                               neg_qmax_v_f), qmax_v_f)
    vq_int = vq.to(tl.int32)
    tl.store(v_cache_ptr + cache_offset, vq_int.to(v_cache_ptr.dtype.element_ty))
    vs_off = (block_idx * block_size * num_kv_heads +
              block_offset * num_kv_heads + head_idx)
    tl.store(v_scale_ptr + vs_off, v_scale)


def store_kvcache_kivi(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    k_zero: torch.Tensor,
    v_scale: torch.Tensor,
    k_residual: torch.Tensor,
    v_residual: torch.Tensor,
    slot_mapping: torch.Tensor,
    residual_slots: torch.Tensor,
    residual_lens: torch.Tensor,
    block_size: int,
    group_size: int,
    residual_length: int,
    bits: int,
):
    """
    KIVI 写入封装: 同时写 residual fp16 buffer 与 paged 量化 cache。
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    key = key.contiguous()
    value = value.contiguous()
    assert head_dim % group_size == 0
    n_groups = head_dim // group_size

    grid = (num_tokens, num_kv_heads)
    store_kvcache_kivi_kernel[grid](
        key, value, k_cache, v_cache,
        k_scale, k_zero, v_scale,
        k_residual, v_residual,
        slot_mapping, residual_slots, residual_lens,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        group_size=group_size,
        residual_length=residual_length,
        n_groups=n_groups,
        BITS=bits,
    )


@triton.jit
def paged_attention_decode_kivi_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,        # (num_blocks, num_kv_heads, n_groups)
    k_zero_ptr,         # 同上
    v_scale_ptr,        # (num_blocks, block_size, num_kv_heads)
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    group_size: tl.constexpr,
    n_groups: tl.constexpr,
):
    """
    decode 阶段, 从 KIVI 量化 paged cache 读取并反量化后做注意力。
    与 paged_attention_decode_kernel 同构, 仅 K/V 载入后多一步反量化。
    本实现"读全部 paged + residual 同时写入" — residual buffer 仅用于
    需要更高精度的尾段计算 (可选), 这里为最小可跑, decode 路径只走 paged 反量化。
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    context_len = tl.load(context_lens_ptr + batch_idx)

    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    group_ids = offs_d // group_size  # 每个 head_dim 元素属于的 group

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N
        if token_start < context_len:
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len

            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        if physical_block_idx != -1:
                            # 载入 int8 K 并反量化
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            k_q = tl.load(k_cache_ptr + k_offset).to(tl.float32)
                            # 加载该 (block, slot, head, group) 的 scale/zero, 按 offs_d 广播
                            # k_scale shape: (num_blocks, block_size, num_kv_heads, n_groups)
                            sz_base = (physical_block_idx * block_size * num_kv_heads * n_groups +
                                       block_offset * num_kv_heads * n_groups +
                                       kv_head_idx * n_groups)
                            # 用循环展开各 group
                            k_vec = tl.zeros([head_dim], dtype=tl.float32)
                            for g in tl.static_range(0, n_groups):
                                mask_g = group_ids == g
                                s = tl.load(k_scale_ptr + sz_base + g)
                                z = tl.load(k_zero_ptr + sz_base + g)
                                k_vec = tl.where(mask_g, k_q * s + z, k_vec)

                            score = tl.sum(q * k_vec) * scale
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)

            qk = tl.where(mask_n, qk, -1e10)

            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            acc = acc * alpha
            l_i = l_i * alpha

            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        if physical_block_idx != -1:
                            v_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            v_q = tl.load(v_cache_ptr + v_offset).to(tl.float32)
                            vs_off = (physical_block_idx * block_size * num_kv_heads +
                                      block_offset * num_kv_heads + kv_head_idx)
                            v_s = tl.load(v_scale_ptr + vs_off)
                            v_vec = v_q * v_s

                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            acc = acc + weight * v_vec
                            l_i = l_i + weight

            m_i = m_i_new

    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output.to(output_ptr.dtype.element_ty))


def paged_attention_decode_kivi(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    k_zero: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """
    KIVI decode 反量化 paged attention 的 Python 封装。
    输出 dtype 与 query 一致。
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    query = query.contiguous()
    output = torch.empty_like(query)
    BLOCK_N = 64 if head_dim <= 128 else 32
    assert head_dim % group_size == 0
    n_groups = head_dim // group_size
    grid = (batch_size, num_heads)
    paged_attention_decode_kivi_kernel[grid](
        output, query, k_cache, v_cache,
        k_scale, k_zero, v_scale,
        block_tables, context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
        group_size=group_size,
        n_groups=n_groups,
    )
    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
        block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])
        # KV cache 量化相关属性，默认不量化；由 ModelRunner.allocate_kv_cache 注入实际值
        self.kv_cache_dtype = 'auto'   # "auto" | "fp8_per_tensor" | "fp8_per_token_head" | "kivi_2bit" | "kivi_4bit"
        self.k_scale = None            # FP8/KIVI 量化时的 K scale 张量
        self.v_scale = None            # FP8/KIVI 量化时的 V scale 张量
        # KIVI 专用
        self.k_zero = None             # K 的 zero-point (KIVI 非对称量化)
        self.k_residual = None         # K 的 fp16 residual buffer
        self.v_residual = None         # V 的 fp16 residual buffer
        self.kivi_bits = 0
        self.kivi_group_size = 0
        self.kivi_residual_length = 0

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 路径选择: auto / fp8_* / kivi_*
        kivi_enabled = isinstance(self.kv_cache_dtype, str) and self.kv_cache_dtype.startswith('kivi')
        fp8_enabled = (self.kv_cache_dtype != 'auto') and not kivi_enabled

        # Store current k, v into cache if cache is allocated
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            # Ensure k, v are in the right shape: (num_tokens, num_kv_heads, head_dim)
            if k.dim() == 4:
                # Batched: (B, N, num_kv_heads, head_dim) -> reshape to (B*N, num_kv_heads, head_dim)
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # Already in correct shape (num_tokens, num_kv_heads, head_dim)
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()

            if kivi_enabled:
                # KIVI 路径: 同步写 paged 量化 cache + fp16 residual buffer
                store_kvcache_kivi(
                    k_to_store, v_to_store, k_cache, v_cache,
                    self.k_scale, self.k_zero, self.v_scale,
                    self.k_residual, self.v_residual,
                    context.slot_mapping, context.residual_slots, context.residual_lens,
                    self.block_size, self.kivi_group_size, self.kivi_residual_length, self.kivi_bits,
                )
            elif fp8_enabled:
                # FP8 量化写入：量化 + 写 scale
                store_kvcache_fp8(
                    k_to_store, v_to_store, k_cache, v_cache,
                    self.k_scale, self.v_scale,
                    context.slot_mapping, self.block_size, self.kv_cache_dtype,
                )
            else:
                store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            # Prefill: use flash attention
            # Varlen mode: (total_tokens, num_heads, head_dim)
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            if context.positions is not None and context.positions.numel() > 0:
                first_pos = int(context.positions[0].item())
                if first_pos != 0:
                    if fp8_enabled or kivi_enabled:
                        raise NotImplementedError(
                            "chunked prefill for quantized KV cache needs a "
                            "dequantizing paged prefill/extend attention path"
                        )
                    o = paged_attention_prefill_torch(
                        q, k_cache, v_cache,
                        context.block_tables, context.context_lens,
                        cu_seqlens, context.positions,
                        scale, self.num_heads, self.num_kv_heads,
                        self.block_size,
                    )
                    return o.reshape(o.shape[0], self.num_heads * self.head_dim)
            
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale, 
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            # Output: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            if kivi_enabled:
                # KIVI decode: 反量化 paged cache 后做 attention
                o = paged_attention_decode_kivi(
                    q, k_cache, v_cache,
                    self.k_scale, self.k_zero, self.v_scale,
                    context.block_tables, context.context_lens,
                    scale, self.num_heads, self.num_kv_heads, self.head_dim,
                    self.block_size, self.kivi_group_size,
                )
            elif fp8_enabled:
                # decode 阶段从 FP8 cache 读取并反量化
                o = paged_attention_decode_fp8(
                    q, k_cache, v_cache, self.k_scale, self.v_scale,
                    context.block_tables, context.context_lens,
                    scale, self.num_heads, self.num_kv_heads, self.head_dim,
                    self.block_size, self.kv_cache_dtype,
                )
            else:
                o = paged_attention_decode(
                    q, 
                    k_cache, 
                    v_cache,
                    context.block_tables,
                    context.context_lens,
                    scale,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.block_size
                )
            # o: (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    # Example usage
    layer = Attention(num_heads=8, head_dim=64).cuda()
    B, N, D = 4, 1024, 512
    q = torch.randn(B, N, D).cuda()
    k = torch.randn(B, N, D).cuda()
    v = torch.randn(B, N, D).cuda()
    layer.k_cache = torch.zeros(B, N, D).cuda()
    layer.v_cache = torch.zeros(B, N, D).cuda()
    slot_mapping = torch.arange(N).cuda()

    for _ in range(10):  # Warm-up iterations
        _ = layer(q, k, v)

    import time
    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")