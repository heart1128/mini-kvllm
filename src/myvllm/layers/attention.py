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
    
    # load key and value value floats from the pointers's memory
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
    k_cache_ptr,        # FP8 K cache
    v_cache_ptr,        # FP8 V cache
    k_scale_ptr,        # K 反量化 scale
    v_scale_ptr,        # V 反量化 scale
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PER_TOKEN_HEAD: tl.constexpr,  # 1=per_token_head; 0=per_tensor
):
    """
    decode 阶段的 paged attention，读取 FP8 cache 并在计算前反量化。
    与 paged_attention_decode_kernel 逻辑一致，区别在于:
      - K/V 以 FP8 载入后乘以对应 scale 还原为 float32
      - per_token_head: scale 按 (block, slot, kv_head) 逐位置读取
      - per_tensor:     scale 为整层共享的单个标量
    Grid 布局: (batch_size, num_heads)
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # GQA: 该 query head 对应的 KV head
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    context_len = tl.load(context_lens_ptr + batch_idx)

    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    # per_tensor 模式: scale 是单个标量，提前载入复用
    if PER_TOKEN_HEAD == 0:
        k_scale_scalar = tl.load(k_scale_ptr)
        v_scale_scalar = tl.load(v_scale_ptr)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N
        if token_start < context_len:
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len

            # 计算当前 chunk 内每个 token 的注意力分数
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
                            # 载入 FP8 的 K 向量并转 float32
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            k_vec = tl.load(k_cache_ptr + k_offset).to(tl.float32)
                            # 反量化: k_fp16 = k_fp8 * scale
                            if PER_TOKEN_HEAD:
                                ks_off = (physical_block_idx * block_size * num_kv_heads +
                                          block_offset * num_kv_heads + kv_head_idx)
                                k_s = tl.load(k_scale_ptr + ks_off)
                            else:
                                k_s = k_scale_scalar
                            k_vec = k_vec * k_s

                            score = tl.sum(q * k_vec) * scale
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)

            qk = tl.where(mask_n, qk, -1e10)

            # online softmax 更新
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            acc = acc * alpha
            l_i = l_i * alpha

            # 载入 FP8 的 V 向量、反量化并加权累加
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
                            v_vec = tl.load(v_cache_ptr + v_offset).to(tl.float32)
                            if PER_TOKEN_HEAD:
                                vs_off = (physical_block_idx * block_size * num_kv_heads +
                                          block_offset * num_kv_heads + kv_head_idx)
                                v_s = tl.load(v_scale_ptr + vs_off)
                            else:
                                v_s = v_scale_scalar
                            v_vec = v_vec * v_s

                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            acc = acc + weight * v_vec
                            l_i = l_i + weight

            m_i = m_i_new

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
        self.kv_cache_dtype = 'auto'   # "auto" | "fp8_per_tensor" | "fp8_per_token_head"
        self.k_scale = None            # FP8 量化时的 K scale 张量
        self.v_scale = None            # FP8 量化时的 V scale 张量

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 是否启用 FP8 量化路径
        fp8_enabled = self.kv_cache_dtype != 'auto'

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

            if fp8_enabled:
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
            
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale, 
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            # Output: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            if fp8_enabled:
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