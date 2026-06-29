import torch
import time
import triton 
import triton.language as tl
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from myvllm.layers.attention import (
    paged_attention_decode,
    store_kvcache,
    quantize_fp8_per_head,
)

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
    """Optimized paged attention kernel for decode phase."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    
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
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            k_vec = tl.load(k_cache_ptr + k_offset)
                            
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
                            v_vec = tl.load(v_cache_ptr + v_offset)
                            
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            
                            acc = acc + weight * v_vec
                            l_i = l_i + weight
            
            m_i = m_i_new
    
    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode_triton(
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
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    query = query.contiguous()
    output = torch.empty_like(query)
    
    BLOCK_N = 64 if head_dim <= 128 else 32
    grid = (batch_size, num_heads)
    
    paged_attention_decode_kernel[grid](
        output, query, k_cache, v_cache, block_tables, context_lens,
        scale=scale, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, block_size=block_size, 
        max_num_blocks=max_num_blocks, BLOCK_N=BLOCK_N,
    )
    return output


def decode_torch_optimized(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    
    max_context_len = context_lens.max().item()
    
    padded_k = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    padded_v = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    
    for i in range(batch_size):
        seq_len = context_lens[i].item()
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        
        valid_blocks = block_tables[i, :num_blocks_needed]
        valid_blocks = valid_blocks[valid_blocks != -1]
        
        if len(valid_blocks) > 0:
            gathered_k = k_cache[valid_blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
            gathered_v = v_cache[valid_blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
            
            padded_k[i, :seq_len] = gathered_k
            padded_v[i, :seq_len] = gathered_v
    
    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        padded_k = padded_k.repeat_interleave(num_groups, dim=2)
        padded_v = padded_v.repeat_interleave(num_groups, dim=2)
    
    q = q.unsqueeze(2)
    padded_k = padded_k.transpose(1, 2)
    padded_v = padded_v.transpose(1, 2)
    
    attn_scores = torch.matmul(q, padded_k.transpose(-2, -1)) * scale
    
    mask = torch.arange(max_context_len, device=device)[None, :] < context_lens[:, None]
    mask = mask[:, None, None, :]
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, padded_v).squeeze(2)
    
    return output


def naive_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """
    Naive decode implementation
    This reconstructs full K, V sequences and uses standard PyTorch attention.
    """
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    
    max_context_len = context_lens.max().item()
    
    # Gather K, V into full sequences (inefficient for large contexts)
    all_k = []
    all_v = []
    
    for i in range(batch_size):
        seq_len = context_lens[i].item()
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        
        seq_k_list = []
        seq_v_list = []
        for block_idx in range(num_blocks_needed):
            block_id = block_tables[i, block_idx].item()
            if block_id == -1:
                break
            seq_k_list.append(k_cache[block_id])
            seq_v_list.append(v_cache[block_id])
        
        if len(seq_k_list) > 0:
            seq_k = torch.cat(seq_k_list, dim=0)[:seq_len]
            seq_v = torch.cat(seq_v_list, dim=0)[:seq_len]
            all_k.append(seq_k)
            all_v.append(seq_v)
    
    # Pad sequences
    padded_k = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim,
                           device=device, dtype=dtype)
    padded_v = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim,
                           device=device, dtype=dtype)
    
    for i, (k_seq, v_seq) in enumerate(zip(all_k, all_v)):
        seq_len = len(k_seq)
        padded_k[i, :seq_len] = k_seq
        padded_v[i, :seq_len] = v_seq
    
    # GQA
    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        padded_k = padded_k.repeat_interleave(num_groups, dim=2)
        padded_v = padded_v.repeat_interleave(num_groups, dim=2)
    
    # Reshape and compute attention
    q = q.unsqueeze(2)  # (B, H, 1, D)
    padded_k = padded_k.transpose(1, 2)  # (B, H, N, D)
    padded_v = padded_v.transpose(1, 2)  # (B, H, N, D)
    
    # This is the inefficient part - materializes full attention matrix
    attn_scores = torch.matmul(q, padded_k.transpose(-2, -1)) * scale
    
    mask = torch.arange(max_context_len, device=device)[None, :] < context_lens[:, None]
    mask = mask[:, None, None, :]
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, padded_v).squeeze(2)
    
    return output



def setup_test_data(batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size, device='cuda'):
    """Setup fp16 test data for benchmarking"""
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    max_num_blocks = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * max_num_blocks
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    block_tables = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(batch_size, max_num_blocks)
    context_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    scale = 1.0 / (head_dim ** 0.5)
    return q, k_cache, v_cache, block_tables, context_lens, scale


def setup_fp8_test_data(batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size, device='cuda'):
    """
    Build fp8 KV cache from fp16 data via quantize_fp8_per_head.
    Returns the same query/block_tables/context_lens, but fp8 caches + scale tensors.
    """
    q, k_cache_fp16, v_cache_fp16, block_tables, context_lens, scale = setup_test_data(
        batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size, device
    )
    total_blocks, block_size_, _, _ = k_cache_fp16.shape
    # reshape to (total_blocks * block_size, num_kv_heads, head_dim) for quantize
    k_flat = k_cache_fp16.view(total_blocks * block_size_, num_kv_heads, head_dim)
    v_flat = v_cache_fp16.view(total_blocks * block_size_, num_kv_heads, head_dim)

    k_fp8_flat, k_scale_flat = quantize_fp8_per_head(k_flat)  # (T, H, D), (T, H, 1)
    v_fp8_flat, v_scale_flat = quantize_fp8_per_head(v_flat)

    k_cache_fp8 = k_fp8_flat.view(total_blocks, block_size_, num_kv_heads, head_dim)
    v_cache_fp8 = v_fp8_flat.view(total_blocks, block_size_, num_kv_heads, head_dim)
    # scale layout needed by kernel: (num_blocks, block_size, num_kv_heads)
    k_scale = k_scale_flat.view(total_blocks, block_size_, num_kv_heads)
    v_scale = v_scale_flat.view(total_blocks, block_size_, num_kv_heads)

    return q, k_cache_fp16, v_cache_fp16, k_cache_fp8, v_cache_fp8, k_scale, v_scale, block_tables, context_lens, scale


def benchmark(batch_size, seq_len, num_heads=32, num_kv_heads=8,
              head_dim=128, block_size=16, num_iterations=100):
    """Compare fp16 vs fp8 KV cache: latency, memory, accuracy."""
    print(f"\n{'='*70}")
    print(f"batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, block_size={block_size}")
    print(f"{'='*70}")

    q, k_fp16, v_fp16, k_fp8, v_fp8, k_scale, v_scale, \
        block_tables, context_lens, scale = setup_fp8_test_data(
            batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size
        )

    results = {}

    # ── 1. Naive PyTorch (fp16) ──────────────────────────────────────────────
    print("\n1. Naive PyTorch (fp16)...")
    for _ in range(10):
        naive_decode_attention(q, k_fp16, v_fp16, block_tables, context_lens,
                               scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        out_ref = naive_decode_attention(q, k_fp16, v_fp16, block_tables, context_lens,
                                         scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    results['Naive PyTorch (fp16)'] = (time.perf_counter() - t0) / num_iterations
    print(f"   Time: {results['Naive PyTorch (fp16)']*1000:.3f}ms")

    # ── 2. Optimized PyTorch (fp16) ──────────────────────────────────────────
    print("\n2. Optimized PyTorch (fp16)...")
    for _ in range(10):
        decode_torch_optimized(q, k_fp16, v_fp16, block_tables, context_lens,
                               scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        decode_torch_optimized(q, k_fp16, v_fp16, block_tables, context_lens,
                               scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    results['Optimized PyTorch (fp16)'] = (time.perf_counter() - t0) / num_iterations
    print(f"   Time: {results['Optimized PyTorch (fp16)']*1000:.3f}ms")

    # ── 3. Triton decode (fp16) ───────────────────────────────────────────────
    print("\n3. Triton paged attention (fp16)...")
    for _ in range(10):
        paged_attention_decode_triton(q, k_fp16, v_fp16, block_tables, context_lens,
                                      scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        out_triton_fp16 = paged_attention_decode_triton(
            q, k_fp16, v_fp16, block_tables, context_lens,
            scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    results['Triton (fp16)'] = (time.perf_counter() - t0) / num_iterations
    print(f"   Time: {results['Triton (fp16)']*1000:.3f}ms")

    # ── 4. Triton decode (fp8 KV cache) ──────────────────────────────────────
    print("\n4. Triton paged attention (fp8 KV cache)...")
    for _ in range(10):
        paged_attention_decode(q, k_fp8, v_fp8, block_tables, context_lens,
                               scale, num_heads, num_kv_heads, head_dim, block_size,
                               k_scale=k_scale, v_scale=v_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        out_fp8 = paged_attention_decode(q, k_fp8, v_fp8, block_tables, context_lens,
                                          scale, num_heads, num_kv_heads, head_dim, block_size,
                                          k_scale=k_scale, v_scale=v_scale)
    torch.cuda.synchronize()
    results['Triton (fp8 KV)'] = (time.perf_counter() - t0) / num_iterations
    print(f"   Time: {results['Triton (fp8 KV)']*1000:.3f}ms")

    # ── 5. Memory comparison ─────────────────────────────────────────────────
    mem_fp16 = k_fp16.nbytes + v_fp16.nbytes
    mem_fp8  = k_fp8.nbytes + v_fp8.nbytes + k_scale.nbytes + v_scale.nbytes
    print(f"\nMemory (KV cache):")
    print(f"  fp16:          {mem_fp16 / 1024:.1f} KB")
    print(f"  fp8 + scales:  {mem_fp8  / 1024:.1f} KB  ({100*(1-mem_fp8/mem_fp16):.1f}% reduction)")

    # ── 6. Accuracy comparison ────────────────────────────────────────────────
    out_ref_f = out_ref.float()
    out_fp8_f = out_fp8.float()
    max_err = (out_ref_f - out_fp8_f).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_ref_f.reshape(1, -1), out_fp8_f.reshape(1, -1)
    ).item()
    print(f"\nAccuracy (fp8 KV vs fp16 ref):")
    print(f"  Max abs error: {max_err:.6f}")
    print(f"  Cosine sim:    {cos_sim:.6f}")

    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PAGED ATTENTION DECODE BENCHMARK: fp16 vs fp8 KV cache")
    print("="*70)

    benchmark(batch_size=1,  seq_len=512,  num_iterations=100)
    benchmark(batch_size=4,  seq_len=512,  num_iterations=100)
    benchmark(batch_size=16, seq_len=256,  num_iterations=50)
    benchmark(batch_size=4,  seq_len=2048, num_iterations=20)