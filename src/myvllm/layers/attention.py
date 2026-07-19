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
    """将 K/V 写入分页 KV cache。

    并行粒度是 ``(token, kv_head)``：每个 program 负责一个 token 的一个 KV
    head，并沿 ``head_dim`` 维度向量化读写。这样做的性能收益是：

    - token 和 head 两个维度可以同时暴露并行度，prefill 时可启动大量 program；
    - 一次 program 连续访问完整 head_dim，能形成合并访存，避免逐元素写入；
    - K 和 V 在同一个 program 中读取并写入，复用 block/slot 地址计算；
    - 不需要把逻辑连续 cache 搬移到物理连续区域，分页 cache 可以直接复用空闲 block。

    ``slot_mapping`` 把逻辑 token 映射到物理 slot，再拆成
    ``(block_idx, block_offset)``。``-1`` 表示该 token 命中 prefix cache，不需要重复写入。

    主要代价是：每个 token/head 都要做一次整数除法、取模和非连续物理地址计算；
    因此该 kernel 更偏向内存带宽瓶颈，通常不应使用过小的 head_dim 或过细的并行粒度。
    Grid: ``(num_tokens, num_kv_heads)``。
    Cache: ``(num_blocks, block_size, num_kv_heads, head_dim)``。
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


# ===================== KV Cache 量化相关 =====================
# FP8 e4m3fn 与 INT8 的量化范围，用于计算 scale (amax / quant_max)
FP8_E4M3_MAX = 448.0
INT8_MAX = 127.0
INT8_MIN = -128.0


# ===================== Randomized Hadamard Transform (RHT) =====================
# 用于 INT4 groupwise 量化前对 K/V 做 Gaussianize 变换，改善量化质量。
# 算法: RHT(x) = WHT(x * D₁)  其中 D₁ 为固定 ±1 符号向量（固定种子，确定性）
# 逆变换: RHT⁻¹(y) = WHT(y) * D₁
# 参考 vLLM int4_per_token_head.py 中的 single_rht / fast_hadamard_transform
_RHT_SIGNS_CACHE: dict = {}
_HADAMARD_MATRIX_CACHE: dict = {}


def _get_hadamard_matrix(
    d: int,
    dtype: torch.dtype,
    device: torch.device,
    inverse: bool,
) -> torch.Tensor:
    """缓存融合随机符号的 Hadamard 矩阵。

    按行向量约定，正向 RHT 为 x @ D @ H，符号乘在 H 的行；
    逆向为 x @ H @ D，符号乘在 H 的列。
    """
    key = (d, dtype, str(device), inverse)
    if key not in _HADAMARD_MATRIX_CACHE:
        matrix = torch.ones(1, 1, dtype=torch.float32)
        while matrix.shape[0] < d:
            top = torch.cat((matrix, matrix), dim=1)
            bottom = torch.cat((matrix, -matrix), dim=1)
            matrix = torch.cat((top, bottom), dim=0)
        fused_signs = _get_rht_signs(d, device).float()
        matrix = matrix.to(device=device)
        matrix = matrix * (fused_signs[None, :] if inverse else fused_signs[:, None])
        _HADAMARD_MATRIX_CACHE[key] = matrix.to(dtype=dtype).contiguous()
    return _HADAMARD_MATRIX_CACHE[key]


@triton.jit
def _hadamard_mma_kernel(
    x_ptr,
    h_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """用一次 tile 化矩阵乘完成多行 Walsh-Hadamard Transform。

    一个 program 处理 ``BLOCK_M`` 行，加载 ``[BLOCK_M, d]`` 输入和完整的
    ``[d, d]`` Hadamard 矩阵，再通过 ``tl.dot`` 计算矩阵乘。它把原本多轮
    Python/Torch butterfly 和中间 Tensor，压缩成一次 GPU kernel，主要收益是：

    - 减少多轮 butterfly 的 kernel launch 和中间结果读写；
    - ``tl.dot`` 可映射到 GPU 的矩阵乘单元，提高 d 为 16/32/64/128 时的吞吐；
    - 同一个 Hadamard 矩阵被多个输入行复用，提升权重缓存复用率。

    代价是矩阵是 ``O(d^2)`` 的，并且每个 program 会加载完整矩阵；因此它适合
    head_dim 较小且需要批量处理多行的 RHT，不适合把它当成通用大矩阵乘。
    """
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    input_cols = tl.arange(0, d)
    output_cols = tl.arange(0, d)
    row_mask = rows < n_rows
    x = tl.load(
        x_ptr + rows[:, None] * d + input_cols[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    h = tl.load(h_ptr + input_cols[:, None] * d + output_cols[None, :])
    out = tl.dot(x, h)
    tl.store(
        out_ptr + rows[:, None] * d + output_cols[None, :],
        out,
        mask=row_mask[:, None],
    )


def _triton_hadamard_transform(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """CUDA fast path：单个 Triton MMA kernel 完成含随机符号的 RHT。"""
    d = x.shape[-1]
    original_shape = x.shape
    original_dtype = x.dtype
    work_dtype = torch.bfloat16 if original_dtype == torch.float32 else original_dtype
    x2d = x.contiguous().to(work_dtype).reshape(-1, d)
    output_dtype = torch.float32 if original_dtype == torch.float32 else original_dtype
    out2d = torch.empty(x2d.shape, dtype=output_dtype, device=x.device)
    matrix = _get_hadamard_matrix(d, work_dtype, x.device, inverse=inverse)
    block_m = 16
    _hadamard_mma_kernel[(triton.cdiv(x2d.shape[0], block_m),)](
        x2d,
        matrix,
        out2d,
        n_rows=x2d.shape[0],
        d=d,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=1,
    )
    return out2d.reshape(original_shape)


def _get_rht_signs(d: int, device: torch.device) -> torch.Tensor:
    """返回长度 d 的确定性 ±1 符号向量（固定种子，按 device 缓存）。"""
    key = (d, str(device))
    if key not in _RHT_SIGNS_CACHE:
        gen = torch.Generator(device=device)
        gen.manual_seed(0x9E3779B9)
        signs = 2.0 * torch.bernoulli(torch.full((d,), 0.5, device=device), generator=gen) - 1.0
        _RHT_SIGNS_CACHE[key] = signs
    return _RHT_SIGNS_CACHE[key]


def _fast_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard Transform（butterfly 实现），最后一维须为 2 的幂。"""
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"head_dim 必须为 2 的幂，当前 {d}"
    h = 1
    while h < d:
        # 每轮在长度 2h 的块内做 butterfly，保持标准 Walsh-Hadamard 排列。
        xv = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        a = xv[..., 0, :]
        b = xv[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(x.shape)
        h <<= 1
    return x


def single_rht(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """单轮 Randomized Hadamard Transform: WHT(x * D₁) 或逆变换 WHT(x) * D₁。

    Args:
        x:       输入张量，最后一维为 head_dim（须为 2 的幂）
        inverse: True 表示逆变换（用于 decode 输出）

    Returns:
        变换后的张量，形状与 x 相同
    """
    d = x.shape[-1]
    if x.is_cuda and 16 <= d <= 128 and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return _triton_hadamard_transform(x, inverse=inverse)
    d1 = _get_rht_signs(d, x.device)
    if inverse:
        # 逆变换: WHT(x) * D₁  (注意 WHT 是自逆的，差一个 d 的系数)
        return _fast_hadamard(x) * d1
    # 正变换: WHT(x * D₁)
    return _fast_hadamard(x * d1)


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
    """FP8 量化并写入分页 KV cache。

    每个 program 负责一个 ``(token, kv_head)``，沿 ``head_dim`` 向量化加载 K/V，
    在同一 kernel 内完成 scale、裁剪、FP8 转换和 cache 写入，避免先生成 FP32
    量化 Tensor 再由另一个 kernel 搬运。

    ``per_token_head`` 需要对 head_dim 做 ``amax`` reduction，精度和动态范围更好，
    但会增加 reduction、scale 写入以及 decode 时 scale 读取；``per_tensor`` 只需
    读取一个标量，写入路径更轻，但所有历史 token 共用同一 scale，容易被异常值
    或后续分布变化截断。

    性能上，K/V 的向量化 load/store 和量化融合减少了全局内存往返；FP8 数据量
    约为 FP16 的一半，可降低 KV cache 写入和后续 decode 的显存带宽压力。代价是
    每个 token/head 都要做地址映射，per-token-head 还要额外写 FP32 scale。
    Grid: ``(num_tokens, num_kv_heads)``。

    量化公式：
        ``scale = max(abs(x)) / FP8_MAX``（per-token-head）或外部标量（per-tensor）；
        ``x_fp8 = clamp(x / scale, -FP8_MAX, FP8_MAX)``。
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


@triton.jit
def store_kvcache_int8_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    INT8_QMAX: tl.constexpr,
    INT8_QMIN: tl.constexpr,
):
    """将 K/V 按 per-token-head 动态 scale 量化为 INT8 并写入分页 cache。

    一个 program 处理一个 token/head，先沿 head_dim 求 amax，再完成 scale 写入、
    rounding、裁剪和 INT8 写入。INT8 相比 FP16 将 cache 数据量降到约四分之一，
    能显著降低长上下文 decode 的显存容量和读带宽；但每个 token/head 都要做
    reduction，并额外写入 FP32 scale，因此短上下文或极小 batch 下量化开销可能
    抵消收益。

    这里显式实现 half-away-from-zero rounding，避免依赖不同硬件/类型转换的舍入
    规则造成数值偏差。K/V 的 scale 与物理 slot 对齐，decode 时可以直接按同一地址
    反量化，省去复杂的元数据查找。
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

    k_amax = tl.max(tl.abs(key))
    v_amax = tl.max(tl.abs(value))
    k_scale = tl.maximum(k_amax / INT8_QMAX, 1e-6)
    v_scale = tl.maximum(v_amax / INT8_QMAX, 1e-6)

    scale_offset = (block_idx * block_size * num_kv_heads +
                    block_offset * num_kv_heads +
                    head_idx)
    tl.store(k_scale_ptr + scale_offset, k_scale)
    tl.store(v_scale_ptr + scale_offset, v_scale)

    k_q = key * (1.0 / k_scale)
    v_q = value * (1.0 / v_scale)
    # Triton 写 int8 时会截断；先做 half-away-from-zero rounding 对齐 vLLM。
    k_q = tl.where(k_q >= 0, k_q + 0.5, k_q - 0.5)
    v_q = tl.where(v_q >= 0, v_q + 0.5, v_q - 0.5)
    k_q = tl.clamp(k_q, INT8_QMIN, INT8_QMAX)
    v_q = tl.clamp(v_q, INT8_QMIN, INT8_QMAX)

    tl.store(k_cache_ptr + cache_offset, k_q.to(k_cache_ptr.dtype.element_ty))
    tl.store(v_cache_ptr + cache_offset, v_q.to(v_cache_ptr.dtype.element_ty))


@triton.jit
def pack_int4_nibbles(lo, hi):
    """把两个 ``[0, 15]`` 的 uint4 值打包到一个 uint8。

    低 nibble 和高 nibble 共用一个 byte，使 INT4 cache 的数据占用减半；
    代价是 decode 读取后必须增加位运算和 unpack。
    """
    return (lo & 0xF) | ((hi & 0xF) << 4)


@triton.jit
def unpack_int4_nibbles(packed):
    """把一个 packed byte 拆成低、高两个 INT4 nibble。

    该操作是 INT4 decode 的必要开销；将它保留在 attention kernel 内，可以避免
    先把整个 cache 反量化成 FP16，节省一次大规模全局内存写入。
    """
    return packed & 0xF, (packed >> 4) & 0xF


@triton.jit
def pack_int4_scale_zp(scale, zero_point):
    """把 FP32 scale 的高 28 bit 与 4-bit zero-point 共存于一个 FP32 槽位。

    这样 scale 和 zero-point 可以一次读取，减少 INT4 cache 的元数据指针和全局
    load；代价是依赖 bitcast，必须由配套的 unpack 函数读取，不能把该 Tensor 当
    普通数值直接参与计算。
    """
    scale_bits = scale.to(tl.int32, bitcast=True)
    zp_bits = zero_point.to(tl.int32) & 0xF
    return ((scale_bits & -16) | zp_bits).to(tl.float32, bitcast=True)


@triton.jit
def unpack_int4_scale_zp(packed_scale):
    """从 packed FP32 槽位恢复 scale 和 4-bit zero-point。

    unpack 在 attention kernel 内即时完成，避免预先把所有 scale/zp 展开成更大的
    元数据 Tensor；这有利于显存占用和带宽，但会增加 decode 的位运算指令。
    """
    scale_bits = packed_scale.to(tl.int32, bitcast=True)
    zero_point = scale_bits & 0xF
    scale = (scale_bits & -16).to(tl.float32, bitcast=True)
    return scale, zero_point


@triton.jit
def quantize_int4_asymmetric(x, scale, zero_point):
    """执行非对称 INT4 的向量化量化、舍入和裁剪。

    作为 device inline helper 被 store kernel 调用，不产生独立 launch；这既避免了
    中间 Tensor，也让每个 group 的量化直接复用已计算的 scale/zp。量化数据更小，
    但 decode 需要承担对应的反量化指令和精度误差。
    """
    q = x * (1.0 / scale) + zero_point.to(tl.float32)
    q = tl.where(q >= 0, q + 0.5, q - 0.5)
    return tl.clamp(q, 0.0, 15.0).to(tl.int32)


@triton.jit
def store_kvcache_int4_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    packed_head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    """将 K/V 做 per-token-head 非对称 INT4 量化并打包写入分页 cache。

    每个 byte 存两个 4-bit nibble：偶数维写低 4 bit，奇数维写高 4 bit。
    因此 cache 数据量约为 FP16 的八分之一；长上下文 decode 主要受 KV 读带宽
    限制时，这种压缩通常比单纯减少计算更有价值。

    kernel 在一个 program 内完成 min/max reduction、scale/zero-point 计算、两路
    K/V 量化和 nibble packing，避免为中间量化结果分配额外 Tensor。代价是：
    - packed 地址不是原始 head_dim 地址，需要偶/奇维拆分和边界 mask；
    - decode 必须 unpack，并按 zero-point 反量化；
    - head_dim 不是偶数时最后一个 nibble 需要填充。

    scale 和 zero-point 与 ``(physical_block, block_offset, kv_head)`` 对齐，保证
    分页移动只改变 block table，不需要搬移量化元数据。
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    packed_offsets = tl.arange(0, packed_head_dim)
    even_offsets = packed_offsets * 2
    odd_offsets = even_offsets + 1
    even_mask = even_offsets < head_dim
    odd_mask = odd_offsets < head_dim

    input_base = token_idx * num_kv_heads * head_dim + head_idx * head_dim
    k_even = tl.load(key_ptr + input_base + even_offsets, mask=even_mask, other=0.0).to(tl.float32)
    k_odd = tl.load(key_ptr + input_base + odd_offsets, mask=odd_mask, other=0.0).to(tl.float32)
    v_even = tl.load(value_ptr + input_base + even_offsets, mask=even_mask, other=0.0).to(tl.float32)
    v_odd = tl.load(value_ptr + input_base + odd_offsets, mask=odd_mask, other=0.0).to(tl.float32)

    k_even_for_min = tl.where(even_mask, k_even, 1.0e20)
    k_odd_for_min = tl.where(odd_mask, k_odd, 1.0e20)
    v_even_for_min = tl.where(even_mask, v_even, 1.0e20)
    v_odd_for_min = tl.where(odd_mask, v_odd, 1.0e20)
    k_even_for_max = tl.where(even_mask, k_even, -1.0e20)
    k_odd_for_max = tl.where(odd_mask, k_odd, -1.0e20)
    v_even_for_max = tl.where(even_mask, v_even, -1.0e20)
    v_odd_for_max = tl.where(odd_mask, v_odd, -1.0e20)

    k_min = tl.minimum(tl.min(k_even_for_min), tl.min(k_odd_for_min))
    k_max = tl.maximum(tl.max(k_even_for_max), tl.max(k_odd_for_max))
    v_min = tl.minimum(tl.min(v_even_for_min), tl.min(v_odd_for_min))
    v_max = tl.maximum(tl.max(v_even_for_max), tl.max(v_odd_for_max))

    k_scale = tl.maximum((k_max - k_min) / 15.0, 1e-6)
    v_scale = tl.maximum((v_max - v_min) / 15.0, 1e-6)
    # 当前 Triton 的 tl.clamp 仅支持浮点类型，因此先 clamp 再转 int32。
    k_zp = tl.clamp(-k_min / k_scale + 0.5, 0.0, 15.0).to(tl.int32)
    v_zp = tl.clamp(-v_min / v_scale + 0.5, 0.0, 15.0).to(tl.int32)

    k_q_even = quantize_int4_asymmetric(k_even, k_scale, k_zp)
    k_q_odd = quantize_int4_asymmetric(k_odd, k_scale, k_zp)
    v_q_even = quantize_int4_asymmetric(v_even, v_scale, v_zp)
    v_q_odd = quantize_int4_asymmetric(v_odd, v_scale, v_zp)
    k_q_odd = tl.where(odd_mask, k_q_odd, 0)
    v_q_odd = tl.where(odd_mask, v_q_odd, 0)

    cache_offset = (block_idx * block_size * num_kv_heads * packed_head_dim +
                    block_offset * num_kv_heads * packed_head_dim +
                    head_idx * packed_head_dim +
                    packed_offsets)
    tl.store(k_cache_ptr + cache_offset, pack_int4_nibbles(k_q_even, k_q_odd).to(tl.uint8))
    tl.store(v_cache_ptr + cache_offset, pack_int4_nibbles(v_q_even, v_q_odd).to(tl.uint8))

    scale_offset = (block_idx * block_size * num_kv_heads +
                    block_offset * num_kv_heads +
                    head_idx)
    tl.store(k_scale_ptr + scale_offset, pack_int4_scale_zp(k_scale, k_zp))
    tl.store(v_scale_ptr + scale_offset, pack_int4_scale_zp(v_scale, v_zp))


def store_kvcache_int4(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
):
    """将 key/value 按 per-token-head 非对称 INT4 量化并 packed 写入 paged cache。"""
    num_tokens, num_kv_heads, head_dim = key.shape
    key = key.contiguous()
    value = value.contiguous()
    packed_head_dim = (head_dim + 1) // 2

    grid = (num_tokens, num_kv_heads)
    store_kvcache_int4_kernel[grid](
        key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        packed_head_dim=packed_head_dim,
        block_size=block_size,
    )


# ===================== INT4 Group-wise 量化 store kernel =====================

@triton.jit
def store_kvcache_int4_groupwise_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,        # K scale 张量，形状 (num_blocks, block_size, num_kv_heads, n_groups), float32
    v_scale_ptr,        # V scale 张量，同上
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    packed_head_dim: tl.constexpr,   # head_dim // 2
    block_size: tl.constexpr,
    group_size: tl.constexpr,        # 每个 group 包含的 head_dim 元素数
    n_groups: tl.constexpr,          # head_dim // group_size
):
    """INT4 group-wise 非对称量化：每个 ``(token, head, group)`` 独立计算 scale/zero-point。

    分组量化比整头量化更能适应不同维度的数值范围，通常可降低 INT4 误差；
    ``group_size`` 越小，scale 更精细但 scale 元数据、reduction 和 kernel 指令
    越多。kernel 同时完成偶/奇维加载、每组 min/max、量化、scale/zp 写入和 nibble
    packing，减少全局内存往返。

    主要性能权衡：cache 数据只有 4 bit，但每个 group 仍需存储 scale/zp；decode
    端要对每个 group 做 unpack/dequant。对于长上下文，带宽节省通常明显；对于
    短上下文，group 循环和量化元数据访问可能成为固定开销。

    分组量化公式：

    量化公式（每个 group 内）:
        scale = (max - min) / 15.0   (对应 [0, 15] 的 uint4 范围)
        zero_point = round(-min / scale)，clamped 到 [0, 15]
        q[i] = clamp(round(x[i] / scale + zero_point), 0, 15)
        dequant: x[i] = (q[i] - zero_point) * scale

    scale 与 zero_point 打包存储（zero-point steganography）：
        packed = (scale_bits & -16) | (zp & 0xF)
        存储于 k_scale_ptr[..., group_idx]

    Grid 布局: (num_tokens, num_kv_heads)
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # ---- 加载当前 (token, head) 的 key/value ----
    # 输入 shape: (num_tokens, num_kv_heads, head_dim)
    input_base = token_idx * num_kv_heads * head_dim + head_idx * head_dim
    packed_offsets = tl.arange(0, packed_head_dim)  # [0, 1, ..., packed_head_dim-1]
    even_offsets = packed_offsets * 2               # [0, 2, 4, ...]
    odd_offsets = even_offsets + 1                  # [1, 3, 5, ...]
    even_mask = even_offsets < head_dim
    odd_mask = odd_offsets < head_dim

    k_even = tl.load(key_ptr + input_base + even_offsets, mask=even_mask, other=0.0).to(tl.float32)
    k_odd  = tl.load(key_ptr + input_base + odd_offsets,  mask=odd_mask,  other=0.0).to(tl.float32)
    v_even = tl.load(value_ptr + input_base + even_offsets, mask=even_mask, other=0.0).to(tl.float32)
    v_odd  = tl.load(value_ptr + input_base + odd_offsets,  mask=odd_mask,  other=0.0).to(tl.float32)

    # 每个 packed 位置对应的 group id（group_size 个元素为一组）
    group_of_even = even_offsets // group_size   # packed 位置 i 对应 even 元素的 group
    group_of_odd  = odd_offsets  // group_size   # packed 位置 i 对应 odd 元素的 group

    # cache 地址：(num_blocks, block_size, num_kv_heads, packed_head_dim)
    cache_base = (block_idx * block_size * num_kv_heads * packed_head_dim +
                  block_offset * num_kv_heads * packed_head_dim +
                  head_idx * packed_head_dim)
    # scale 地址：(num_blocks, block_size, num_kv_heads, n_groups)
    scale_base = (block_idx * block_size * num_kv_heads * n_groups +
                  block_offset * num_kv_heads * n_groups +
                  head_idx * n_groups)

    # ---- 初始化量化结果向量（用于收集各 group 的量化值）----
    k_q_even = tl.zeros([packed_head_dim], dtype=tl.int32)
    k_q_odd  = tl.zeros([packed_head_dim], dtype=tl.int32)
    v_q_even = tl.zeros([packed_head_dim], dtype=tl.int32)
    v_q_odd  = tl.zeros([packed_head_dim], dtype=tl.int32)

    # ---- 逐 group 计算 scale/zero-point 并量化 ----
    for g in tl.static_range(0, n_groups):
        # 属于本 group 的 even/odd 位置掩码
        mask_even_g = (group_of_even == g) & even_mask
        mask_odd_g  = (group_of_odd  == g) & odd_mask

        # K: 收集本 group 所有元素求 min/max
        k_min_e = tl.min(tl.where(mask_even_g, k_even,  1.0e20))
        k_max_e = tl.max(tl.where(mask_even_g, k_even, -1.0e20))
        k_min_o = tl.min(tl.where(mask_odd_g,  k_odd,   1.0e20))
        k_max_o = tl.max(tl.where(mask_odd_g,  k_odd,  -1.0e20))
        k_min_g = tl.minimum(k_min_e, k_min_o)
        k_max_g = tl.maximum(k_max_e, k_max_o)
        k_scale_g = tl.maximum((k_max_g - k_min_g) / 15.0, 1e-6)
        k_zp_g = tl.clamp(-k_min_g / k_scale_g + 0.5, 0.0, 15.0).to(tl.int32)

        # V: 同上
        v_min_e = tl.min(tl.where(mask_even_g, v_even,  1.0e20))
        v_max_e = tl.max(tl.where(mask_even_g, v_even, -1.0e20))
        v_min_o = tl.min(tl.where(mask_odd_g,  v_odd,   1.0e20))
        v_max_o = tl.max(tl.where(mask_odd_g,  v_odd,  -1.0e20))
        v_min_g = tl.minimum(v_min_e, v_min_o)
        v_max_g = tl.maximum(v_max_e, v_max_o)
        v_scale_g = tl.maximum((v_max_g - v_min_g) / 15.0, 1e-6)
        v_zp_g = tl.clamp(-v_min_g / v_scale_g + 0.5, 0.0, 15.0).to(tl.int32)

        # 量化本 group 的 even/odd 元素
        kqe = tl.where(mask_even_g, quantize_int4_asymmetric(k_even, k_scale_g, k_zp_g), 0)
        kqo = tl.where(mask_odd_g,  quantize_int4_asymmetric(k_odd,  k_scale_g, k_zp_g), 0)
        vqe = tl.where(mask_even_g, quantize_int4_asymmetric(v_even, v_scale_g, v_zp_g), 0)
        vqo = tl.where(mask_odd_g,  quantize_int4_asymmetric(v_odd,  v_scale_g, v_zp_g), 0)
        k_q_even = k_q_even + kqe
        k_q_odd  = k_q_odd  + kqo
        v_q_even = v_q_even + vqe
        v_q_odd  = v_q_odd  + vqo

        # 写 scale（含 zero-point steganography）到 scale tensor 的 group 槽位
        tl.store(k_scale_ptr + scale_base + g, pack_int4_scale_zp(k_scale_g, k_zp_g))
        tl.store(v_scale_ptr + scale_base + g, pack_int4_scale_zp(v_scale_g, v_zp_g))

    # ---- 打包 nibble 并写入 cache ----
    k_packed = pack_int4_nibbles(k_q_even, k_q_odd).to(tl.uint8)
    v_packed = pack_int4_nibbles(v_q_even, v_q_odd).to(tl.uint8)
    tl.store(k_cache_ptr + cache_base + packed_offsets, k_packed)
    tl.store(v_cache_ptr + cache_base + packed_offsets, v_packed)


def store_kvcache_int4_groupwise(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    group_size: int,
    use_rht: bool = False,
):
    """INT4 group-wise 非对称量化写入 paged cache 的 Python 封装。

    Args:
        key/value:   (num_tokens, num_kv_heads, head_dim)
        k_cache/v_cache: (num_blocks, block_size, num_kv_heads, head_dim//2) uint8
        k_scale/v_scale: (num_blocks, block_size, num_kv_heads, n_groups) float32
        slot_mapping: (num_tokens,)
        block_size:   cache 块大小
        group_size:   量化分组大小（要求整除 head_dim）
        use_rht:      是否在量化前对 K/V 应用 RHT（改善量化质量）
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    assert head_dim % group_size == 0, f"head_dim {head_dim} 须被 group_size {group_size} 整除"
    n_groups = head_dim // group_size
    packed_head_dim = head_dim // 2  # head_dim 须为偶数

    if use_rht:
        # RHT 使 K/V 分布更接近高斯，改善 INT4 量化质量
        # 注意: RHT 后量化，decode 端 Q 也须做 RHT，输出须做逆 RHT
        key = single_rht(key.float())
        value = single_rht(value.float())

    key = key.contiguous()
    value = value.contiguous()

    grid = (num_tokens, num_kv_heads)
    store_kvcache_int4_groupwise_kernel[grid](
        key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        packed_head_dim=packed_head_dim,
        block_size=block_size,
        group_size=group_size,
        n_groups=n_groups,
    )


def store_kvcache_int8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
):
    """将 key/value 按 per-token-head 动态 scale 量化为 INT8 写入 paged cache。"""
    num_tokens, num_kv_heads, head_dim = key.shape
    key = key.contiguous()
    value = value.contiguous()

    grid = (num_tokens, num_kv_heads)
    store_kvcache_int8_kernel[grid](
        key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        INT8_QMAX=INT8_MAX,
        INT8_QMIN=INT8_MIN,
    )


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
    """变长序列的 FlashAttention 风格 prefill kernel。

    Grid ``(ceil(max_seq_len / BLOCK_M), num_heads, num_seqs)`` 中，一个 program
    负责一条序列、一个 Q head 的 ``BLOCK_M`` 个 query。Q 保存在片段连续的
    packed token buffer 中，``cu_seqlens_q`` 用来定位每条序列的起止位置。

    K/V 以 ``BLOCK_N`` tile 流式扫描，使用 ``m_i``、``l_i``、``acc`` 做 online
    softmax：不物化 ``[query_len, key_len]`` 的完整 attention 矩阵，因此显著降低
    prefill 的显存峰值，并允许 QK 和 PV 通过 ``tl.dot`` 使用矩阵乘硬件。

    这样做的性能收益来自：
    - Q/K/V tile 复用，减少重复的全局内存访问；
    - FP32 累积保证数值稳定，同时输出按目标 dtype 写回；
    - Q tile 和 K/V tile 提高并行度，适合长 prompt。

    主要代价是 block 尺寸必须平衡寄存器/共享存储和 occupancy；当前 grid 按最长
    序列分配，短序列会通过 early return 空转 program。GQA 通过多个 Q head 映射
    到较少的 KV head，减少 K/V 读取和 cache 容量。
    """
    # Program IDs
    start_m = tl.program_id(0) # block index 这个是这批中最大序列长度 / BLOCK_M的块大小，也就是处理一个block_M的token数量
    off_h = tl.program_id(1) # head index  这个是处理的第几个头（因为一个block_M是处理一个头的）
    seq_idx = tl.program_id(2) # sequence index 这个是处理的第几个序列

    # Determine which KV head to use (for GQA)
    # GQA一个Q对应K/V头，所以需要通过off_h // (num_heads // num_kv_heads)来确定使用哪个K/V头
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # Load sequence boundaries
    # 当前program的token的数量
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    # Early exit if this block is beyond sequence length
    # 边界，这个block_M超出了序列的长度，就直接返回，不处理了。
    # 因为start_m是按照最大的序列长度来分配的。小的序列超过自己的长度就跳出了
    if start_m * BLOCK_M >= seq_len:
        return
    
    # Offset for this block of queries
    # 找到第几个block，offs_m是这个block里面所有的token idx
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 这是head的idx，每个token的head数量都是固定的，所以这个offs_d也是固定的。
    offs_d = tl.arange(0, head_dim)
    
    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    # Q的第一维是全部token的q拼接的，取从当前序列开始第几个token的q，要跳过前面其他序列的q。
    # 所以需要加上seq_start，找到从当前序列开始的q。off_h是处理的第几个头，offs_d是处理的这个头的第几个token。
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    
    # Load Q block - shape (BLOCK_M, head_dim)
    mask_m = offs_m < seq_len # 不能超过当前序列的长度，否则会访问越界
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize output accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) # 到目前为止，每个 query token 的 softmax 分母
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10 # 到目前为止，每个 query token 的最大值
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32) # 到目前为止，每个 query token 的 weighted value累和
    
    # Number of blocks to process
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    # Q是外循环，KV分块进行内循环计算。这样Q-KV的计算可以并行化。
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
        m_ij = tl.max(qk, axis=1) # 当前 K block 内，每个 query 的最大 score
        m_i_new = tl.maximum(m_i, m_ij) # 历史最大 score 和当前 block 最大 score 的合并
        alpha = tl.exp(m_i - m_i_new) # 分母：历史最大 score 和当前 block 最大 score 的差值的指数
        p = tl.exp(qk - m_i_new[:, None]) # 分子：当前 score - 当前 block 最大 score，然后指数化
        
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
    # 统一为连续布局，保证 Triton 的二维/三维地址公式与实际 stride 一致。
    # 若输入本身已连续，这些调用只返回原 Tensor，不会复制数据。
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # 只分配最终输出；FlashAttention 不分配完整 [T, T] score/prob 矩阵。
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
    # 每个program负责一个seq的block大小
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
    """非量化分页 KV cache 的 decode attention。

    Grid ``(batch, num_heads)``：一个 program 负责一条请求的一个 Q head，沿历史
    token 扫描全部 K/V。逻辑 token 通过 ``block_tables`` 映射到物理 block，因此
    请求可以动态增长、不同请求可以共享物理内存而无需搬移整个 cache。

    每个历史 tile 都执行 QK、online softmax 和 PV 累积。online softmax 只保留
    当前 head 的 ``acc/l_i/m_i``，避免生成完整 attention 矩阵；分页 cache 则把
    显存容量和调度灵活性与序列长度解耦。

    当前实现为了处理动态 block table，在 ``BLOCK_N`` 内逐 token 查表并加载 K/V。
    它的正确性和地址逻辑简单，但会重复做整数除法、间接访存和标量 dot；长上下文
    decode 下通常受 KV 带宽和控制流限制。将 K/V 改成 ``[BLOCK_N, head_dim]``
    tile load、复用 physical block 映射，通常比微调逐元素算术更有性能收益。
    """
    # axis 0 绑定请求，axis 1 绑定 Q head；两者组合后每个 program 只写一个输出 head。
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # GQA/MQA 下多个 Q head 共享一个 KV head，整除操作将 Q head 映射到 KV head。
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # 读取当前请求真实可见长度；超过该长度的 cache slot 都不能参与 softmax。
    context_len = tl.load(context_lens_ptr + batch_idx)

    # head_dim 是编译期常量，生成连续维度下标，后续 load/store 可以向量化。
    offs_d = tl.arange(0, head_dim)
    # 按 [batch, head, dim] 的 row-major 布局定位当前 Q head 起始地址。
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    # 一次加载完整 Q 向量，后续所有历史 token 复用，避免重复读取 Q。
    q = tl.load(query_ptr + q_offset)

    # acc 保存 sum(exp(score - m_i) * V)，l_i 保存对应 softmax 分母，m_i 保存运行最大值。
    # 三者都用 FP32，避免 FP16 长序列累加误差和 softmax 下溢。
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    # 按 BLOCK_N 计算静态展开的 chunk 数；用最大容量保证不同请求可复用同一 kernel。
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    # 每次处理一段逻辑 token，逻辑 token 再通过 block table 映射到物理 cache。
    for chunk_idx in range(max_chunks):
        # 当前 chunk 的第一个逻辑 token 下标。
        token_start = chunk_idx * BLOCK_N
        
        # Only process if within valid range
        if token_start < context_len:
            # 生成当前 chunk 的逻辑 token 下标，并屏蔽超过真实上下文长度的 lane。
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len

            # qk 是当前 chunk 的 score 向量；无效 lane 预置为近似 -inf，softmax 权重为 0。
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



# chunked prefill/extend attention 使用的 Triton kernel。
# 普通 full prefill 可以直接在当前 q/k/v 上做 varlen flash attention；
# 但 chunked prefill 的后续 chunk 需要看见“之前 chunk 已经写入 paged KV cache 的历史 K/V”，
# 因此这里按 block_tables 从 KV cache 读取完整可见上下文。
@triton.jit
def paged_attention_prefill_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    cu_seqlens_q_ptr,
    positions_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Triton program id 的切分方式：
    # - axis 0: 全局 query token 下标 q_idx，对应 query 张量的第 q_idx 行。
    # - axis 1: attention head 下标 head_idx。
    # - axis 2: batch 内 sequence 下标 seq_idx。
    # 这种写法是 correctness-first：会为 (q_idx, head_idx, seq_idx) 的笛卡尔积启动 program，
    # 再用 cu_seqlens_q 过滤掉不属于当前 seq 的 q_idx，因此会存在一些空 program。
    # 性能上，这种 grid 可以直接处理长度不规则的 mixed batch，但空 program 会浪费 launch
    # 和调度开销；后续若要优化，应预先构造 q_idx -> seq_idx 映射，把 grid 压缩成有效配对。
    q_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # cu_seqlens_q 记录 mixed batch 中每个 seq 的 query token 范围。
    # 例如 cu_seqlens_q=[0, 512, 513] 表示 seq0 有 512 个 q，seq1 有 1 个 q。
    q_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    q_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    if q_idx < q_start or q_idx >= q_end:
        return

    # GQA/MQA 映射：多个 Q head 共享一个 KV head。
    # 例如 num_heads=16, num_kv_heads=8 时，head 0/1 读 kv_head 0，head 2/3 读 kv_head 1。
    kv_group_size = num_heads // num_kv_heads
    kv_head_idx = head_idx // kv_group_size
    context_len = tl.load(context_lens_ptr + seq_idx)
    token_pos = tl.load(positions_ptr + q_idx)
    # causal 可见长度：当前 query 的绝对位置是 token_pos，只能看见 [0, token_pos]。
    # context_len 是该 seq 当前 KV cache 中已有/将有的上下文长度，取 min 防止越过有效上下文。
    visible_len = tl.minimum(context_len, token_pos + 1)

    # 读取当前 query 向量 Q[q_idx, head_idx, :]。
    # head_dim 是 constexpr，因此 tl.arange(0, head_dim) 展开成一个向量化 load。
    offs_d = tl.arange(0, head_dim)
    q_offset = q_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    # FlashAttention 的 online softmax 状态：
    # - m_i: 已扫描 key token 的 running max，用于数值稳定。
    # - l_i: 已扫描 key token 的 exp(score - m_i) 归一化分母。
    # - acc: 已扫描 key token 的 sum(P * V) 累积值。
    # 每扫描一个 BLOCK_N 的 key/value tile，就用新的局部 max 更新这三个状态。
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -3.4028234663852886e38
    token_start = 0

    # chunk prefill 下，当前 q/k/v 张量只包含“本轮新增 chunk”的 token。
    # 但 attention 需要看见从 0 到 token_pos 的完整历史 KV，所以这里必须按 block_tables
    # 从 paged KV cache 中逐段读取 K/V，而不能直接只用当前 chunk 的 k/v 做 flash attention。
    # 这里的 while 按实际 visible_len 扫描，避免按 max_num_blocks 展开导致 Triton JIT 编译过大。
    while token_start < visible_len:
        # 当前 tile 覆盖的逻辑 token 下标范围：[token_start, token_start + BLOCK_N)。
        # offs_n 是逻辑 token index，不是物理 cache slot。
        offs_n = token_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < visible_len
        # qk 保存当前 tile 内 BLOCK_N 个 key token 对当前 query 的 score。
        # 不可见位置初始化为 -inf，后面 softmax 权重会变成 0。
        qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 3.4028234663852886e38

        # 第一遍扫描当前 tile 的 K：计算 QK score。
        # 当前实现为了简单，逐 token 找到 physical block，再向量化读取 head_dim。
        # 逻辑 token -> block table -> physical KV block 的映射：
        #   block_num = token_idx // block_size
        #   block_offset = token_idx % block_size
        #   physical_block_idx = block_tables[seq_idx, block_num]
        for i in range(BLOCK_N):
            token_idx = token_start + i
            if token_idx < visible_len:
                block_num = token_idx // block_size
                block_offset = token_idx % block_size
                block_table_offset = seq_idx * max_num_blocks + block_num
                physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                # physical_block_idx == -1 表示这个逻辑 block 无效/未分配，保持 score 为 -inf。
                if physical_block_idx != -1:
                    k_offset = (
                        physical_block_idx * block_size * num_kv_heads * head_dim
                        + block_offset * num_kv_heads * head_dim
                        + kv_head_idx * head_dim
                        + offs_d
                    )
                    k_vec = tl.load(k_cache_ptr + k_offset)
                    # score = Q dot K * scale，这里的 scale 已经包含 1/sqrt(head_dim) 以及外部额外 scale。
                    score = tl.sum(q * k_vec) * scale
                    qk = tl.where(tl.arange(0, BLOCK_N) == i, score, qk)

        # 对当前 tile 做局部 online softmax 更新。
        # m_ij 是当前 tile 的最大 score；m_i_new 是历史 max 和当前 tile max 的合并值。
        qk = tl.where(mask_n, qk, -3.4028234663852886e38)
        m_ij = tl.max(qk)
        m_i_new = tl.maximum(m_i, m_ij)
        # alpha 把旧 acc/l_i 从旧 max 标尺 m_i 重新缩放到新 max 标尺 m_i_new。
        alpha = tl.exp(m_i - m_i_new)
        # p 是当前 tile 在新 max 标尺下的未归一化 softmax 权重。
        p = tl.exp(qk - m_i_new)
        acc = acc * alpha
        l_i = l_i * alpha

        # 第二遍扫描当前 tile 的 V：用当前 tile 的 softmax 权重累加 P*V。
        # 这里和 FlashAttention v2 的思想一致：不 materialize 完整 attention matrix，
        # 只保留当前 query 的 online softmax 累积状态。
        for i in range(BLOCK_N):
            token_idx = token_start + i
            if token_idx < visible_len:
                block_num = token_idx // block_size
                block_offset = token_idx % block_size
                block_table_offset = seq_idx * max_num_blocks + block_num
                physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                if physical_block_idx != -1:
                    v_offset = (
                        physical_block_idx * block_size * num_kv_heads * head_dim
                        + block_offset * num_kv_heads * head_dim
                        + kv_head_idx * head_dim
                        + offs_d
                    )
                    v_vec = tl.load(v_cache_ptr + v_offset)
                    # 取出当前 token i 对应的权重 p[i]。p 是长度 BLOCK_N 的向量，
                    # tl.where 生成 one-hot 后 sum，得到标量 weight。
                    weight = tl.sum(tl.where(tl.arange(0, BLOCK_N) == i, p, 0.0))
                    acc = acc + weight * v_vec
                    l_i = l_i + weight

        # 完成本 tile 后，提交新的 running max，并继续扫描下一个 KV tile。
        m_i = m_i_new
        token_start += BLOCK_N

    # online softmax 的最终归一化：acc / l_i = softmax(QK) @ V。
    out = acc / l_i
    output_offset = q_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, out)


# Python wrapper：只覆盖非量化 KV cache 的 chunked prefill。
# FP8/KIVI 的 chunked prefill 需要在 kernel 内反量化，当前仍在 Attention.forward 中显式禁止。
def paged_attention_prefill_triton(
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
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    # Triton 需要连续的 query，避免 kernel 内出现复杂 stride 和非合并访存。
    query = query.contiguous()
    # 输出与 query 同 shape/dtype，kernel 只负责写入，不需要额外转换。
    output = torch.empty_like(query)
    # block table 的第二维决定每条请求最多可访问多少个逻辑 block。
    max_num_blocks = block_tables.shape[1]
    # sequence 数用于 grid 的第三维；实际 query 范围由 cu_seqlens_q 过滤。
    num_seqs = context_lens.shape[0]
    # mixed batch 中所有 query token 已沿第 0 维拼接。
    total_q = query.shape[0]
    # head_dim 较大时减小 tile，降低寄存器和临时 score 的压力。
    BLOCK_N = 64 if head_dim <= 128 else 32

    # 一个 q/head/seq program 负责一组候选配对，kernel 内会过滤无关 sequence。
    grid = (total_q, num_heads, num_seqs)
    paged_attention_prefill_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        positions,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )
    return output



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
    # 第 0 维是 decode batch，每个请求只产生一个新 query token。
    batch_size = query.shape[0]
    # block table 第二维是逻辑 block 容量，用于计算 kernel 最大扫描范围。
    max_num_blocks = block_tables.shape[1]

    # 连续布局让每个 program 可以一次向量化读取完整 Q head。
    query = query.contiguous()

    # kernel 直接写入最终输出，避免 Python 侧拼接中间结果。
    output = torch.empty_like(query)

    # head_dim <= 128 时使用更大的 token tile，通常能提高访存吞吐；否则降低压力。
    BLOCK_N = 64 if head_dim <= 128 else 32

    # 每个 program 对应一个请求和一个 Q head。
    grid = (batch_size, num_heads)

    # 异步发射 kernel；后续只有在真正读取 output 时才需要同步。
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
    """decode 阶段读取 FP8 分页 KV cache，并融合反量化与 attention。

    一个 program 负责 ``(batch_idx, head_idx)``，沿历史 token 扫描一条请求的
    全部 K/V。逻辑 token 通过 block table 映射到物理 block，因此请求可以动态增长，
    不同请求也可以共享物理 cache，而无需搬移整个序列。

    K/V cache 中保存 FP8 数值，读取后在寄存器中乘 scale 还原；这样不需要先把
    整个 cache 反量化成 FP16 Tensor，避免一次额外的全局写入。FP8 数据为 1 byte
    而 FP16 为 2 byte，长上下文 decode 时可降低显存带宽和 cache 容量。

    per-tensor 只读取一个 K scale 和一个 V scale，访存最少；per-token-head 为
    每个 token/head 保存独立 scale，动态范围更好，但会增加 scale 读取和地址计算。
    两种模式都使用 online softmax 的 ``m_i/l_i/acc``，不保存完整 attention 矩阵。

    性能敏感点是 BLOCK_N、num_warps、scale 读取和 block table 间接访存；BLOCK_N
    太大可能增加寄存器压力，太小则会增加循环控制开销。Grid: ``(batch, num_heads)``。
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


@triton.jit
def paged_attention_decode_quantized_tile_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
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
    """Tile 化 per-token-head FP8 decode。

    一个 program 处理一个 ``(batch, Q head)``，每轮以 ``[BLOCK_N, head_dim]``
    读取 K/V，并将 scale、QK dot、online softmax 和 PV 累积放在同一 kernel 中。
    相比逐 token 循环，它一次完成一组连续向量 load，减少 block table 查表次数和
    标量循环控制，通常更适合长上下文 decode。

    量化数据仍保持 1 byte/element，主要收益来自显存带宽；scale 在寄存器中广播
    到 head_dim，避免 materialize 反量化 Tensor。代价是 tile 临时值会占用寄存器，
    ``BLOCK_N`` 和 num_warps 需要按 head_dim、batch、上下文长度实测选择。
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    context_len = tl.load(context_lens_ptr + batch_idx)
    offs_d = tl.arange(0, head_dim)
    offs_n_base = tl.arange(0, BLOCK_N)

    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset).to(tl.float32)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N
        if token_start < context_len:
            token_idxs = token_start + offs_n_base
            valid_tokens = token_idxs < context_len
            block_nums = token_idxs // block_size
            block_offsets = token_idxs % block_size

            block_table_offsets = batch_idx * max_num_blocks + block_nums
            physical_blocks = tl.load(
                block_tables_ptr + block_table_offsets,
                mask=block_nums < max_num_blocks,
                other=-1,
            )
            valid_cache = valid_tokens & (physical_blocks != -1)

            cache_base = (physical_blocks * block_size * num_kv_heads * head_dim +
                          block_offsets * num_kv_heads * head_dim +
                          kv_head_idx * head_dim)
            scale_offsets = (physical_blocks * block_size * num_kv_heads +
                             block_offsets * num_kv_heads + kv_head_idx)

            K_TILE = tl.load(
                k_cache_ptr + cache_base[:, None] + offs_d[None, :],
                mask=valid_cache[:, None],
                other=0.0,
            ).to(tl.float32)
            k_scales = tl.load(k_scale_ptr + scale_offsets, mask=valid_cache, other=0.0)
            qk = tl.sum(K_TILE * q[None, :], axis=1) * k_scales * scale
            qk = tl.where(valid_tokens, qk, -1e10)

            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            p = tl.where(valid_tokens, p, 0.0)

            V_TILE = tl.load(
                v_cache_ptr + cache_base[:, None] + offs_d[None, :],
                mask=valid_cache[:, None],
                other=0.0,
            ).to(tl.float32)
            v_scales = tl.load(v_scale_ptr + scale_offsets, mask=valid_cache, other=0.0)
            weights = p * v_scales

            acc = acc * alpha + tl.sum(V_TILE * weights[:, None], axis=0)
            l_i = l_i * alpha + tl.sum(p)
            m_i = m_i_new

    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output.to(output_ptr.dtype.element_ty))


@triton.jit
def paged_attention_decode_int4_tile_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    packed_head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tile 化 packed INT4 per-token-head decode。

    每轮加载 ``[BLOCK_N, packed_head_dim]`` 的 K/V byte tile，解出高低 nibble，
    再利用
    ``dot(Q, (q - zp) * scale) = (dot(Q, q) - sum(Q) * zp) * scale``
    融合完成 QK 反量化。这样无需构造完整 FP32 K/V tile，既减少显存写入，也把
    zero-point 修正压缩成一次 Q 的维度和乘法。

    INT4 将 KV 数据压缩到 FP16 的约四分之一，长上下文下通常显著降低带宽；代价
    是 unpack、zero-point 修正和 scale 元数据读取。``BLOCK_N`` 太大可能增加
    packed tile 的寄存器占用，太小则降低访存吞吐。
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    context_len = tl.load(context_lens_ptr + batch_idx)
    offs_d = tl.arange(0, head_dim)
    packed_d = offs_d // 2
    is_high = (offs_d & 1) == 1
    offs_n_base = tl.arange(0, BLOCK_N)

    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset).to(tl.float32)
    # 非对称 INT4 的 zero-point 修正只需要 Q 的维度和，无需物化反量化 K。
    q_sum = tl.sum(q)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N
        if token_start < context_len:
            token_idxs = token_start + offs_n_base
            valid_tokens = token_idxs < context_len
            block_nums = token_idxs // block_size
            block_offsets = token_idxs % block_size

            block_table_offsets = batch_idx * max_num_blocks + block_nums
            physical_blocks = tl.load(
                block_tables_ptr + block_table_offsets,
                mask=block_nums < max_num_blocks,
                other=-1,
            )
            valid_cache = valid_tokens & (physical_blocks != -1)

            cache_base = (physical_blocks * block_size * num_kv_heads * packed_head_dim +
                          block_offsets * num_kv_heads * packed_head_dim +
                          kv_head_idx * packed_head_dim)
            scale_offsets = (physical_blocks * block_size * num_kv_heads +
                             block_offsets * num_kv_heads + kv_head_idx)

            k_packed = tl.load(
                k_cache_ptr + cache_base[:, None] + packed_d[None, :],
                mask=valid_cache[:, None],
                other=0,
            ).to(tl.int32)
            k_lo, k_hi = unpack_int4_nibbles(k_packed)
            k_q = tl.where(is_high[None, :], k_hi, k_lo).to(tl.float32)
            k_scale_packed = tl.load(k_scale_ptr + scale_offsets, mask=valid_cache, other=0.0)
            k_s, k_zp = unpack_int4_scale_zp(k_scale_packed)
            # 融合反量化点积：dot(q, (kq-zp)*s) = (dot(q,kq)-sum(q)*zp)*s。
            raw_qk = tl.sum(k_q * q[None, :], axis=1)
            qk = (raw_qk - q_sum * k_zp.to(tl.float32)) * k_s * scale
            qk = tl.where(valid_tokens, qk, -1e10)

            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            p = tl.where(valid_tokens, p, 0.0)

            v_packed = tl.load(
                v_cache_ptr + cache_base[:, None] + packed_d[None, :],
                mask=valid_cache[:, None],
                other=0,
            ).to(tl.int32)
            v_lo, v_hi = unpack_int4_nibbles(v_packed)
            v_q = tl.where(is_high[None, :], v_hi, v_lo).to(tl.float32)
            v_scale_packed = tl.load(v_scale_ptr + scale_offsets, mask=valid_cache, other=0.0)
            v_s, v_zp = unpack_int4_scale_zp(v_scale_packed)
            # 融合 V 反量化与 softmax 权重，避免构造完整 FP32 V_TILE。
            v_weighted = p * v_s
            raw_v_acc = tl.sum(v_q * v_weighted[:, None], axis=0)
            zp_v_acc = tl.sum(v_weighted * v_zp.to(tl.float32))

            acc = acc * alpha + raw_v_acc - zp_v_acc
            l_i = l_i * alpha + tl.sum(p)
            m_i = m_i_new

    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output.to(output_ptr.dtype.element_ty))


def paged_attention_decode_int4_per_token_head_rht(
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
) -> torch.Tensor:
    """INT4 per-token-head 专用 decode：外层 RHT + 内层融合 unpack/dequant attention。"""
    original_dtype = query.dtype
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    packed_head_dim = head_dim // 2

    # 与 vLLM 一致：RHT 由独立变换实现，attention kernel 专注 packed INT4 点积。
    query_rht = single_rht(query.float()).contiguous()
    output_rht = torch.empty(
        batch_size, num_heads, head_dim, dtype=torch.float32, device=query.device
    )
    block_n = 64 if head_dim <= 128 else 32
    grid = (batch_size, num_heads)
    paged_attention_decode_int4_tile_kernel[grid](
        output_rht, query_rht, k_cache, v_cache, k_scale, v_scale,
        block_tables, context_lens,
        scale=scale / head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        packed_head_dim=packed_head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return (single_rht(output_rht, inverse=True) / head_dim).to(original_dtype)


# ===================== INT4 Group-wise decode kernel =====================

@triton.jit
def paged_attention_decode_int4_groupwise_tile_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,         # (num_blocks, block_size, num_kv_heads, n_groups) float32，含 zp steganography
    v_scale_ptr,         # 同上
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    packed_head_dim: tl.constexpr,   # head_dim // 2
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    group_size: tl.constexpr,
    n_groups: tl.constexpr,
):
    """INT4 group-wise 量化 decode kernel（tile 化）。

    每个 program 负责一个 ``(batch, Q head)``，每个 token tile 内按 group 解包、
    反量化 K/V，再进行 QK 和 PV。group-wise scale/zp 能比整头 scale 更准确地
    描述不同维度的范围；同时保留 packed INT4 的带宽收益。

    这里在 kernel 内构造 ``K_TILE`` 和 ``V_TILE``，实现直观且便于验证，但会带来
    ``BLOCK_N * head_dim`` 的寄存器/临时值压力；``n_groups`` 越大，static loop、
    scale 读取和 mask 选择越多。实际调优时应同时扫描 group_size、BLOCK_N 和
    num_warps，不能只看理论压缩率。

    流程:
      1. 读取 packed INT4 tile 并拆成 nibble；
      2. 按 group 应用 ``x = (q - zp) * scale``；
      3. 用 online softmax 累积 QK/PV，避免完整 attention 矩阵。
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    context_len = tl.load(context_lens_ptr + batch_idx)
    offs_d = tl.arange(0, head_dim)
    # 每个 head_dim 位置所属的 group id
    group_ids = offs_d // group_size
    # 每个 head_dim 位置在 packed 数组中的索引，以及是否是高 nibble
    packed_d = offs_d // 2
    is_high = (offs_d & 1) == 1
    offs_n_base = tl.arange(0, BLOCK_N)

    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset).to(tl.float32)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N
        if token_start < context_len:
            token_idxs = token_start + offs_n_base
            valid_tokens = token_idxs < context_len
            block_nums = token_idxs // block_size
            block_offsets = token_idxs % block_size

            block_table_offsets = batch_idx * max_num_blocks + block_nums
            physical_blocks = tl.load(
                block_tables_ptr + block_table_offsets,
                mask=block_nums < max_num_blocks,
                other=-1,
            )
            valid_cache = valid_tokens & (physical_blocks != -1)

            # cache 基地址 (BLOCK_N,): 每个 token 对应的 packed uint8 行基址
            cache_base = (physical_blocks * block_size * num_kv_heads * packed_head_dim +
                          block_offsets * num_kv_heads * packed_head_dim +
                          kv_head_idx * packed_head_dim)
            # scale 基地址 (BLOCK_N,): 每个 token 对应的 n_groups 行基址
            scale_base = (physical_blocks * block_size * num_kv_heads * n_groups +
                          block_offsets * num_kv_heads * n_groups +
                          kv_head_idx * n_groups)

            # ---- 加载 K packed tile: (BLOCK_N, packed_head_dim) ----
            k_packed = tl.load(
                k_cache_ptr + cache_base[:, None] + packed_d[None, :],
                mask=valid_cache[:, None],
                other=0,
            ).to(tl.int32)
            k_lo, k_hi = unpack_int4_nibbles(k_packed)                     # (BLOCK_N, head_dim//2)
            k_nibbles = tl.where(is_high[None, :], k_hi, k_lo).to(tl.float32)  # (BLOCK_N, head_dim)

            # ---- 加载 K scale tile: (BLOCK_N, n_groups)，逐 group dequant ----
            K_TILE = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
            for g in tl.static_range(0, n_groups):
                k_scale_packed_g = tl.load(
                    k_scale_ptr + scale_base + g,
                    mask=valid_cache,
                    other=0.0,
                )
                k_s_g, k_zp_g = unpack_int4_scale_zp(k_scale_packed_g)    # (BLOCK_N,)
                mask_g = group_ids[None, :] == g                            # (1, head_dim) broadcast
                # dequant: x = (nibble - zp) * scale
                dequant_g = (k_nibbles - k_zp_g[:, None].to(tl.float32)) * k_s_g[:, None]
                K_TILE = tl.where(mask_g, dequant_g, K_TILE)

            qk = tl.sum(K_TILE * q[None, :], axis=1) * scale              # (BLOCK_N,)
            qk = tl.where(valid_tokens, qk, -1e10)

            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            p = tl.where(valid_tokens, p, 0.0)

            # ---- 加载 V packed tile 并 dequant ----
            v_packed = tl.load(
                v_cache_ptr + cache_base[:, None] + packed_d[None, :],
                mask=valid_cache[:, None],
                other=0,
            ).to(tl.int32)
            v_lo, v_hi = unpack_int4_nibbles(v_packed)
            v_nibbles = tl.where(is_high[None, :], v_hi, v_lo).to(tl.float32)

            V_TILE = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
            for g in tl.static_range(0, n_groups):
                v_scale_packed_g = tl.load(
                    v_scale_ptr + scale_base + g,
                    mask=valid_cache,
                    other=0.0,
                )
                v_s_g, v_zp_g = unpack_int4_scale_zp(v_scale_packed_g)
                mask_g = group_ids[None, :] == g
                dequant_g = (v_nibbles - v_zp_g[:, None].to(tl.float32)) * v_s_g[:, None]
                V_TILE = tl.where(mask_g, dequant_g, V_TILE)

            acc = acc * alpha + tl.sum(V_TILE * p[:, None], axis=0)
            l_i = l_i * alpha + tl.sum(p)
            m_i = m_i_new

    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output.to(output_ptr.dtype.element_ty))


def paged_attention_decode_int4_groupwise(
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
    group_size: int,
    use_rht: bool = False,
) -> torch.Tensor:
    """INT4 group-wise 量化 decode 的 Python 封装。

    若 use_rht=True，对 Q 做 RHT，softmax_scale 除以 head_size；输出做逆 RHT 并除以 head_size。
    （与 vLLM int4_per_token_head 对齐）
    """
    assert head_dim % group_size == 0
    n_groups = head_dim // group_size
    packed_head_dim = head_dim // 2
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    orig_dtype = query.dtype
    if use_rht:
        # RHT 变换 Q，并补偿 RHT norm（sqrt(head_dim) 的平方 = head_dim）
        query = single_rht(query.float())
        scale = scale / head_dim
    else:
        query = query.float()

    query = query.contiguous()
    output = torch.empty(batch_size, num_heads, head_dim, dtype=torch.float32, device=query.device)

    BLOCK_N = 64 if head_dim <= 128 else 32
    grid = (batch_size, num_heads)
    paged_attention_decode_int4_groupwise_tile_kernel[grid](
        output, query, k_cache, v_cache, k_scale, v_scale,
        block_tables, context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        packed_head_dim=packed_head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
        group_size=group_size,
        n_groups=n_groups,
        num_warps=4,
    )

    if use_rht:
        # 逆 RHT 还原输出，并除以 head_dim（补偿 RHT norm 的平方）
        output = single_rht(output, inverse=True) / head_dim

    return output.to(orig_dtype)


def paged_attention_decode_quantized(
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
    decode 阶段基于量化 paged cache 计算注意力 (含反量化) 的 Python 封装。
    支持 fp8_per_tensor、fp8_per_token_head、int8_per_token_head、int4_per_token_head。
    输出 dtype 与 query 一致 (fp16/bf16)。
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    query = query.contiguous()
    output = torch.empty_like(query)

    BLOCK_N = 64 if head_dim <= 128 else 32
    per_token_head = 1 if kv_cache_dtype in ('fp8_per_token_head', 'int8_per_token_head') else 0
    grid = (batch_size, num_heads)

    if kv_cache_dtype == 'int4_per_token_head' and head_dim <= 128:
        paged_attention_decode_int4_tile_kernel[grid](
            output, query, k_cache, v_cache, k_scale, v_scale,
            block_tables, context_lens,
            scale=scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            packed_head_dim=(head_dim + 1) // 2,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
    elif kv_cache_dtype in ('fp8_per_token_head', 'int8_per_token_head') and head_dim <= 128:
        paged_attention_decode_quantized_tile_kernel[grid](
            output, query, k_cache, v_cache, k_scale, v_scale,
            block_tables, context_lens,
            scale=scale,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
    else:
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
    """返回无符号 ``bits`` 位量化的最大整数值。

    作为 Triton 可内联的小函数使用，避免 Python 侧在不同量化位宽间重复生成
    公式；它本身几乎没有运行时成本，真正的性能开销来自后续 scale reduction
    和 cache 访存。
    """
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
    """KIVI 风格的量化 KV cache 写入 kernel。

    一个 program 处理一个 ``(token, kv_head)``，同时完成三类写入：

    1. 把原始 K/V 写入 residual buffer，保留高精度尾部数据；
    2. K 按 head_dim 分组计算 min/max、scale/zero，并写入无符号量化值；
    3. V 按 token/head 求 amax，使用对称 scale 量化后写入 cache。

    K 使用非对称量化可更好覆盖分布不以零为中心的数据，V 使用对称量化则实现
    更简单。量化数据目前放在 int8 容器而不是 bit-pack：这样 decode 地址和普通
    cache 同构，减少索引复杂度和 kernel 分支，但没有获得真正 2/4-bit 的全部
    显存收益。若进一步 bit-pack，带宽会下降，但需要额外 unpack 和更复杂的边界处理。

    residual 与量化 cache 在同一个 program 写入，减少一次独立 kernel launch；代价
    是每个 token/head 需要同时写三套数据，写带宽和寄存器压力较大。当前 K 的 scale
    按 token/group 标定，避免跨 token 同步，适合分页 cache，但比论文中整块/整段
    per-channel 标定的量化精度略弱。
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
    """KIVI 量化分页 cache 的 decode attention。

    Grid ``(batch, num_heads)``，每个 program 扫描一条请求的历史 token。K 读取后
    按 group 应用 ``x = q * scale + zero``，V 使用 token/head scale 对称反量化，
    再通过 online softmax 完成注意力。与普通 paged decode 结构一致，方便复用
    block table 和调度逻辑。

    KIVI 的收益来自较低的 cache 数据精度和较小的带宽；代价是每个 token/head 要
    额外读取多个 group 的 scale/zero，并在当前实现中逐 token、逐 group 反量化。
    这会增加控制流和寄存器压力，长上下文的带宽收益可能被反量化算术部分抵消。
    当前 residual buffer 只在写入阶段维护，本 kernel 仍扫描完整量化 paged cache，
    因此不要把 residual 分配误认为已经获得尾部高精度 decode 或相应性能收益。
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
        self.kv_cache_dtype = 'auto'   # "auto" | "fp8_per_tensor" | "fp8_per_token_head" | "int8_per_token_head" | "int4_per_token_head" | "kivi_2bit" | "kivi_4bit"
        self.k_scale = None            # FP8/KIVI 量化时的 K scale 张量
        self.v_scale = None            # FP8/KIVI 量化时的 V scale 张量
        # KIVI 专用
        self.k_zero = None             # K 的 zero-point (KIVI 非对称量化)
        self.k_residual = None         # K 的 fp16 residual buffer
        self.v_residual = None         # V 的 fp16 residual buffer
        self.kivi_bits = 0
        self.kivi_group_size = 0
        self.kivi_residual_length = 0
        # INT4 group-wise 专用参数，由 ModelRunner.allocate_kv_cache 注入。
        self.int4_group_size = 32
        self.int4_use_rht = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 路径选择: auto / fp8_* / int8_per_token_head / int4_per_token_head / int4_groupwise / kivi_*
        kivi_enabled = isinstance(self.kv_cache_dtype, str) and self.kv_cache_dtype.startswith('kivi')
        fp8_enabled = self.kv_cache_dtype in ('fp8_per_tensor', 'fp8_per_token_head')
        int8_enabled = self.kv_cache_dtype == 'int8_per_token_head'
        int4_enabled = self.kv_cache_dtype == 'int4_per_token_head'
        int4_groupwise_enabled = self.kv_cache_dtype == 'int4_groupwise'
        kv_quant_enabled = fp8_enabled or int8_enabled or int4_enabled or int4_groupwise_enabled

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
            elif int4_groupwise_enabled:
                # INT4 group-wise 量化写入：per-(token,head,group) scale/zp + nibble pack
                store_kvcache_int4_groupwise(
                    k_to_store, v_to_store, k_cache, v_cache,
                    self.k_scale, self.v_scale,
                    context.slot_mapping, self.block_size,
                    group_size=self.int4_group_size,
                    use_rht=self.int4_use_rht,
                )
            elif int4_enabled:
                # int4_per_token_head 使用 vLLM 的 RHT 路径。
                # group_size=head_dim 表示每个 token/head 只有一组，scale shape 与原模式兼容。
                store_kvcache_int4_groupwise(
                    k_to_store, v_to_store, k_cache, v_cache,
                    self.k_scale.unsqueeze(-1), self.v_scale.unsqueeze(-1),
                    context.slot_mapping, self.block_size,
                    group_size=self.head_dim,
                    use_rht=True,
                )
            elif int8_enabled:
                # INT8 per-token-head 量化写入：动态 scale + round + clamp
                store_kvcache_int8(
                    k_to_store, v_to_store, k_cache, v_cache,
                    self.k_scale, self.v_scale,
                    context.slot_mapping, self.block_size,
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
            # is_prefill 只是 batch 级标记：表示本轮需要 eager prefill/mixed 路径，不能说明所有 seq 都是 full prefill。
            # 只有 prepare_prefill/prepare_mixed 明确标记所有 seq 都是 pure full prefill 时，才走连续 q/k/v 的 flash prefill。
            # 只要 batch 中存在 chunked prefill、extend 或 mixed decode，就必须走 paged prefill 从 KV cache 读取历史。
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            if not context.is_full_prefill:
                # 量化 KV 的 chunked prefill 需要在 paged prefill kernel 内做反量化，当前只实现非量化路径。
                if kv_quant_enabled or kivi_enabled:
                    raise NotImplementedError(
                        "chunked prefill for quantized KV cache needs a "
                        "dequantizing paged prefill/extend attention path"
                    )
                if context.positions is None or context.context_lens is None or context.block_tables is None:
                    raise ValueError("paged prefill requires positions, context_lens, and block_tables")
                split = context.mixed_attention_split
                if split is not None:
                    if split.decode_token_indices and split.prefill_token_indices:
                        # 对齐 vLLM 的思路：scheduler 仍然可以给出 mixed batch，
                        # attention backend 根据 metadata 把 decode 与 prefill/extend 分开执行。
                        # MiniVLLM 不重排 scheduler 输出；这里只临时 index_select，最后 scatter 回原 token 顺序。
                        if (
                            split.decode_token_indices_tensor is None
                            or split.decode_seq_indices_tensor is None
                            or split.prefill_token_indices_tensor is None
                            or split.prefill_seq_indices_tensor is None
                            or split.prefill_cu_seqlens_q_tensor is None
                        ):
                            raise ValueError("mixed attention split tensors must be prepared")
                        o = torch.empty_like(q)
                        decode_token_indices = split.decode_token_indices_tensor
                        decode_seq_indices = split.decode_seq_indices_tensor
                        decode_out = paged_attention_decode(
                            q.index_select(0, decode_token_indices),
                            k_cache,
                            v_cache,
                            context.block_tables.index_select(0, decode_seq_indices),
                            context.context_lens.index_select(0, decode_seq_indices),
                            scale,
                            self.num_heads,
                            self.num_kv_heads,
                            self.head_dim,
                            self.block_size,
                        )
                        o.index_copy_(0, decode_token_indices, decode_out)

                        prefill_token_indices = split.prefill_token_indices_tensor
                        prefill_seq_indices = split.prefill_seq_indices_tensor
                        prefill_cu_seqlens_q = split.prefill_cu_seqlens_q_tensor
                        prefill_out = paged_attention_prefill_triton(
                            q.index_select(0, prefill_token_indices),
                            k_cache,
                            v_cache,
                            context.block_tables.index_select(0, prefill_seq_indices),
                            context.context_lens.index_select(0, prefill_seq_indices),
                            prefill_cu_seqlens_q,
                            context.positions.index_select(0, prefill_token_indices),
                            scale,
                            self.num_heads,
                            self.num_kv_heads,
                            self.head_dim,
                            self.block_size,
                        )
                        o.index_copy_(0, prefill_token_indices, prefill_out)
                        return o.reshape(o.shape[0], self.num_heads * self.head_dim)

                o = paged_attention_prefill_triton(
                    q, k_cache, v_cache,
                    context.block_tables, context.context_lens,
                    cu_seqlens, context.positions,
                    scale, self.num_heads, self.num_kv_heads,
                    self.head_dim, self.block_size,
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
            elif int4_groupwise_enabled:
                o = paged_attention_decode_int4_groupwise(
                    q, k_cache, v_cache,
                    self.k_scale, self.v_scale,
                    context.block_tables, context.context_lens,
                    scale, self.num_heads, self.num_kv_heads, self.head_dim,
                    self.block_size, self.int4_group_size, self.int4_use_rht,
                )
            elif int4_enabled:
                # 专用 fast path：RHT 保持在 wrapper，INT4 unpack/dequant 与 attention 融合。
                o = paged_attention_decode_int4_per_token_head_rht(
                    q, k_cache, v_cache,
                    self.k_scale, self.v_scale,
                    context.block_tables, context.context_lens,
                    scale, self.num_heads, self.num_kv_heads, self.head_dim,
                    self.block_size,
                )
            elif kv_quant_enabled:
                # decode 阶段从量化 cache 读取并反量化
                o = paged_attention_decode_quantized(
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