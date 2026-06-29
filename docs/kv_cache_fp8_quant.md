# KV Cache FP8 量化设计文档

> 适用项目：MinivLLM（`/home/disk1/guowei/code_test/infra/MinivLLM-main`）
> 主题：为 paged KV cache 增加 FP8 量化（per_tensor / per_token_head），并分析其与 paged attention 的兼容性。

---

## 一、量化与 Paged Attention 是否冲突？

**结论：不冲突。** 两者作用在正交的维度上，可以叠加共存。

### 1.1 两者各自管什么

| 机制 | 管理对象 | 关键数据结构 |
|------|----------|--------------|
| Paged Attention | 内存"怎么寻址"——把连续逻辑序列拆成固定大小的物理块，按需分配/复用 | `slot_mapping`、`block_table`、`physical_block_idx` |
| FP8 量化 | 每个槽位"存什么精度"——把 fp16/bf16 压成 1 字节 fp8 + 一个反量化 scale | `k_cache`(fp8 dtype)、`k_scale`/`v_scale` |

Paging 关心"第 N 个 token 落在哪个物理块的哪个偏移"，量化关心"这个偏移上的数值如何压缩与还原"。前者决定地址，后者决定地址里的内容编码，互不干扰。

### 1.2 为什么能天然对齐

关键设计：**scale 张量与 cache 张量共享完全相同的分页布局**。

- cache data 形状：`(num_blocks, block_size, num_kv_heads, head_dim)`
- per_token_head scale 形状：`(num_blocks, block_size, num_kv_heads)`（仅去掉最后的 head_dim 维）

两者的前三维 `(num_blocks, block_size, num_kv_heads)` 完全一致。这意味着：

- **写入**：`store_kvcache_fp8_kernel` 用同一个 `slot_idx -> (block_idx, block_offset)` 计算 data 偏移和 scale 偏移，数据和 scale 总是写到对应位置。
- **读取**：`paged_attention_decode_fp8_kernel` 通过 `block_table` 查到 `physical_block_idx` 后，用同一物理块号同时索引 data 和 scale，反量化必然取到匹配的 scale。
- **块复用 / 前缀缓存**：`BlockManager` 以"块"为粒度分配、释放、共享（见 `block_manager.py` 的 `allocate`/`deallocate`/`append`）。当一个物理块被复用或被前缀缓存命中时，块内的 fp8 数据和对应 scale 是同一块物理内存里的相邻区域，一起搬移、一起命中，不存在 data 与 scale 错位的风险。

### 1.3 潜在的注意点（非冲突，但需留意）

1. **CUDA Graph**：原 decode 路径用 CUDA graph 加速。量化路径新增了 scale 张量作为 kernel 输入，当前实现走 `enforce_eager=True`（不捕获 graph）。若后续要为量化路径捕获 graph，需要把 scale buffer 也纳入静态预分配（与 `k_cache`/`v_cache` 一样属于 graph 外部固定输入），逻辑上可行。
2. **prefix caching 的 hash**：`compute_hash` 基于 `token_ids` 计算，与 cache 内容的精度无关。所以无论是否量化，前缀缓存命中判定都不受影响——命中后直接复用已量化好的块即可。
3. **per_tensor 的 scale 语义**：per_tensor 用"整层一个标量 scale"，本实现在每次 store 时用当前批 K/V 的 amax 重新标定并原地更新该标量。这是为了实现简单；其代价是不同批写入会刷新 scale，长序列下早写入的块用的是旧 scale。per_token_head 没有这个问题（每个槽位自带 scale），精度更稳，这也是 vLLM 默认推荐 per_token_head 的原因。

---

## 二、修改方案（Plan）

参考 vLLM 的 `KVQuantMode`，支持两种 FP8 模式：

- `fp8_per_tensor`：整层 K/V 各一个标量 scale
- `fp8_per_token_head`：每个 `(token, kv_head)` 独立 scale（对齐 vLLM `FP8_PER_TOKEN_HEAD`）

### 2.1 配置层

引擎 config 新增字段 `kv_cache_dtype`：
- `"auto"`（默认）：不量化，行为与改动前完全一致（向后兼容）
- `"fp8_per_tensor"` / `"fp8_per_token_head"`：启用对应量化

### 2.2 KV Cache 分配（`model_runner.py::allocate_kv_cache`）

- fp8 模式下 cache dtype 改为 `torch.float8_e4m3fn`（1 字节/元素）
- `block_bytes` 用 fp8 itemsize 计算；per_token_head 额外计入 scale 开销 `block_size * 2 * num_layers * num_kv_heads * 4`
- 分配 scale 张量：
  - per_tensor：`(2, num_layers, 1)`
  - per_token_head：`(2, num_layers, num_blocks, block_size, num_kv_heads)`
- 逐层把 `k_cache`/`v_cache`/`k_scale`/`v_scale`/`kv_cache_dtype` 注入每个 `Attention` 模块

### 2.3 量化/反量化 Kernel（`attention.py`）

- 新增 `store_kvcache_fp8_kernel` + `store_kvcache_fp8`：量化写入
- 新增 `paged_attention_decode_fp8_kernel` + `paged_attention_decode_fp8`：读 fp8、反量化、做注意力
- `flash_attention_prefill` 不变（prefill 的 Q/K/V 为新鲜张量，attention 计算本身不量化；只在写 cache 时量化）

### 2.4 调度（`Attention.forward`）

按 `self.kv_cache_dtype` 分支：`auto` 走原路径，fp8 走量化写入 + 反量化 decode。

### 2.5 改动文件清单

- `src/myvllm/engine/model_runner.py`：cache/scale 分配
- `src/myvllm/layers/attention.py`：两个新 kernel + forward 分支
- 模型文件（`llama.py`/`qwen3.py`）：无需改动，scale/dtype 由 model_runner 注入
- 新增 `benchmark_kv_quant.py`：数值微基准 / 端到端 / 显存 / NIAH，表格+图同时输出

---

## 三、量化代码阅读过程与要点

按"分配 → 写入 → 读取 → 调度"的数据流顺序梳理。

### 3.1 入口：cache 与 scale 的分配（`model_runner.py`）

阅读 `allocate_kv_cache`，关注三处新增：

1. **dtype 选择**
   ```python
   self.kv_cache_dtype = self.config.get('kv_cache_dtype', 'auto')
   kv_data_dtype = torch.float8_e4m3fn if self.kv_quant_enabled else self.default_dtype
   ```
   fp8 时每元素 1 字节，是显存收益来源。

2. **block_bytes 计入 scale**
   ```python
   block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * elem_bytes
   if self.kv_cache_dtype == 'fp8_per_token_head':
       block_bytes += self.block_size * 2 * num_layers * num_kv_heads * 4  # float32 scale
   ```
   这一步保证显存预算计算准确，避免分配超出真实容量的块数。

3. **scale 张量布局**：per_token_head 的 scale 形状 `(2, num_layers, max_cached_blocks, block_size, num_kv_heads)`，前三维与 data 对齐——这是 §1.2 兼容性的根基。

### 3.2 写入：量化 store（`attention.py::store_kvcache_fp8_kernel`）

Grid `(num_tokens, num_kv_heads)`，每个 program 处理一个 token 的一个 KV head。阅读重点：

- **slot -> (block, offset) 寻址**与原始 `store_kvcache_kernel` 完全相同，量化只改变写入的"内容"：
  ```python
  block_idx = slot_idx // block_size
  block_offset = slot_idx % block_size
  ```
- **per_token_head**：在 kernel 内对当前 `(token, head)` 的 head_dim 个元素求 amax，算出该位置专属 scale 并写入 scale 张量：
  ```python
  k_amax = tl.max(tl.abs(key))
  k_scale = tl.where(k_amax > 0, k_amax / FP8_MAX, 1.0)
  scale_offset = block_idx*block_size*num_kv_heads + block_offset*num_kv_heads + head_idx
  tl.store(k_scale_ptr + scale_offset, k_scale)
  ```
  注意 `scale_offset` 用的就是 data 偏移去掉 head_dim 后的同构索引。
- **per_tensor**：scale 不在 kernel 内算，而是在 Python 封装 `store_kvcache_fp8` 里用整批 amax 预标定后原地写入标量，kernel 直接 `tl.load(k_scale_ptr)` 读单值。
- **量化公式**：`clamp(x / scale, -448, 448).to(fp8)`，448 是 e4m3fn 上限。

### 3.3 读取：反量化 decode（`attention.py::paged_attention_decode_fp8_kernel`）

与原 `paged_attention_decode_kernel` 的 online-softmax 框架一致，差异只在载入 K/V 后多一步反量化：

```python
k_vec = tl.load(k_cache_ptr + k_offset).to(tl.float32)   # fp8 -> float
if PER_TOKEN_HEAD:
    ks_off = physical_block_idx*block_size*num_kv_heads + block_offset*num_kv_heads + kv_head_idx
    k_s = tl.load(k_scale_ptr + ks_off)
else:
    k_s = k_scale_scalar
k_vec = k_vec * k_s                                       # 反量化
```

注意 `ks_off` 用的是 `physical_block_idx`（来自 `block_table` 查表），与 data 的 `k_offset` 同源——这保证了 paging 复用/重排后 scale 仍对得上（§1.2）。

### 3.4 调度：forward 分支（`attention.py::Attention.forward`）

```python
fp8_enabled = self.kv_cache_dtype != 'auto'
if fp8_enabled:
    store_kvcache_fp8(...)            # 写入量化
else:
    store_kvcache(...)               # 原路径
...
if context.is_prefill:
    o = flash_attention_prefill(...)  # prefill 不量化
else:
    o = paged_attention_decode_fp8(...) if fp8_enabled else paged_attention_decode(...)
```

`auto` 模式下所有调用与改动前逐字节一致，确保不影响既有功能。

### 3.5 阅读过程中验证的关键不变量

- scale 张量与 data 张量前三维布局一致（分配处确认）
- 写入与读取使用同构的 slot/block 寻址（两个 kernel 对比确认）
- prefill 路径未触碰（forward 分支确认）
- prefix cache 的 hash 与精度无关（`block_manager.py::compute_hash` 确认）

---

## 四、FP8 数值约定

- 格式：`torch.float8_e4m3fn`，可表示绝对值上限 **448.0**
- 量化：`x_fp8 = clamp(x / scale, -448, 448)`，其中 `scale = amax / 448`
- 反量化：`x_fp16 = x_fp8.to(float) * scale`
- 环境要求：PyTorch ≥ 2.1，GPU compute capability ≥ 8.9（旧卡可改用 e5m2）
