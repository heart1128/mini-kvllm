# Chunked Prefill 与 Mixed Decode 优先设计

## 目标

本次改造参考 vLLM V1 的调度思想，把 MinivLLM 从“waiting prefill 优先，只有没有 prefill 时才 decode”改为可选的 chunked prefill 策略：优先调度 running 请求中的 decode 或未完成 prefill，再用剩余 token budget 接纳 waiting 请求。这样长 prompt 不再必须一次性占满整轮 prefill，短请求和 decode 请求可以更快获得调度机会。

默认启用：

```python
config = {
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 1024,
    # 0 表示不额外限制长 prefill chunk，保持只受 token budget 限制的旧行为。
    "long_prefill_token_threshold": 0,
    # None 表示不限制同时处于 partial/chunked prefill 的请求数。
    "max_num_partial_prefills": None,
    # None 表示不限制同时处于 long partial prefill 的请求数。
    "max_long_partial_prefills": None,
}
```

关闭后会恢复旧策略：waiting prompt 只有完整放入 `max_num_batched_tokens` 时才会调度；只要调度了 waiting prefill，本轮就不会混入 decode。

## 调度数据流

```text
Scheduler
  -> ScheduledSequence(seq, num_scheduled_tokens)
  -> ModelRunner.prepare_mixed()
  -> Context(cu_seqlens_q, cu_seqlens_k, slot_mapping, block_tables, positions)
  -> Attention
  -> Sampler
  -> Scheduler.postprocess()
```

核心状态是 `Sequence.num_computed_tokens`。它表示该请求已经完成模型计算的 token 数。scheduler 每轮只计算 `seq.num_tokens - seq.num_computed_tokens` 中当前 budget 能覆盖的部分。

## 实现概要

### Scheduler

`Scheduler.schedule()` 根据 `enable_chunked_prefill` 分发：

- `True`：running-first mixed 策略，running 内部和 vLLM V1 一样按队列顺序扫描。
  - running 中每个请求根据 `num_computed_tokens` 判断本轮行为：未完成 prompt 时继续调度一个 prefill chunk；已完成 prompt 时调度 1 个 decode token。
  - waiting 请求用剩余 token budget 做 admission；长 prompt 可以只调度一段，进入 running 后后续 chunk 继续推进。
- `False`：旧式 prefill-only 策略。
  - waiting prompt 必须整段放入 budget。
  - 如果本轮调度了 prefill，就直接返回，不混入 decode。

返回值从旧的 `list[Sequence]` 扩展为 `list[ScheduledSequence]`，每个 item 带有本轮调度 token 数。

新增 scheduler 策略参数：

- `long_prefill_token_threshold`：长 prefill 单个 step 的 chunk size 上限。`0` 或未配置时关闭该额外限制。
- `max_num_partial_prefills`：限制 running 中同时处于 partial/chunked prefill 状态的请求数量；达到上限时，waiting 中新的 partial prefill 暂不 admission。
- `max_long_partial_prefills`：限制 running 中同时处于 long partial prefill 状态的请求数量；long 的判定使用 `long_prefill_token_threshold`，短 full prefill 不受该限制阻塞。

chunked prefill 的本轮 token 数计算为：

```python
num_new_tokens = min(
    remaining_prompt_tokens,
    remaining_budget,
    long_prefill_token_threshold,  # 仅 long prompt 且阈值 > 0 时参与
)
```

### ModelRunner

`ModelRunner.run()` 保留 decode-only 快路：当本轮全是 decode item 时仍使用 `prepare_decode()` 和 CUDA graph。混合 batch 或 chunked prefill 使用 `prepare_mixed()`，构造：

- `input_ids`：本轮实际要计算的 token。
- `cu_seqlens_q`：每条请求本轮 query 长度累计。
- `cu_seqlens_k`：每条请求本轮结束后的上下文长度累计。
- `slot_mapping`：本轮 token 写入 KV cache 的物理位置。
- `block_tables`：paged KV cache 的 block 映射。
- `positions`：真实 token position，避免后续 chunk 从 0 重新编号。

### Attention

普通整段 prefill 仍走现有 FlashAttention prefill 路径。chunked prefill/extend 场景需要读取之前 chunk 写入的 KV cache；当前非量化 KV 已切到 `paged_attention_prefill_triton()`。

mixed batch 中 decode 会被当作 `q_len=1` 的 extend item 处理，chunked prefill 则是 `q_len>1` 的 extend item。两者都通过 `block_tables` 读取 paged KV cache 中的历史 K/V，并用 `positions` / `context_lens` 保证 causal 可见范围正确。

当前 attention path 已使用 `Context.is_full_prefill` 做路径选择：`is_prefill` 只是 batch 级标记，决定本轮不用纯 decode CUDA graph；`is_full_prefill` 表示所有 seq 都是从 position 0 开始且 q_len 覆盖完整 context 的 pure full prefill。只有 `is_full_prefill=True` 时才走 `flash_attention_prefill()`；只要存在 decode、extend 或 chunked prefill，就走 paged prefill/extend attention。

## 性能评测

新增脚本：

```bash
python benchmark_chunk_prefill.py \
  --model Qwen/Qwen3-0.6B \
  --max-output-tokens 64 \
  --max-num-batched-tokens 1024
```

脚本会分别运行：

- `enable_chunked_prefill=False`
- `enable_chunked_prefill=True`

默认场景：

- `short_only`：短 prompt 批量，观察常规请求是否被影响。
- `long_only`：长 prompt，观察长请求下 TTFT 和总时延变化。
- `mixed_long_short`：长 prompt 与短 prompt 混合，观察短请求在长 prefill 下的 TTFT。

输出指标：

- `TTFT(ms)`：请求加入后到首 token 产出的时间。
- `TPOT(ms/token)`：首 token 后平均每输出 token 时间。
- `total_latency(s)`：所有请求完成总耗时。
- `decode_tps(tok/s)`：输出 token 吞吐。
- `short_request_ttft_under_long_prefill(ms)`：混合长短请求时，短请求平均 TTFT。

结果会写入：

```text
results/chunk_prefill_benchmark.json
```

## 结果解读

当前实现应重点观察几类变化：

1. 调度公平性：长 prompt 存在时，chunked prefill 是否降低短请求 TTFT。
2. decode 延迟：running 队列按 vLLM 风格顺序扫描后，长 prefill chunk 是否仍会拖高后续 decode 的 TPOT。
3. chunk 阈值影响：`long_prefill_token_threshold` 越小，单个长 prefill 连续占用的 token budget 越少，但 prefill step 数会增加。
4. partial prefill 限制：`max_num_partial_prefills` / `max_long_partial_prefills` 是否减少多个长 prompt 同时占用 prefill budget。
5. kernel 成本：非量化 KV 已使用 Triton paged prefill kernel，性能比 PyTorch fallback 明显更接近可用路径，但当前 kernel 仍是 correctness-first 实现，不代表最终优化上限。

## 未实现特性与限制

- FP8/KIVI quantized KV 的 chunked prefill 被显式阻止。原因是缺少反量化 paged prefill/extend attention 路径。
- Triton paged prefill kernel 仍是 correctness-first 实现，尚未做更细粒度的 block/query tiling、decode/prefill 融合优化。
- 未实现 vLLM 的 batch reorder，没有把 batch 分成 decode、short extend、long extend、prefill 四段。
- 未实现 skipped waiting queue。当前 waiting 队首如果因资源不足或 partial prefill limit 无法调度，后续 waiting 请求不会被跳过尝试。
- 未实现 priority scheduling，只有简化的 FCFS 行为；running 内部按 vLLM V1 风格顺序扫描，不额外强制 decode-ready 优先。
- 未覆盖 LoRA、encoder input、spec decode、remote KV loading 等 vLLM 生产级调度路径。
- mixed batch 不复用 CUDA graph。decode-only 仍保留原 CUDA graph 快路。
- benchmark 目前输出表格和 JSON，没有自动生成图表。后续可以补 matplotlib，把 TTFT/TPOT/吞吐放在同一张对比图中。

## 验证命令

```bash
pytest tests/test_scheduler.py -q
python -m compileall src/myvllm/engine/scheduler.py src/myvllm/engine/llm_engine.py tests/test_scheduler.py
```

有 CUDA 和模型权重环境时再运行 benchmark 脚本。
