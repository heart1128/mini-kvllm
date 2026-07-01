# Chunked Prefill 与 Mixed Decode 优先设计

## 目标

本次改造参考 vLLM V1 的调度思想，把 MinivLLM 从“waiting prefill 优先，只有没有 prefill 时才 decode”改为可选的 chunked prefill 策略：优先调度 running 请求中的 decode 或未完成 prefill，再用剩余 token budget 接纳 waiting 请求。这样长 prompt 不再必须一次性占满整轮 prefill，短请求和 decode 请求可以更快获得调度机会。

默认启用：

```python
config = {
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 1024,
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

- `True`：running-first mixed 策略。
  - running 中未完成 prefill 的请求继续消耗一个 chunk。
  - running 中已完成 prefill 的请求调度一个 decode token。
  - waiting 请求用剩余 token budget 做 prefill；长 prompt 可以只调度一段。
- `False`：旧式 prefill-only 策略。
  - waiting prompt 必须整段放入 budget。
  - 如果本轮调度了 prefill，就直接返回，不混入 decode。

返回值从旧的 `list[Sequence]` 扩展为 `list[ScheduledSequence]`，每个 item 带有本轮调度 token 数。

### ModelRunner

`ModelRunner.run()` 保留 decode-only 快路：当本轮全是 decode item 时仍使用 `prepare_decode()` 和 CUDA graph。混合 batch 或 chunked prefill 使用 `prepare_mixed()`，构造：

- `input_ids`：本轮实际要计算的 token。
- `cu_seqlens_q`：每条请求本轮 query 长度累计。
- `cu_seqlens_k`：每条请求本轮结束后的上下文长度累计。
- `slot_mapping`：本轮 token 写入 KV cache 的物理位置。
- `block_tables`：paged KV cache 的 block 映射。
- `positions`：真实 token position，避免后续 chunk 从 0 重新编号。

### Attention

普通整段 prefill 仍走现有 FlashAttention prefill 路径。chunked prefill/extend 场景需要读取之前 chunk 写入的 KV cache；当前实现提供了一个非量化 KV 的 PyTorch fallback，用于保证语义正确。

这条 fallback 不是最终性能实现。它的目的只是让功能闭环，方便先观察调度策略对 TTFT 和公平性的影响。

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

当前实现应重点观察两类变化：

1. 调度公平性：长 prompt 存在时，chunked prefill 是否降低短请求 TTFT。
2. 当前实现成本：由于 chunked prefill 的 extend attention 仍是 PyTorch fallback，总吞吐和 TPOT 可能变差。

因此，如果看到 `enable_chunked_prefill=True` 的短请求 TTFT 更好，但总吞吐下降，这是符合当前实现阶段预期的。它说明调度策略生效，但 kernel 还不是最终优化版本。

## 未实现特性与限制

- 没有 Triton/CUDA paged prefill/extend kernel。当前 chunked prefill 对非量化 KV 使用 PyTorch fallback，性能不代表最终优化上限。
- FP8/KIVI quantized KV 的 chunked prefill 被显式阻止。原因是缺少反量化 paged prefill/extend attention 路径。
- 未实现 vLLM 的 batch reorder，没有把 batch 分成 decode、short extend、long extend、prefill 四段。
- 未实现 `max_num_partial_prefills`、`max_long_partial_prefills`、`long_prefill_token_threshold` 等细粒度限流参数。
- 未实现 skipped waiting queue。当前 waiting 队首如果因资源不足无法调度，后续 waiting 请求不会被跳过尝试。
- 未实现 priority scheduling，只有简化的 FCFS 行为。
- 未覆盖 LoRA、encoder input、spec decode、remote KV loading 等 vLLM 生产级调度路径。
- mixed batch 不复用 CUDA graph。decode-only 仍保留原 CUDA graph 快路。
- benchmark 目前输出表格和 JSON，没有自动生成图表。后续可以补 matplotlib，把 TTFT/TPOT/吞吐放在同一张对比图中。

## 验证命令

```bash
pytest tests/test_scheduler.py -v
python -m compileall src/myvllm tests/test_scheduler.py benchmark_chunk_prefill.py
```

有 CUDA 和模型权重环境时再运行 benchmark 脚本。
