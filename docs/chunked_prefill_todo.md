# Chunked Prefill TODO

## Scheduler 策略对齐 vLLM

- [x] 增加 `long_prefill_token_threshold` / prefill chunk size 阈值。
  - 目标：限制单个长 prefill request 在一个 scheduler step 中最多消耗多少 token budget。
  - 已实现行为：`num_new_tokens = min(remaining_prompt_tokens, long_prefill_token_threshold, remaining_budget)`；阈值为 `0` 或未配置时保持现有行为。

- [x] 增加 `max_num_partial_prefills`。
  - 目标：限制同时处于 partial/chunked prefill 状态的请求数量。
  - 已实现行为：当 running 中 inflight partial prefill 数达到上限时，waiting 中新的 partial prefill 请求暂不 admission。

- [x] 增加 `max_long_partial_prefills`。
  - 目标：限制同时处于 partial prefill 状态的长请求数量。
  - 已实现行为：使用 `long_prefill_token_threshold` 判定 long prompt；达到上限时，新的 long partial prefill 不进入 running，短 full prefill 不受该限制阻塞。

- [x] 对齐并明确 running 队列调度顺序。
  - 已实现行为：chunked prefill scheduler 和 vLLM V1 一样 running-first，running 内部按队列顺序扫描，不额外强制 decode-ready 优先。
  - 说明：每个 running seq 根据 `num_computed_tokens` 判断本轮是继续 prefill chunk，还是 decode 1 个 token；长 prefill 的影响主要通过 `long_prefill_token_threshold` 和 partial prefill 限制控制。

## Correctness / Attention metadata

- [ ] 实现量化 KV cache 的 chunked prefill / paged prefill attention。
  - vLLM per-tensor FP8 做法：cache 写入时按 layer 级 `_k_scale/_v_scale` 量化 K/V；paged attention 读取 cache 后用同一组 scale 反量化，或把 scale 融入 score/value 聚合。
  - vLLM per-token-head FP8 做法：写 KV cache 时为每个 `(block, slot, kv_head)` 动态计算 K/V scale，并把 scale 存在 KV cache head 维 padding 区；unified attention kernel 按 physical block/slot/head 读取 scale。
  - vLLM kernel 融合方式：K scale 乘到 `QK` score 上，V scale 乘到 softmax probability 后再聚合 V，避免显式 materialize 反量化后的完整 K/V。
  - 当前 MinivLLM 状态：`Attention.forward()` 对 `fp8_per_tensor`、`fp8_per_token_head`、KIVI 的 chunked prefill 直接 `NotImplementedError`，只支持非量化 KV 的 `paged_attention_prefill_triton()`。
  - 预期实现：先补 FP8 per-tensor paged prefill dequant path，再补 FP8 per-token-head scale cache 和 kernel 融合；KIVI 可单独列后续任务。
  - 需要补测试：量化 KV 下 chunked prefill logits 与非 chunked/full prefill 或非量化参考路径做误差对齐；覆盖 per-tensor 和 per-token-head 两种 scale 读取路径。
  - 需要补 benchmark：长 prompt/mixed prompt 下对比 FP16 KV、FP8 per-tensor KV、FP8 per-token-head KV 的 TTFT、TPOT、显存和吞吐。

- [x] 修正 mixed batch 下 attention path 的判定逻辑，下沉到 seq/token 级 metadata。
  - 已实现行为：`is_prefill` 仍是 batch 级标记，只决定不用纯 decode CUDA graph；新增 `Context.is_full_prefill` 表示本轮所有 seq 是否都是 pure full prefill。
  - pure full prefill 判定：每个 seq 的 query 从 position 0 开始，且本轮 q_len 覆盖完整 context；decode、extend、chunked prefill 都会让 `is_full_prefill=False`。
  - attention path：`Attention.forward()` 不再读取 `context.positions[0]` 判断路径；只有 `context.is_full_prefill=True` 才走 `flash_attention_prefill()`，否则走 `paged_attention_prefill_triton()`。
  - 已补测试：source 级测试验证 chunked path 使用 Triton、不依赖第一个 position，并验证 `ModelRunner` 写入 `is_full_prefill` metadata。

## Benchmark 和验证

- [ ] 扩展 mixed benchmark，对比新增调度策略前后的 `short TTFT(ms)`、`long TTFT(ms)`、`prefill steps`、`decode steps`。
- [ ] 增加 scheduler 单测覆盖：
  - long prefill chunk 不超过阈值。
  - `max_num_partial_prefills` 达到上限时不 admission 新 partial prefill。
  - `max_long_partial_prefills` 达到上限时不 admission 新 long partial prefill。
  - decode-ready 与 prefill-running 的相对调度顺序符合最终选择的 vLLM 对齐策略。
- [ ] 更新 `docs/chunked_prefill_mixed_decode.md`，说明新增配置、默认值、与 vLLM 的差异。
