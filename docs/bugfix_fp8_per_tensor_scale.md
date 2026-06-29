# Bug 记录：FP8 per_tensor KV cache 输出乱码

> 日期：2026-06-29
> 模式：`kv_cache_dtype='fp8_per_tensor'`
> 现象：Qwen3-0.6B 推理输出乱码（如 `2222...` / `prime prime prime...`），`auto` 与 `fp8_per_token_head` 正常。

---

## 一、现象

同样的 prompt，`auto`（fp16）输出正常中文/英文回答，但 `fp8_per_tensor` 输出退化为重复字符：

```
Prompt: 1+3等于多少
auto:           嗯，用户问的是"1+3等于多少"... 答案应该是4 ...   (正常)
fp8_per_tensor: <think>ๆementiaementia 1 12 2 2222 ...           (乱码)
```

`fp8_per_token_head` 不受影响，输出正常。

---

## 二、根因

**per_tensor 的全局 scale 在每次写入时被刷新，但 paged cache 中历史 token 是用旧 scale 量化的，导致反量化时比例错配。**

错误代码（`attention.py::store_kvcache_fp8`）：

```python
if per_token_head == 0:
    k_amax = key.abs().amax().clamp(min=1e-8)
    v_amax = value.abs().amax().clamp(min=1e-8)
    k_scale.fill_((k_amax / FP8_E4M3_MAX).item())   # 每次调用都覆盖
    v_scale.fill_((v_amax / FP8_E4M3_MAX).item())
```

数据流分析：

1. per_tensor 整层只有**一个标量** scale，K/V cache 里所有 token 共用它。
2. 但 `store_kvcache_fp8` 每个 decode step 都被调用一次，每次都用**当前批**的 amax 重新覆盖该标量。
3. paged cache 里的历史 token 是用**写入时刻的 scale** 量化存进去的（fp8 值 = 原值 / scale_old）。
4. decode 反量化时统一用**最新 scale**：`x ≈ fp8_val * scale_new`。
5. 由于 `scale_new != scale_old`，历史 token 被按错误比例放大/缩小 → K/V 数值整体失真 → attention 分布崩坏 → 输出乱码。

**为什么 per_token_head 没事**：它每个 `(token, kv_head)` 槽位都存了自己的 scale，写入和读取永远用同一个，不存在全局刷新问题。这也是 vLLM 默认推荐 per_token_head 的根本原因。

---

## 三、修复

per_tensor 的 scale 改为**首次标定后固定**，不再随每次写入刷新。

```python
if per_token_head == 0:
    # 约定: scale 初值为 1.0 (allocate_kv_cache 用 torch.ones 初始化)，
    #       一旦被标定过(!=1.0)就跳过，保持固定 -> 历史与新 token 共享同一 scale。
    if float(k_scale.reshape(-1)[0].item()) == 1.0:
        k_amax = key.abs().amax().clamp(min=1e-8)
        k_scale.fill_((k_amax / FP8_E4M3_MAX).item())
    if float(v_scale.reshape(-1)[0].item()) == 1.0:
        v_amax = value.abs().amax().clamp(min=1e-8)
        v_scale.fill_((v_amax / FP8_E4M3_MAX).item())
```

关键不变量：**写入与读取必须使用同一个 scale**。per_tensor 通过"标定一次后冻结"来保证；per_token_head 通过"每槽位自带 scale"来保证。

---

## 四、修复的代价与权衡

- 首批（通常是 prefill）数据的 amax 决定整层 scale。若后续 token 幅值显著超过首批，会触发 fp8 截断（clamp 到 ±448），带来额外误差。
- 对正常推理足够稳定，可正确产出文本；但精度天花板低于 per_token_head。
- **结论**：per_tensor 适合显存极度敏感、对精度不苛刻的场景；追求精度优先选 `fp8_per_token_head`。

---

## 五、排查与验证要点

- 对照实验：`auto` 正常、`per_token_head` 正常，仅 `per_tensor` 乱码 → 锁定 per_tensor 专属逻辑（即全局 scale 的处理）。
- 验证修复：用 `kv_cache_dtype='fp8_per_tensor'` 跑 `main.py`，输出应恢复为连贯文本。
- 关联设计文档：见 [kv_cache_fp8_quant.md](kv_cache_fp8_quant.md) §1.3，该隐患在设计阶段已被记录，本次为其在实现中的实际暴露与修复。

---

## 六、附带修复（同次发现）

1. **`greedy sampling is not permitted`**：项目 `SamplingParams` 断言 `temperature > 1e-10`，benchmark 中的 `temperature=0.0` 改为 `1e-6`（近似贪心）。
2. **atexit `'LLMEngine' object has no attribute 'model_runner'`**：benchmark 显式 `exit()` 后进程退出再触发 atexit，`model_runner` 已删。`LLMEngine.exit()` 加幂等保护（已退出直接返回）。仅为退出噪音，不影响结果。
