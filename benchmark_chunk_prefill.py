import argparse
import contextlib
import gc
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    # 需要 >= 高并发场景下同时在跑的请求数（long + concurrent short），否则部分
    # short 请求会被挡在 waiting 里排队，测不出"高并发下 decode 被打断"的效果。
    "max_num_sequences": 64,
    "max_num_batched_tokens": 1024,
    "max_num_batch_tokens": 4096,
    "long_prefill_token_threshold": 0,
    "max_num_partial_prefills": None,
    "max_long_partial_prefills": None,
    "max_cached_blocks": 2048,
    "block_size": 256,
    "world_size": 1,
    "model_name_or_path": "Qwen/Qwen3-0.6B",
    "enforce_eager": True,
    "vocab_size": 151936,
    "hidden_size": 1024,
    "num_heads": 16,
    "head_dim": 128,
    "num_kv_heads": 8,
    "intermediate_size": 3072,
    "num_layers": 28,
    "tie_word_embeddings": True,
    "base": 1000000,
    "rms_norm_epsilon": 1e-6,
    "qkv_bias": False,
    "scale": 1,
    "max_position": 32768,
    "ffn_bias": False,
    "max_model_length": 4096,
    "gpu_memory_utilization": 0.9,
    "eos": 151645,
}

# baseline：完全关闭 chunked prefill（enable_chunked_prefill=False）。
BASELINE_POLICY = "baseline"


@contextlib.contextmanager
def nvtx_range(name: str, enabled: bool = True):
    if not enabled:
        yield
        return
    try:
        import torch

        nvtx = torch.cuda.nvtx
        nvtx.range_push(name)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        nvtx.range_pop()


def filter_policies(
    policies: list[tuple[str, bool, dict[str, int | None]]],
    requested: str | None,
) -> list[tuple[str, bool, dict[str, int | None]]]:
    if requested is None or requested.strip() in {"", "all"}:
        return policies
    requested_names = [item.strip() for item in requested.split(",") if item.strip()]
    available = {name: (name, enabled, config) for name, enabled, config in policies}
    unknown = [name for name in requested_names if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown policy/policies: {', '.join(unknown)}. "
            f"Available policies: {', '.join(available)}"
        )
    return [available[name] for name in requested_names]


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_engine(engine: Any | None) -> None:
    import torch

    if engine is not None:
        try:
            engine.exit()
        finally:
            del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def build_prompt(kind: str, target_words: int, request_id: int = 0) -> str:
    # 每个请求都加入唯一内容，避免 prefix cache 把多个请求变成相同前缀命中。
    # 否则 benchmark 测到的是 prefix-cache 效果，而不是 chunked prefill 对高并发长文本的调度收益。
    unique_topics = [
        "database migration safety and rollback checkpoints",
        "mobile client latency and streaming user experience",
        "observability dashboards and incident response workflow",
        "recommendation ranking service and feature freshness",
        "payment risk control and audit trail retention",
        "search indexing pipeline and cache invalidation",
        "multi tenant scheduling fairness and quota isolation",
        "model serving admission control and queue backpressure",
    ]
    topic = unique_topics[request_id % len(unique_topics)]
    seed = {
        "short": (
            f"Request {request_id}: summarize the operational impact of {topic} "
            "in one concise sentence."
        ),
        "long": (
            f"Request {request_id}: explain a production case study about {topic}, "
            "including requirements, system design, prefill, decode, KV cache, batching, "
            "fairness, failure modes, monitoring, capacity planning, and tuning tradeoffs."
        ),
    }[kind]
    words = seed.split()
    repeated = []
    while len(repeated) < target_words:
        repeated.extend(words)
    # 末尾也加入唯一 marker，避免长 prompt 由于重复 seed 造成大段相同后缀。
    suffix = f" unique_request_marker_{kind}_{request_id}"
    return " ".join(repeated[:target_words]) + suffix


def make_chat_prompts(tokenizer: Any, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if getattr(tokenizer, "chat_template", None)
        else prompt
        for prompt in prompts
    ]


def scenario_prompts(
    name: str,
    short_words: int,
    long_words: int,
    num_concurrent_short: int = 2,
    num_concurrent_long: int = 1,
) -> list[tuple[str, str]]:
    if name == "short_only":
        return [(f"short_{i}", build_prompt("short", short_words, i)) for i in range(4)]
    if name == "long_only":
        return [(f"long_{i}", build_prompt("long", long_words, i)) for i in range(num_concurrent_long)]
    if name in ("mixed_long_short", "high_concurrency_long_context"):
        # high_concurrency_long_context 是 chunked prefill 真正体现优势的场景：
        # num_concurrent_short 模拟高并发下同时在线 decode 的短请求（在线服务的常驻负载），
        # num_concurrent_long 模拟持续到达的长文本请求（陆续插入，而非一次性到达），
        # 用来验证"chunked prefill 是否缓解高并发下部分请求 decode token 等待过长"。
        return [(f"long_{i}", build_prompt("long", long_words, i)) for i in range(num_concurrent_long)] + [
            (f"short_{i}", build_prompt("short", short_words, i)) for i in range(num_concurrent_short)
        ]
    raise ValueError(f"Unknown scenario: {name}")


# high_concurrency_long_context / mixed_long_short 场景下，long prompt 不是按固定
# engine.step() 数插入（那样在并发数变化时会漂移：并发越高，同样的 step 数里
# short 请求可能还没轮到 decode），而是等 immediate 批次里的请求都至少产出 1 个
# token（即都已进入 decode 稳定状态）之后才插入，且多个 long 请求按
# LONG_PROMPT_ARRIVAL_GAP_STEPS 错开到达，模拟持续到来的长文本负载而不是一次性突发。
LONG_PROMPT_ARRIVAL_GAP_STEPS = 5


# benchmark 中 non-chunked baseline 不能强行跑超预算长 prompt：
# 关闭 chunk prefill 时，scheduler 要求整个 prompt 一次放入 max_num_batched_tokens。
# 如果 prompt 超预算，这组没有可比结果，只能标记 skipped，否则主循环会无进展。
def get_skip_reason(
    config: dict[str, Any],
    tokenizer: Any,
    named_prompts: list[tuple[str, str]],
) -> str | None:
    if config.get("enable_chunked_prefill", True):
        return None

    max_num_batched_tokens = config["max_num_batched_tokens"]
    rendered_prompts = make_chat_prompts(tokenizer, [prompt for _, prompt in named_prompts])
    prompt_lengths = [
        (name, len(tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]))
        for (name, _), rendered_prompt in zip(named_prompts, rendered_prompts)
    ]
    too_long = [
        (name, length)
        for name, length in prompt_lengths
        if length > max_num_batched_tokens
    ]
    if not too_long:
        return None

    details = ", ".join(f"{name}={length}" for name, length in too_long)
    return (
        "non-chunked prefill requires each prompt to fit in max_num_batched_tokens; "
        f"over-budget prompts: {details}, budget={max_num_batched_tokens}"
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * pct) - 1))
    return ordered[index]


_NAN_STATS = {"mean_ms": float("nan"), "p99_ms": float("nan"), "max_ms": float("nan")}


def skipped_result(reason: str) -> dict[str, Any]:
    return {
        "skipped": True,
        "skip_reason": reason,
        "total_latency_s": float("nan"),
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tps": float("nan"),
        "output_tps": float("nan"),
        "ttft": dict(_NAN_STATS),
        "itl": dict(_NAN_STATS),
        "ttft_short": dict(_NAN_STATS),
        "itl_short": dict(_NAN_STATS),
        "ttft_long": dict(_NAN_STATS),
        "itl_long": dict(_NAN_STATS),
        "per_request": {},
        "steps": [],
    }


def raise_if_no_progress(
    scheduler: Any,
    outputs: list[Any],
    num_processed_tokens: int,
    config: dict[str, Any],
) -> None:
    if outputs or num_processed_tokens > 0 or scheduler.is_finished():
        return

    raise RuntimeError(
        "No progress while running benchmark. This usually means the current "
        "scenario is unschedulable, for example enable_chunked_prefill=False "
        "with a prompt longer than max_num_batched_tokens "
        f"({config.get('max_num_batched_tokens')})."
    )


def run_warmup(config: dict[str, Any], tokenizer: Any, engine: Any) -> None:
    # 用一个极短的 dummy 请求预热：让 Triton kernel（flash_attention_prefill /
    # paged_attention_prefill_triton 等）完成首次 JIT 编译，避免这部分一次性开销
    # 被计入正式测量的第一个请求的 TTFT。跑完后不计入任何计时和统计数据。
    from myvllm.sampling_parameters import SamplingParams

    warmup_prompt = make_chat_prompts(tokenizer, ["hi"])[0]
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=1,
        max_model_length=config["max_model_length"],
    )
    engine.add_prompt(warmup_prompt, sampling_params)
    while not engine.scheduler.is_finished():
        engine.step()
    cuda_sync()


def request_kind(name: str) -> str:
    return "long" if name.startswith("long") else "short"


def run_engine_steps(
    config: dict[str, Any],
    tokenizer: Any,
    named_prompts: list[tuple[str, str]],
    max_output_tokens: int,
    progress: bool = False,
    max_steps: int | None = None,
    warmup: bool = True,
    delay_insert_kind: str | None = None,
    arrival_gap_steps: int = 0,
    nvtx: bool = False,
    nvtx_prefix: str = "benchmark",
) -> dict[str, Any]:
    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.sampling_parameters import SamplingParams

    engine = None
    engine = LLMEngine(config=config)
    if warmup:
        run_warmup(config, tokenizer, engine)
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=max_output_tokens,
        max_model_length=config["max_model_length"],
    )

    rendered_prompts = make_chat_prompts(tokenizer, [prompt for _, prompt in named_prompts])
    prompt_tokens = {
        name: len(tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"])
        for (name, _), rendered_prompt in zip(named_prompts, rendered_prompts)
    }

    # delay_insert_kind 指定的请求类型（例如 "long"）延迟插入；其余请求（例如 short）
    # 在 t=0 就全部加入，让它们先进入 decode 阶段。这样才能观察到 baseline 下 long
    # prefill 插入时是否打断了正在进行的 short decode（ITL 尖峰）。
    immediate = [
        (name, rendered) for (name, _), rendered in zip(named_prompts, rendered_prompts)
        if delay_insert_kind is None or request_kind(name) != delay_insert_kind
    ]
    delayed = [
        (name, rendered) for (name, _), rendered in zip(named_prompts, rendered_prompts)
        if delay_insert_kind is not None and request_kind(name) == delay_insert_kind
    ]

    seq_id_to_name = {}

    def add_prompt(name: str, rendered_prompt: str) -> None:
        engine.add_prompt(rendered_prompt, sampling_params)
        seq_id_to_name[engine.scheduler.waiting[-1].seq_id] = name

    for name, rendered_prompt in immediate:
        add_prompt(name, rendered_prompt)

    per_request = {
        name: {
            "kind": request_kind(name),
            "ttft_s": None,
            "finished_s": None,
            "output_tokens": 0,
            "last_decode_time_s": None,
            # inter-token latency：解码阶段每两个相邻 token 之间的间隔（即 TBT / ITL）。
            "inter_token_times_s": [],
            # 每个输出 token 产出的时间戳（相对 start），用于画时间序列图观察
            # decode 是否被 long prefill 打断（baseline 应出现明显断点，chunked 应更连续）。
            "token_timestamps_s": [],
        }
        for name, _ in named_prompts
    }

    # 计时从第一批请求加入 scheduler 后开始，不包含模型加载和 KV cache 分配。
    # progress/max_steps 是调试卡住场景用的：能区分 scheduler 无进展、Triton JIT 卡住、还是普通慢 step。
    start = time.perf_counter()
    step_records = []
    pending_delayed = list(delayed)
    # immediate 请求是否已全部进入过 decode（拿到过至少 1 个 token）；只有这样才认为
    # "高并发 decode 负载已经建立"，此时插入 long prompt 才有意义，不受并发数变化影响。
    immediate_names = {name for name, _ in immediate}
    immediate_all_decoding = not immediate_names
    next_long_arrival_step: int | None = 0 if pending_delayed else None
    total_latency_s = float("nan")
    try:
        while not engine.scheduler.is_finished() or pending_delayed:
            if not immediate_all_decoding:
                immediate_all_decoding = immediate_names <= {
                    name for name, req in per_request.items() if req["ttft_s"] is not None
                }
                if immediate_all_decoding and pending_delayed:
                    next_long_arrival_step = len(step_records)
            if (
                pending_delayed
                and immediate_all_decoding
                and next_long_arrival_step is not None
                and len(step_records) >= next_long_arrival_step
            ):
                name, rendered_prompt = pending_delayed.pop(0)
                add_prompt(name, rendered_prompt)
                next_long_arrival_step = (
                    len(step_records) + arrival_gap_steps if pending_delayed else None
                )
            if max_steps is not None and len(step_records) >= max_steps:
                raise RuntimeError(
                    f"Benchmark exceeded --max-steps={max_steps}; "
                    f"waiting={len(engine.scheduler.waiting)}, running={len(engine.scheduler.running)}"
                )
            step_idx = len(step_records)
            with nvtx_range(f"{nvtx_prefix}/step={step_idx}", enabled=nvtx):
                step_start = time.perf_counter()
                outputs, num_processed_tokens, is_prefill = engine.step()
                raise_if_no_progress(engine.scheduler, outputs, num_processed_tokens, config)
                cuda_sync()
                step_end = time.perf_counter()
                elapsed = step_end - start
                step_time = step_end - step_start
            if progress:
                print(
                    "step="
                    f"{step_idx} elapsed={elapsed:.3f}s "
                    f"step_time={step_time:.3f}s processed={num_processed_tokens} "
                    f"prefill={is_prefill} waiting={len(engine.scheduler.waiting)} "
                    f"running={len(engine.scheduler.running)} outputs={len(outputs)}",
                    flush=True,
                )
            step_records.append(
                {
                    "elapsed_s": elapsed,
                    "step_time_s": step_time,
                    "num_processed_tokens": num_processed_tokens,
                    "tokens_per_s": num_processed_tokens / step_time if step_time > 0 else float("nan"),
                    "budget_utilization": num_processed_tokens / config["max_num_batched_tokens"],
                    "is_prefill": is_prefill,
                    "waiting": len(engine.scheduler.waiting),
                    "running": len(engine.scheduler.running),
                    "outputs": len(outputs),
                }
            )

            for seq_id, token_ids in outputs:
                name = seq_id_to_name[seq_id]
                per_request[name]["finished_s"] = elapsed
                per_request[name]["output_tokens"] = len(token_ids)

            for item in engine.scheduler.running:
                name = seq_id_to_name.get(item.seq_id)
                if name is None:
                    continue
                req = per_request[name]
                if item.num_completion_tokens > 0 and req["ttft_s"] is None:
                    req["ttft_s"] = elapsed
                    req["token_timestamps_s"].append(elapsed)
                elif item.num_completion_tokens > 1:
                    if req["last_decode_time_s"] is not None:
                        req["inter_token_times_s"].append(elapsed - req["last_decode_time_s"])
                    req["last_decode_time_s"] = elapsed
                    req["token_timestamps_s"].append(elapsed)
                if item.num_completion_tokens > 0:
                    if req["last_decode_time_s"] is None:
                        req["last_decode_time_s"] = elapsed
        total_latency_s = time.perf_counter() - start
    finally:
        cleanup_engine(engine)


    total_input_tokens = sum(prompt_tokens.values())
    total_output_tokens = sum(req["output_tokens"] for req in per_request.values())

    def timing_stats_ms(times_s: list[float]) -> dict[str, float]:
        if not times_s:
            return {
                "mean_ms": float("nan"),
                "p50_ms": float("nan"),
                "p90_ms": float("nan"),
                "p95_ms": float("nan"),
                "p99_ms": float("nan"),
                "max_ms": float("nan"),
            }
        return {
            "mean_ms": statistics.mean(times_s) * 1000,
            "p50_ms": percentile(times_s, 0.50) * 1000,
            "p90_ms": percentile(times_s, 0.90) * 1000,
            "p95_ms": percentile(times_s, 0.95) * 1000,
            "p99_ms": percentile(times_s, 0.99) * 1000,
            "max_ms": max(times_s) * 1000,
        }

    def itl_stats(reqs: list[dict[str, Any]]) -> dict[str, float]:
        return timing_stats_ms([t for req in reqs for t in req["inter_token_times_s"]])

    def ttft_stats(reqs: list[dict[str, Any]]) -> dict[str, float]:
        return timing_stats_ms([req["ttft_s"] for req in reqs if req["ttft_s"] is not None])

    all_reqs = list(per_request.values())
    short_reqs = [req for req in all_reqs if req["kind"] == "short"]
    long_reqs = [req for req in all_reqs if req["kind"] == "long"]

    return {
        "total_latency_s": total_latency_s,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        # total tok/s：(输入+输出) token 数 / 总耗时，衡量系统整体吞吐（工业界常见口径，如 vLLM benchmark_serving）。
        "total_tps": (
            (total_input_tokens + total_output_tokens) / total_latency_s
            if total_latency_s > 0 else float("nan")
        ),
        # output tok/s：仅输出 token 数 / 总耗时，衡量生成吞吐。
        "output_tps": total_output_tokens / total_latency_s if total_latency_s > 0 else float("nan"),
        "ttft": ttft_stats(all_reqs),
        "itl": itl_stats(all_reqs),
        # 按请求类型（short/long）拆分的 TTFT 与 ITL：均值会被不同类型的请求相互掩盖，
        # 分开看才能观察到 chunked prefill 对 short 请求 decode 平滑度/公平性的真实影响。
        "ttft_short": ttft_stats(short_reqs),
        "itl_short": itl_stats(short_reqs),
        "ttft_long": ttft_stats(long_reqs),
        "itl_long": itl_stats(long_reqs),
        "prompt_tokens": prompt_tokens,
        "per_request": per_request,
        "steps": step_records,
    }


def summarize_result(
    scenario: str,
    enabled: bool,
    result: dict[str, Any],
    policy: str,
    repeat: int | None = None,
) -> dict[str, Any]:
    short_itl = result["itl_short"]
    short_itl_p50 = short_itl.get("p50_ms", float("nan"))
    short_itl_max = short_itl.get("max_ms", float("nan"))
    if short_itl_p50 and not math.isnan(short_itl_p50):
        short_itl_spike_ratio = short_itl_max / short_itl_p50
    else:
        short_itl_spike_ratio = float("nan")

    return {
        "scenario": scenario,
        "policy": policy,
        "repeat": repeat,
        "chunked_prefill": enabled,
        "skipped": result.get("skipped", False),
        "skip_reason": result.get("skip_reason"),
        "input_tokens": result.get("total_input_tokens", 0),
        "output_tokens": result.get("total_output_tokens", 0),
        "total_latency_s": result["total_latency_s"],
        "total_tps": result.get("total_tps", float("nan")),
        "output_tps": result.get("output_tps", float("nan")),
        "ttft": result["ttft"],
        "itl": result["itl"],
        "ttft_short": result["ttft_short"],
        "itl_short": result["itl_short"],
        "ttft_long": result["ttft_long"],
        "itl_long": result["itl_long"],
        "short_itl_spike_ratio": short_itl_spike_ratio,
    }


def mean_finite(values: list[float]) -> float:
    finite = [value for value in values if value is not None and not math.isnan(value)]
    return statistics.mean(finite) if finite else float("nan")


def aggregate_metric_dict(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    metric_names = ["mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"]
    return {
        metric_name: mean_finite([row[key].get(metric_name, float("nan")) for row in rows])
        for metric_name in metric_names
    }


def aggregate_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (row["scenario"], row["policy"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    aggregated = []
    for key in order:
        group = grouped[key]
        first = group[0]
        completed = [row for row in group if not row.get("skipped")]
        if not completed:
            row = dict(first)
            row["repeats"] = len(group)
            aggregated.append(row)
            continue
        aggregated.append(
            {
                "scenario": first["scenario"],
                "policy": first["policy"],
                "repeats": len(completed),
                "chunked_prefill": first["chunked_prefill"],
                "skipped": False,
                "skip_reason": None,
                "input_tokens": round(mean_finite([row["input_tokens"] for row in completed])),
                "output_tokens": round(mean_finite([row["output_tokens"] for row in completed])),
                "total_latency_s": mean_finite([row["total_latency_s"] for row in completed]),
                "total_tps": mean_finite([row["total_tps"] for row in completed]),
                "output_tps": mean_finite([row["output_tps"] for row in completed]),
                "ttft": aggregate_metric_dict(completed, "ttft"),
                "itl": aggregate_metric_dict(completed, "itl"),
                "ttft_short": aggregate_metric_dict(completed, "ttft_short"),
                "itl_short": aggregate_metric_dict(completed, "itl_short"),
                "ttft_long": aggregate_metric_dict(completed, "ttft_long"),
                "itl_long": aggregate_metric_dict(completed, "itl_long"),
                "short_itl_spike_ratio": mean_finite(
                    [row["short_itl_spike_ratio"] for row in completed]
                ),
            }
        )
    return aggregated


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "scenario",
        "policy",
        "repeats",
        "input_tokens",
        "output_tokens",
        "TTFT p50/p95/p99(ms)",
        "ITL p50/p95/p99/max(ms)",
        "short ITL spike",
        "total tok/s",
        "output tok/s",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        if row.get("skipped"):
            print(
                "| "
                + " | ".join(
                    [
                        row["scenario"],
                        row["policy"],
                        str(row.get("repeats", 0)),
                        str(row["input_tokens"]),
                        str(row["output_tokens"]),
                        "skipped",
                        "skipped",
                        "skipped",
                        "skipped",
                        "skipped",
                    ]
                )
                + " |"
            )
            continue
        ttft = row["ttft"]
        itl = row["itl"]
        print(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    row["policy"],
                    str(row.get("repeats", 1)),
                    str(row["input_tokens"]),
                    str(row["output_tokens"]),
                    f"{ttft['p50_ms']:.2f}/{ttft['p95_ms']:.2f}/{ttft['p99_ms']:.2f}",
                    f"{itl['p50_ms']:.2f}/{itl['p95_ms']:.2f}/{itl['p99_ms']:.2f}/{itl['max_ms']:.2f}",
                    f"{row['short_itl_spike_ratio']:.2f}",
                    f"{row['total_tps']:.2f}",
                    f"{row['output_tps']:.2f}",
                ]
            )
            + " |"
        )


def print_short_request_table(rows: list[dict[str, Any]]) -> None:
    # 只在混合场景下有意义：short 请求单独的 TTFT/ITL，剔除 long 请求均值掩盖的影响。
    # ITL max 是关键指标——它体现 baseline 下 long prefill 独占一个 step 时，
    # short 请求的 decode 是否出现被打断的尖峰；chunked prefill 应显著压低这个尖峰。
    headers = [
        "scenario",
        "policy",
        "repeats",
        "short TTFT p50/p95/p99(ms)",
        "short ITL p50/p95/p99/max(ms)",
        "short ITL spike",
    ]
    print("\n| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        if row.get("skipped"):
            continue
        ttft = row["ttft_short"]
        itl = row["itl_short"]
        if math.isnan(ttft["mean_ms"]):
            continue
        print(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    row["policy"],
                    str(row.get("repeats", 1)),
                    f"{ttft['p50_ms']:.2f}/{ttft['p95_ms']:.2f}/{ttft['p99_ms']:.2f}",
                    f"{itl['p50_ms']:.2f}/{itl['p95_ms']:.2f}/{itl['p99_ms']:.2f}/{itl['max_ms']:.2f}",
                    f"{row['short_itl_spike_ratio']:.2f}",
                ]
            )
            + " |"
        )


def build_chunk_size_policies(chunk_sizes: list[int]) -> dict[str, dict[str, int | None]]:
    # max_num_partial_prefills / max_long_partial_prefills 固定为 vLLM 的默认值 1
    # (vllm-main/vllm/config/scheduler.py: SchedulerConfig.max_num_partial_prefills=1,
    # max_long_partial_prefills=1)。每个 chunk size 对应一个 long_prefill_token_threshold 取值，
    # 作为一组独立的 policy 参与横向对比。
    return {
        f"chunk_{size}": {
            "long_prefill_token_threshold": size,
            "max_num_partial_prefills": 1,
            "max_long_partial_prefills": 1,
        }
        for size in chunk_sizes
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MinivLLM chunked prefill scheduling.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    # 默认值需要 >= long prompt 长度，否则 non-chunked baseline 会因超预算被 skip，
    # 导致 long_only/mixed_long_short 场景完全没有 baseline 对照（3072 覆盖默认 --long-words=1800 生成的 prompt）。
    parser.add_argument("--max-num-batched-tokens", type=int, default=3072)
    parser.add_argument("--max-model-length", type=int, default=4096)
    parser.add_argument(
        "--chunk-sizes",
        default="256,512,1024,2048",
        help="Comma separated long_prefill_token_threshold values to sweep as chunk sizes",
    )
    parser.add_argument("--short-words", type=int, default=32)
    parser.add_argument("--long-words", type=int, default=1800)
    parser.add_argument(
        "--num-concurrent-short",
        type=int,
        default=16,
        help="Number of concurrent short (decode-heavy) requests in mixed_long_short, to simulate high concurrency",
    )
    parser.add_argument(
        "--num-concurrent-long",
        type=int,
        default=1,
        help="Number of long prompts arriving during decode in mixed_long_short (and count in long_only)",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of measured repeats per scenario/policy")
    parser.add_argument(
        "--policies",
        default="all",
        help="Comma separated policies to run: baseline, chunk_<size>, or all",
    )
    parser.add_argument("--shuffle-runs", action="store_true", help="Shuffle scenario/policy/repeat run order")
    parser.add_argument("--nvtx", action="store_true", help="Add NVTX ranges for Nsight Systems/Compute profiling")
    parser.add_argument(
        "--long-arrival-gap-steps",
        type=int,
        default=LONG_PROMPT_ARRIVAL_GAP_STEPS,
        help=(
            "In high_concurrency_long_context/mixed_long_short: after all immediate (short) "
            "requests reach decode, long prompts stream in one at a time spaced by this many "
            "engine.step() calls, simulating continuously arriving long-context requests"
        ),
    )
    parser.add_argument("--output", default="results/chunk_prefill_benchmark.json")
    parser.add_argument("--progress", action="store_true", help="Print per-step scheduler progress")
    parser.add_argument("--max-steps", type=int, default=None, help="Abort a scenario after this many engine steps")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable warmup request before timing; warmup is on by default to exclude JIT/cold-start overhead",
    )
    parser.add_argument(
        "--scenarios",
        default="short_only,long_only,mixed_long_short",
        help="Comma separated scenarios: short_only,long_only,mixed_long_short",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_chunk_prefill.py requires CUDA for meaningful latency results")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    base_config = dict(DEFAULT_CONFIG)
    base_config.update(
        {
            "model_name_or_path": args.model,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_batch_tokens": max(
                args.max_num_batched_tokens,
                args.max_model_length,
                DEFAULT_CONFIG["max_num_batch_tokens"],
            ),
            "max_model_length": args.max_model_length,
        }
    )

    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    chunk_sizes = [int(item.strip()) for item in args.chunk_sizes.split(",") if item.strip()]
    chunk_policies = build_chunk_size_policies(chunk_sizes)

    benchmark_runs: list[dict[str, Any]] = []
    for scenario in scenarios:
        named_prompts = scenario_prompts(
            scenario,
            args.short_words,
            args.long_words,
            num_concurrent_short=args.num_concurrent_short,
            num_concurrent_long=args.num_concurrent_long,
        )
        policies: list[tuple[str, bool, dict[str, int | None]]] = [(BASELINE_POLICY, False, {})]
        policies.extend((name, True, dict(policy_config)) for name, policy_config in chunk_policies.items())
        policies = filter_policies(policies, args.policies)
        for policy, enabled, policy_config in policies:
            for repeat in range(args.repeats):
                benchmark_runs.append(
                    {
                        "scenario": scenario,
                        "named_prompts": named_prompts,
                        "policy": policy,
                        "enabled": enabled,
                        "policy_config": dict(policy_config),
                        "repeat": repeat,
                    }
                )

    if args.shuffle_runs:
        random.shuffle(benchmark_runs)

    raw_results = []
    repeat_rows = []
    for run in benchmark_runs:
        scenario = run["scenario"]
        named_prompts = run["named_prompts"]
        policy = run["policy"]
        enabled = run["enabled"]
        policy_config = run["policy_config"]
        repeat = run["repeat"]
        config = dict(base_config)
        config["enable_chunked_prefill"] = enabled
        if enabled:
            config.update(policy_config)

        print(
            f"Running scenario={scenario}, policy={policy}, repeat={repeat + 1}/{args.repeats}, "
            f"chunked_prefill={enabled}, scheduler={policy_config}"
        )
        run_range = f"scenario={scenario}/policy={policy}/repeat={repeat}"
        with nvtx_range(run_range, enabled=args.nvtx):
            skip_reason = get_skip_reason(config, tokenizer, named_prompts)
            if skip_reason is not None:
                print(f"Skipping scenario={scenario}, policy={policy}, chunked_prefill={enabled}: {skip_reason}")
                result = skipped_result(skip_reason)
            else:
                # mixed_long_short / high_concurrency_long_context 场景延迟插入 long 请求：
                # 等 short 请求都进入 decode 后再陆续插入，才能观察到 long prefill 对正在
                # 进行的 short decode 的打断（ITL 尖峰），且插入时机不受并发数变化影响。
                delay_kwargs = (
                    {"delay_insert_kind": "long", "arrival_gap_steps": args.long_arrival_gap_steps}
                    if scenario in ("mixed_long_short", "high_concurrency_long_context")
                    else {}
                )
                result = run_engine_steps(
                    config,
                    tokenizer,
                    named_prompts,
                    max_output_tokens=args.max_output_tokens,
                    progress=args.progress,
                    max_steps=args.max_steps,
                    warmup=not args.no_warmup,
                    nvtx=args.nvtx,
                    nvtx_prefix=run_range,
                    **delay_kwargs,
                )
        raw_results.append(
            {
                "scenario": scenario,
                "policy": policy,
                "repeat": repeat,
                "chunked_prefill": enabled,
                "result": result,
            }
        )
        repeat_rows.append(summarize_result(scenario, enabled, result, policy, repeat=repeat))

    summary_rows = aggregate_summary_rows(repeat_rows)

    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "scenarios": scenarios,
                    "chunk_sizes": chunk_sizes,
                    "policies": args.policies,
                    "repeats": args.repeats,
                    "nvtx": args.nvtx,
                },
                "summary": summary_rows,
                "repeat_summary": repeat_rows,
                "raw": raw_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Chunk Prefill Benchmark Summary ===")
    print_table(summary_rows)
    print_short_request_table(summary_rows)
    print(f"\nSaved JSON results to {output_path}")


if __name__ == "__main__":
    main()
