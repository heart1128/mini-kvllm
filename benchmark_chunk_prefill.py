import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "max_num_sequences": 16,
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


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "off"}:
        return None
    return int(value)


POLICY_PRESETS: dict[str, dict[str, int | None]] = {
    # 正常 chunked prefill：不额外限制单步 chunk size，也不限制 partial prefill 并发。
    "default": {
        "long_prefill_token_threshold": 0,
        "max_num_partial_prefills": None,
        "max_long_partial_prefills": None,
    },
    # 单步 chunk size 上限：限制单个长 prompt 每轮最多吃多少 prefill token budget。
    "chunk_limit": {
        "long_prefill_token_threshold": 512,
        "max_num_partial_prefills": None,
        "max_long_partial_prefills": None,
    },
    # 长短调度限制：在 chunk size 上限基础上，只允许少量 partial/long partial prefill 同时在 running 中。
    "partial_limits": {
        "long_prefill_token_threshold": 512,
        "max_num_partial_prefills": 1,
        "max_long_partial_prefills": 1,
    },
}


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_prompt(kind: str, target_words: int) -> str:
    seed = {
        "short": "Summarize the main benefit of chunked prefill in one sentence.",
        "long": "Explain the design of a language model serving scheduler, including prefill, decode, KV cache, batching, and fairness.",
    }[kind]
    words = seed.split()
    repeated = []
    while len(repeated) < target_words:
        repeated.extend(words)
    return " ".join(repeated[:target_words])


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


def scenario_prompts(name: str, short_words: int, long_words: int) -> list[tuple[str, str]]:
    if name == "short_only":
        return [(f"short_{i}", build_prompt("short", short_words)) for i in range(4)]
    if name == "long_only":
        return [("long_0", build_prompt("long", long_words))]
    if name == "mixed_long_short":
        return [
            ("long_0", build_prompt("long", long_words)),
            ("short_0", build_prompt("short", short_words)),
            ("short_1", build_prompt("short", short_words)),
        ]
    raise ValueError(f"Unknown scenario: {name}")


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


def skipped_result(reason: str) -> dict[str, Any]:
    return {
        "skipped": True,
        "skip_reason": reason,
        "total_latency_s": float("nan"),
        "total_output_tokens": 0,
        "decode_tps": float("nan"),
        "mean_ttft_ms": float("nan"),
        "mean_tpot_ms": float("nan"),
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


def run_engine_steps(
    config: dict[str, Any],
    tokenizer: Any,
    named_prompts: list[tuple[str, str]],
    max_output_tokens: int,
    progress: bool = False,
    max_steps: int | None = None,
) -> dict[str, Any]:
    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.sampling_parameters import SamplingParams

    engine = LLMEngine(config=config)
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
    seq_id_to_name = {}
    for (name, _), rendered_prompt in zip(named_prompts, rendered_prompts):
        engine.add_prompt(rendered_prompt, sampling_params)
        seq_id_to_name[engine.scheduler.waiting[-1].seq_id] = name

    per_request = {
        name: {
            "ttft_s": None,
            "finished_s": None,
            "output_tokens": 0,
            "decode_step_times_s": [],
        }
        for name, _ in named_prompts
    }

    # 计时从请求加入 scheduler 后开始，不包含模型加载和 KV cache 分配。
    # progress/max_steps 是调试卡住场景用的：能区分 scheduler 无进展、Triton JIT 卡住、还是普通慢 step。
    start = time.perf_counter()
    step_records = []
    try:
        while not engine.scheduler.is_finished():
            if max_steps is not None and len(step_records) >= max_steps:
                raise RuntimeError(
                    f"Benchmark exceeded --max-steps={max_steps}; "
                    f"waiting={len(engine.scheduler.waiting)}, running={len(engine.scheduler.running)}"
                )
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
                    f"{len(step_records)} elapsed={elapsed:.3f}s "
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
                if item.num_completion_tokens > 0 and per_request[name]["ttft_s"] is None:
                    per_request[name]["ttft_s"] = elapsed
                if item.num_completion_tokens > 1:
                    per_request[name]["decode_step_times_s"].append(step_time)
    finally:
        engine.exit()

    total_latency_s = time.perf_counter() - start

    total_output_tokens = sum(req["output_tokens"] for req in per_request.values())
    decode_times = [
        t
        for req in per_request.values()
        for t in req["decode_step_times_s"]
    ]
    ttfts = [req["ttft_s"] for req in per_request.values() if req["ttft_s"] is not None]

    return {
        "total_latency_s": total_latency_s,
        "total_output_tokens": total_output_tokens,
        "decode_tps": total_output_tokens / total_latency_s if total_latency_s > 0 else float("nan"),
        "mean_ttft_ms": statistics.mean(ttfts) * 1000 if ttfts else float("nan"),
        "mean_tpot_ms": statistics.mean(decode_times) * 1000 if decode_times else float("nan"),
        "prompt_tokens": prompt_tokens,
        "per_request": per_request,
        "steps": step_records,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * pct) - 1))
    return ordered[index]


def mean_or_nan(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def sum_or_nan(values: list[float]) -> float:
    return sum(values) if values else float("nan")


def summarize_result(
    scenario: str,
    enabled: bool,
    result: dict[str, Any],
    policy: str = "baseline",
) -> dict[str, Any]:
    per_request = result["per_request"]
    prompt_tokens_by_name = result.get("prompt_tokens", {})
    short_ttfts = [
        req["ttft_s"] * 1000
        for name, req in per_request.items()
        if name.startswith("short") and req["ttft_s"] is not None
    ]
    long_ttfts = [
        req["ttft_s"] * 1000
        for name, req in per_request.items()
        if name.startswith("long") and req["ttft_s"] is not None
    ]
    steps = result.get("steps", [])
    prefill_step_records = [step for step in steps if step.get("is_prefill")]
    decode_step_records = [step for step in steps if not step.get("is_prefill")]
    prefill_steps = len(prefill_step_records)
    decode_steps = len(decode_step_records)
    prefill_step_times_ms = [step["step_time_s"] * 1000 for step in prefill_step_records]
    decode_step_times_ms = [step["step_time_s"] * 1000 for step in decode_step_records]
    processed_tokens = [step.get("num_processed_tokens", 0) for step in steps]
    prefill_processed_tokens = [step.get("num_processed_tokens", 0) for step in prefill_step_records]
    decode_processed_tokens = [step.get("num_processed_tokens", 0) for step in decode_step_records]
    budget_utilizations = [step.get("budget_utilization", float("nan")) for step in steps]
    long_prompt_tokens = sum(
        tokens for name, tokens in prompt_tokens_by_name.items() if name.startswith("long")
    )
    short_prompt_tokens = sum(
        tokens for name, tokens in prompt_tokens_by_name.items() if name.startswith("short")
    )
    return {
        "scenario": scenario,
        "policy": policy,
        "chunked_prefill": enabled,
        "skipped": result.get("skipped", False),
        "skip_reason": result.get("skip_reason"),
        "ttft_ms": result["mean_ttft_ms"],
        "long_ttft_ms": statistics.mean(long_ttfts) if long_ttfts else float("nan"),
        "short_request_ttft_under_long_prefill_ms": statistics.mean(short_ttfts) if short_ttfts else float("nan"),
        "short_p90_ttft_ms": percentile(short_ttfts, 0.9),
        "tpot_ms": result["mean_tpot_ms"],
        "total_latency_s": result["total_latency_s"],
        "decode_tps": result["decode_tps"],
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "prompt_tokens": sum(prompt_tokens_by_name.values()),
        "long_prompt_tokens": long_prompt_tokens,
        "short_prompt_tokens": short_prompt_tokens,
        "prefill_step_ms": mean_or_nan(prefill_step_times_ms),
        "prefill_p90_step_ms": percentile(prefill_step_times_ms, 0.9),
        "decode_step_ms": mean_or_nan(decode_step_times_ms),
        "decode_p90_step_ms": percentile(decode_step_times_ms, 0.9),
        "processed_tokens_per_s": (
            sum_or_nan(processed_tokens) / result["total_latency_s"]
            if processed_tokens and result["total_latency_s"] > 0 else float("nan")
        ),
        "prefill_processed_tokens_per_s": (
            sum_or_nan(prefill_processed_tokens) / sum(step["step_time_s"] for step in prefill_step_records)
            if prefill_step_records and sum(step["step_time_s"] for step in prefill_step_records) > 0 else float("nan")
        ),
        "decode_processed_tokens_per_s": (
            sum_or_nan(decode_processed_tokens) / sum(step["step_time_s"] for step in decode_step_records)
            if decode_step_records and sum(step["step_time_s"] for step in decode_step_records) > 0 else float("nan")
        ),
        "avg_budget_utilization": mean_or_nan(budget_utilizations),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "scenario",
        "policy",
        "chunked",
        "TTFT(ms)",
        "long TTFT(ms)",
        "short TTFT(ms)",
        "short p90(ms)",
        "TPOT(ms)",
        "latency(s)",
        "decode tok/s",
        "prefill steps",
        "decode steps",
        "prefill step(ms)",
        "decode step(ms)",
        "proc tok/s",
        "budget util",
        "prompt toks",
        "long toks",
        "short toks",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    row["policy"],
                    "on" if row["chunked_prefill"] else "off",
                    "skipped" if row.get("skipped") else f"{row['ttft_ms']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['long_ttft_ms']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['short_request_ttft_under_long_prefill_ms']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['short_p90_ttft_ms']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['tpot_ms']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['total_latency_s']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['decode_tps']:.2f}",
                    "skipped" if row.get("skipped") else str(row["prefill_steps"]),
                    "skipped" if row.get("skipped") else str(row["decode_steps"]),
                    "skipped" if row.get("skipped") else f"{row['prefill_step_ms']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['decode_step_ms']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['processed_tokens_per_s']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['avg_budget_utilization']:.2f}",
                    "skipped" if row.get("skipped") else str(row["prompt_tokens"]),
                    "skipped" if row.get("skipped") else str(row["long_prompt_tokens"]),
                    "skipped" if row.get("skipped") else str(row["short_prompt_tokens"]),
                ]
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MinivLLM chunked prefill scheduling.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-model-length", type=int, default=4096)
    parser.add_argument(
        "--policies",
        default="default,chunk_limit,partial_limits",
        help="Comma separated chunked policies: default,chunk_limit,partial_limits",
    )
    parser.add_argument(
        "--long-prefill-token-threshold",
        type=int,
        default=None,
        help="Override long prefill chunk size threshold for chunked policy runs; 0 disables the threshold",
    )
    parser.add_argument(
        "--max-num-partial-prefills",
        default="__unset__",
        help="Override max partial prefill count; use none/null/off for unlimited",
    )
    parser.add_argument(
        "--max-long-partial-prefills",
        default="__unset__",
        help="Override max long partial prefill count; use none/null/off for unlimited",
    )
    parser.add_argument("--short-words", type=int, default=32)
    parser.add_argument("--long-words", type=int, default=1800)
    parser.add_argument("--output", default="results/chunk_prefill_benchmark.json")
    parser.add_argument("--progress", action="store_true", help="Print per-step scheduler progress")
    parser.add_argument("--max-steps", type=int, default=None, help="Abort a scenario after this many engine steps")
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
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    unknown_policies = [policy for policy in policies if policy not in POLICY_PRESETS]
    if unknown_policies:
        raise ValueError(f"Unknown policies: {unknown_policies}; choose from {sorted(POLICY_PRESETS)}")

    raw_results = []
    summary_rows = []
    for scenario in scenarios:
        named_prompts = scenario_prompts(scenario, args.short_words, args.long_words)
        benchmark_runs: list[tuple[str, bool, dict[str, int | None]]] = [("baseline", False, {})]
        benchmark_runs.extend((policy, True, dict(POLICY_PRESETS[policy])) for policy in policies)

        for policy, enabled, policy_config in benchmark_runs:
            config = dict(base_config)
            config["enable_chunked_prefill"] = enabled
            if enabled:
                if args.long_prefill_token_threshold is not None:
                    policy_config["long_prefill_token_threshold"] = args.long_prefill_token_threshold
                if args.max_num_partial_prefills != "__unset__":
                    policy_config["max_num_partial_prefills"] = parse_optional_int(args.max_num_partial_prefills)
                if args.max_long_partial_prefills != "__unset__":
                    policy_config["max_long_partial_prefills"] = parse_optional_int(args.max_long_partial_prefills)
                config.update(policy_config)

            print(
                f"Running scenario={scenario}, policy={policy}, "
                f"chunked_prefill={enabled}, scheduler={policy_config}"
            )
            skip_reason = get_skip_reason(config, tokenizer, named_prompts)
            if skip_reason is not None:
                print(f"Skipping scenario={scenario}, policy={policy}, chunked_prefill={enabled}: {skip_reason}")
                result = skipped_result(skip_reason)
            else:
                result = run_engine_steps(
                    config,
                    tokenizer,
                    named_prompts,
                    args.max_output_tokens,
                    progress=args.progress,
                    max_steps=args.max_steps,
                )
            raw_results.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "chunked_prefill": enabled,
                    "scheduler_config": policy_config,
                    "result": result,
                }
            )
            summary_rows.append(summarize_result(scenario, enabled, result, policy=policy))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary_rows, "raw": raw_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Chunk Prefill Benchmark Summary ===")
    print_table(summary_rows)
    print(f"\nSaved JSON results to {output_path}")


if __name__ == "__main__":
    main()
