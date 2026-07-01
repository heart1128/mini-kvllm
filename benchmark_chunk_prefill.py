import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "max_num_sequences": 16,
    "max_num_batched_tokens": 1024,
    "max_num_batch_tokens": 4096,
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

    start = time.perf_counter()
    step_records = []
    try:
        while not engine.scheduler.is_finished():
            step_start = time.perf_counter()
            outputs, num_processed_tokens, is_prefill = engine.step()
            raise_if_no_progress(engine.scheduler, outputs, num_processed_tokens, config)
            cuda_sync()
            step_end = time.perf_counter()
            elapsed = step_end - start
            step_time = step_end - step_start
            step_records.append(
                {
                    "elapsed_s": elapsed,
                    "step_time_s": step_time,
                    "num_processed_tokens": num_processed_tokens,
                    "is_prefill": is_prefill,
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
        "per_request": per_request,
        "steps": step_records,
    }


def summarize_result(scenario: str, enabled: bool, result: dict[str, Any]) -> dict[str, Any]:
    short_ttfts = [
        req["ttft_s"] * 1000
        for name, req in result["per_request"].items()
        if name.startswith("short") and req["ttft_s"] is not None
    ]
    return {
        "scenario": scenario,
        "chunked_prefill": enabled,
        "skipped": result.get("skipped", False),
        "skip_reason": result.get("skip_reason"),
        "ttft_ms": result["mean_ttft_ms"],
        "tpot_ms": result["mean_tpot_ms"],
        "total_latency_s": result["total_latency_s"],
        "decode_tps": result["decode_tps"],
        "short_request_ttft_under_long_prefill_ms": statistics.mean(short_ttfts) if short_ttfts else float("nan"),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "scenario",
        "chunked",
        "TTFT(ms)",
        "TPOT(ms)",
        "latency(s)",
        "decode tok/s",
        "short TTFT(ms)",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    "on" if row["chunked_prefill"] else "off",
                    "skipped" if row.get("skipped") else f"{row['ttft_ms']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['tpot_ms']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['total_latency_s']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['decode_tps']:.2f}",
                    "skipped" if row.get("skipped") else f"{row['short_request_ttft_under_long_prefill_ms']:.2f}",
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
    parser.add_argument("--short-words", type=int, default=32)
    parser.add_argument("--long-words", type=int, default=1800)
    parser.add_argument("--output", default="results/chunk_prefill_benchmark.json")
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
    raw_results = []
    summary_rows = []
    for scenario in scenarios:
        named_prompts = scenario_prompts(scenario, args.short_words, args.long_words)
        for enabled in (False, True):
            config = dict(base_config)
            config["enable_chunked_prefill"] = enabled
            print(f"Running scenario={scenario}, chunked_prefill={enabled}")
            skip_reason = get_skip_reason(config, tokenizer, named_prompts)
            if skip_reason is not None:
                print(f"Skipping scenario={scenario}, chunked_prefill={enabled}: {skip_reason}")
                result = skipped_result(skip_reason)
            else:
                result = run_engine_steps(config, tokenizer, named_prompts, args.max_output_tokens)
            raw_results.append(
                {
                    "scenario": scenario,
                    "chunked_prefill": enabled,
                    "result": result,
                }
            )
            summary_rows.append(summarize_result(scenario, enabled, result))

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
