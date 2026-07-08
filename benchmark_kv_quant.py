"""
KV Cache 量化 Benchmark
======================

评测 Qwen3-0.6B 在不同 KV cache 存储模式下的核心指标：

  1. Memory              显存容量：每 block KV 字节数、max_cached_blocks、容量提升
  2. Latency/Throughput  单请求延迟与 decode 吞吐：TTFT、TPOT、tokens/s
  3. Concurrency         多并发端到端吞吐：batch size 扩展收益
  4. Accuracy            精度：相对 fp16 基线的 token 一致率

输出策略：终端报告 + JSON 原始结果 + PNG 图；不生成表格文件。

用法:
    python benchmark_kv_quant.py
    python benchmark_kv_quant.py --suite memory
    python benchmark_kv_quant.py --suite memory,latency,accuracy
    python benchmark_kv_quant.py --no-plots
    python benchmark_kv_quant.py --output-dir benchmark_results
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))


QUANT_FORMAT = "FP8 E4M3FN (max=448) / INT8 (range=-128..127)"
MODE_COLORS = {
    "auto": "#888888",
    "fp8_per_tensor": "#e88aac",
    "fp8_per_token_head": "#55aa88",
    "int8_per_token_head": "#5588cc",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    output_dir: Path = Path(__file__).parent / "benchmark_results"
    quant_modes: tuple[str, ...] = (
        "auto",
        "fp8_per_tensor",
        "fp8_per_token_head",
        "int8_per_token_head",
    )
    max_num_sequences: int = 8
    max_num_batched_tokens: int = 8192
    max_cached_blocks: int = 2048
    block_size: int = 256
    world_size: int = 1
    vocab_size: int = 151936
    hidden_size: int = 1024
    num_heads: int = 16
    head_dim: int = 128
    num_kv_heads: int = 8
    intermediate_size: int = 3072
    num_layers: int = 28
    max_position: int = 32768
    gpu_memory_utilization: float = 0.85
    eos: int = 151645
    latency_prompt_len: int = 256
    latency_gen_tokens: int = 64
    latency_max_model_length: int = 1024
    concurrency_batch_sizes: tuple[int, ...] = (1, 4, 8)
    concurrency_prompt_len: int = 128
    concurrency_gen_tokens: int = 32
    concurrency_max_model_length: int = 512
    accuracy_max_tokens: int = 48
    accuracy_max_model_length: int = 512
    accuracy_prompts: tuple[str, ...] = (
        "Explain what a transformer is in one sentence.",
        "List the first 10 prime numbers.",
        "What is the capital of France?",
        "用一句话解释什么是梯度下降。",
    )


def require_torch():
    import torch

    return torch


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    cols = len(headers)
    widths = [len(str(headers[c])) for c in range(cols)]
    for row in rows:
        for c in range(cols):
            widths[c] = max(widths[c], len(str(row[c])))

    def fmt_row(cells: list[Any]) -> str:
        return "| " + " | ".join(str(cells[c]).ljust(widths[c]) for c in range(cols)) + " |"

    sep = "|" + "|".join("-" * (widths[c] + 2) for c in range(cols)) + "|"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])


def print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(render_table(headers, rows))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def save_json_report(results: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "kv_quant_results.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    return out


def has_matplotlib() -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 未检测到 matplotlib，将跳过 PNG 图。原因: {exc}")
        return False


def build_engine_config(config: BenchmarkConfig, kv_cache_dtype: str, max_model_length: int) -> dict[str, Any]:
    return {
        "max_num_sequences": config.max_num_sequences,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "max_cached_blocks": config.max_cached_blocks,
        "block_size": config.block_size,
        "world_size": config.world_size,
        "model_name_or_path": config.model_name_or_path,
        "enforce_eager": True,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_heads": config.num_heads,
        "head_dim": config.head_dim,
        "num_kv_heads": config.num_kv_heads,
        "intermediate_size": config.intermediate_size,
        "num_layers": config.num_layers,
        "tie_word_embeddings": True,
        "base": 1000000,
        "rms_norm_epsilon": 1e-6,
        "qkv_bias": False,
        "scale": 1,
        "max_position": config.max_position,
        "ffn_bias": False,
        "max_num_batch_tokens": config.max_num_batched_tokens,
        "max_model_length": max_model_length,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "eos": config.eos,
        "kv_cache_dtype": kv_cache_dtype,
    }


def load_engine(config: BenchmarkConfig, kv_cache_dtype: str, max_model_length: int):
    torch = require_torch()
    from transformers import AutoTokenizer
    from myvllm.engine.llm_engine import LLMEngine as LLM

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    engine_config = build_engine_config(config, kv_cache_dtype, max_model_length)
    tokenizer = AutoTokenizer.from_pretrained(engine_config["model_name_or_path"])
    engine = LLM(config=engine_config)
    return engine, tokenizer


def free_engine(engine: Any) -> None:
    torch = require_torch()
    try:
        model_runner = getattr(engine, "model_runner", None)
        if model_runner is not None and hasattr(model_runner, "model"):
            del model_runner.model
    except Exception:  # noqa: BLE001
        pass
    try:
        engine.exit()
    except Exception:  # noqa: BLE001
        pass
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def chat_prompt(tokenizer: Any, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def gen_prompt_of_len(tokenizer: Any, n_tokens: int) -> str:
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "Paris is a city with a long history of art and culture. "
        "Machine learning models process large amounts of data. "
    )
    ids = tokenizer.encode(filler * max(1, n_tokens // 8))[:n_tokens]
    return tokenizer.decode(ids)


def empty_results(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": config.model_name_or_path,
            "quant_format": QUANT_FORMAT,
            "quant_modes": list(config.quant_modes),
            "config": {k: json_default(v) for k, v in asdict(config).items()},
        },
        "memory": {},
        "latency": {},
        "concurrency": {},
        "accuracy": {},
        "failures": [],
    }


def record_failure(results: dict[str, Any], suite: str, mode: str, error: Exception) -> None:
    message = f"{type(error).__name__}: {error}"
    print(f"  [失败] {suite} / {mode}: {message}")
    results["failures"].append({"suite": suite, "mode": mode, "error": message})


def block_bytes_for_mode(config: BenchmarkConfig, mode: str) -> dict[str, float]:
    elem_bytes = 2 if mode == "auto" else 1
    data_bytes = config.block_size * 2 * config.num_layers * config.num_kv_heads * config.head_dim * elem_bytes
    scale_bytes = 0
    if mode in ("fp8_per_token_head", "int8_per_token_head"):
        scale_bytes = config.block_size * 2 * config.num_layers * config.num_kv_heads * 4
    total_bytes = data_bytes + scale_bytes
    fp16_bytes = config.block_size * 2 * config.num_layers * config.num_kv_heads * config.head_dim * 2
    return {
        "data_bytes": data_bytes,
        "scale_bytes": scale_bytes,
        "block_bytes": total_bytes,
        "relative_to_fp16": total_bytes / fp16_bytes,
    }


def first_tensor(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def tensor_dtype_name(tensor: Any) -> str | None:
    if tensor is None or not hasattr(tensor, "dtype"):
        return None
    return str(tensor.dtype)


def tensor_shape(tensor: Any) -> list[int] | None:
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return [int(dim) for dim in tensor.shape]


def tensor_numel(tensor: Any) -> int:
    if tensor is None or not hasattr(tensor, "numel"):
        return 0
    return int(tensor.numel())


def tensor_element_size(tensor: Any) -> int:
    if tensor is None or not hasattr(tensor, "element_size"):
        return 0
    return int(tensor.element_size())


def first_attention_with_kv_cache(engine: Any) -> Any:
    model_runner = getattr(engine, "model_runner", None)
    model = getattr(model_runner, "model", None)
    modules = model.modules() if model is not None and hasattr(model, "modules") else []
    for module in modules:
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            return module
    return None


def coalesce_tensor(*values: Any) -> Any:
    for value in values:
        tensor = first_tensor(value)
        if tensor is not None:
            return tensor
    return None


def collect_kv_cache_diagnostics(engine: Any, block_size: int) -> dict[str, Any]:
    model_runner = getattr(engine, "model_runner", None)
    attention = first_attention_with_kv_cache(engine)
    k_cache = coalesce_tensor(getattr(attention, "k_cache", None), getattr(model_runner, "k_cache", None))
    v_cache = coalesce_tensor(getattr(attention, "v_cache", None), getattr(model_runner, "v_cache", None))
    k_scale = coalesce_tensor(getattr(attention, "k_scale", None), getattr(model_runner, "k_scale", None))
    v_scale = coalesce_tensor(getattr(attention, "v_scale", None), getattr(model_runner, "v_scale", None))
    kv_scale = coalesce_tensor(getattr(model_runner, "kv_scale", None))

    k_elem_bytes = tensor_element_size(k_cache)
    v_elem_bytes = tensor_element_size(v_cache)
    k_shape = tensor_shape(k_cache)
    v_shape = tensor_shape(v_cache)
    k_scale_shape = tensor_shape(k_scale)
    v_scale_shape = tensor_shape(v_scale)
    kv_scale_shape = tensor_shape(kv_scale)

    k_scale_numel = tensor_numel(k_scale)
    v_scale_numel = tensor_numel(v_scale)
    kv_scale_numel = tensor_numel(kv_scale)
    k_scale_bytes = k_scale_numel * tensor_element_size(k_scale)
    v_scale_bytes = v_scale_numel * tensor_element_size(v_scale)
    kv_scale_bytes = kv_scale_numel * tensor_element_size(kv_scale)

    num_kv_heads = k_shape[2] if k_shape and len(k_shape) >= 4 else 0
    head_dim = k_shape[3] if k_shape and len(k_shape) >= 4 else 0
    actual_block_bytes = block_size * num_kv_heads * head_dim * (k_elem_bytes + v_elem_bytes)
    scale_bytes = k_scale_bytes + v_scale_bytes + kv_scale_bytes

    return {
        "actual_k_cache_dtype": tensor_dtype_name(k_cache),
        "actual_v_cache_dtype": tensor_dtype_name(v_cache),
        "actual_k_cache_shape": k_shape,
        "actual_v_cache_shape": v_shape,
        "actual_cache_elem_bytes": k_elem_bytes,
        "actual_block_bytes": actual_block_bytes,
        "actual_k_scale_shape": k_scale_shape,
        "actual_v_scale_shape": v_scale_shape,
        "actual_scale_shape": kv_scale_shape,
        "actual_scale_numel": k_scale_numel + v_scale_numel + kv_scale_numel,
        "actual_scale_bytes": scale_bytes,
        "actual_total_block_bytes": actual_block_bytes,
    }


def run_memory_benchmark(config: BenchmarkConfig, results: dict[str, Any]) -> None:
    baseline_blocks = None
    baseline_actual_block_bytes = None
    for mode in config.quant_modes:
        stats = block_bytes_for_mode(config, mode)
        engine = None
        try:
            engine, _ = load_engine(config, mode, max_model_length=config.latency_max_model_length)
            max_blocks = int(engine.config.get("max_cached_blocks", 0))
            stats.update(collect_kv_cache_diagnostics(engine, config.block_size))
        except Exception as exc:  # noqa: BLE001
            max_blocks = 0
            record_failure(results, "memory", mode, exc)
        finally:
            if engine is not None:
                free_engine(engine)
        if baseline_blocks is None:
            baseline_blocks = max_blocks or 1
        if baseline_actual_block_bytes is None:
            baseline_actual_block_bytes = stats.get("actual_total_block_bytes") or stats["block_bytes"] or 1
        actual_total = stats.get("actual_total_block_bytes") or stats["block_bytes"]
        stats["max_cached_blocks"] = max_blocks
        stats["capacity_gain"] = max_blocks / baseline_blocks if baseline_blocks > 0 else 0.0
        stats["relative_to_auto_actual"] = actual_total / baseline_actual_block_bytes if baseline_actual_block_bytes else float("nan")
        results["memory"][mode] = stats


def run_latency_benchmark(config: BenchmarkConfig, results: dict[str, Any]) -> None:
    torch = require_torch()
    from myvllm.sampling_parameters import SamplingParams

    for mode in config.quant_modes:
        engine = None
        try:
            engine, tokenizer = load_engine(config, mode, max_model_length=config.latency_max_model_length)
            prompt = chat_prompt(tokenizer, gen_prompt_of_len(tokenizer, config.latency_prompt_len))
            params = SamplingParams(
                temperature=1e-6,
                max_tokens=config.latency_gen_tokens,
                max_model_length=config.latency_max_model_length,
            )
            warm_params = SamplingParams(temperature=1e-6, max_tokens=4, max_model_length=config.latency_max_model_length)
            engine.generate([prompt], warm_params)

            engine.add_prompt(prompt, params)
            torch.cuda.synchronize()
            start = time.time()
            engine.step()
            torch.cuda.synchronize()
            ttft_ms = (time.time() - start) * 1000

            decode_steps = 0
            decode_start = time.time()
            while not engine.scheduler.is_finished():
                engine.step()
                decode_steps += 1
            torch.cuda.synchronize()
            decode_s = time.time() - decode_start

            results["latency"][mode] = {
                "prompt_len": config.latency_prompt_len,
                "gen_tokens": config.latency_gen_tokens,
                "ttft_ms": ttft_ms,
                "tpot_ms": decode_s / decode_steps * 1000 if decode_steps > 0 else float("nan"),
                "decode_tokens_per_s": decode_steps / decode_s if decode_s > 0 else float("nan"),
                "decode_steps": decode_steps,
            }
        except Exception as exc:  # noqa: BLE001
            record_failure(results, "latency", mode, exc)
        finally:
            if engine is not None:
                free_engine(engine)


def run_concurrency_benchmark(config: BenchmarkConfig, results: dict[str, Any]) -> None:
    torch = require_torch()
    from myvllm.sampling_parameters import SamplingParams

    for mode in config.quant_modes:
        mode_result: dict[str, Any] = {"throughput_by_batch": {}}
        for batch_size in config.concurrency_batch_sizes:
            engine = None
            try:
                engine, tokenizer = load_engine(config, mode, max_model_length=config.concurrency_max_model_length)
                prompt = chat_prompt(tokenizer, gen_prompt_of_len(tokenizer, config.concurrency_prompt_len))
                prompts = [prompt] * batch_size
                params = SamplingParams(
                    temperature=1e-6,
                    max_tokens=config.concurrency_gen_tokens,
                    max_model_length=config.concurrency_max_model_length,
                )
                warm_params = SamplingParams(temperature=1e-6, max_tokens=4, max_model_length=config.concurrency_max_model_length)
                engine.generate([prompt], warm_params)

                torch.cuda.synchronize()
                start = time.time()
                output = engine.generate(prompts, params)
                torch.cuda.synchronize()
                elapsed_s = time.time() - start

                input_len = len(tokenizer.encode(prompt))
                total_gen_tokens = sum(max(0, len(ids) - input_len) for ids in output["token_ids"])
                if total_gen_tokens <= 0:
                    total_gen_tokens = batch_size * config.concurrency_gen_tokens
                mode_result["throughput_by_batch"][str(batch_size)] = total_gen_tokens / elapsed_s if elapsed_s > 0 else float("nan")
            except Exception as exc:  # noqa: BLE001
                mode_result["throughput_by_batch"][str(batch_size)] = float("nan")
                record_failure(results, "concurrency", f"{mode}:batch={batch_size}", exc)
            finally:
                if engine is not None:
                    free_engine(engine)

        first_batch = str(config.concurrency_batch_sizes[0])
        last_batch = str(config.concurrency_batch_sizes[-1])
        first_tps = mode_result["throughput_by_batch"].get(first_batch, float("nan"))
        last_tps = mode_result["throughput_by_batch"].get(last_batch, float("nan"))
        mode_result["scaleup"] = last_tps / first_tps if first_tps and not math.isnan(first_tps) else float("nan")
        results["concurrency"][mode] = mode_result


def token_agreement(base_ids: list[int], current_ids: list[int]) -> float:
    n = min(len(base_ids), len(current_ids))
    if n == 0:
        return 1.0
    matches = sum(1 for idx in range(n) if base_ids[idx] == current_ids[idx])
    return matches / n


def first_mismatch_index(base_ids: list[int], current_ids: list[int]) -> int | None:
    n = min(len(base_ids), len(current_ids))
    for idx in range(n):
        if base_ids[idx] != current_ids[idx]:
            return idx
    if len(base_ids) != len(current_ids):
        return n
    return None


def decode_token_slice(tokenizer: Any, ids: list[int], center: int | None, radius: int = 8) -> str:
    if center is None:
        return ""
    start = max(0, center - radius)
    end = min(len(ids), center + radius + 1)
    return tokenizer.decode(ids[start:end], skip_special_tokens=True)


def token_diff_detail(tokenizer: Any, prompt: str, baseline_ids: list[int], current_ids: list[int]) -> dict[str, Any]:
    mismatch = first_mismatch_index(baseline_ids, current_ids)
    baseline_token = baseline_ids[mismatch] if mismatch is not None and mismatch < len(baseline_ids) else None
    current_token = current_ids[mismatch] if mismatch is not None and mismatch < len(current_ids) else None
    return {
        "prompt": prompt,
        "agreement": token_agreement(baseline_ids, current_ids),
        "first_mismatch_index": mismatch,
        "baseline_token_id": baseline_token,
        "current_token_id": current_token,
        "baseline_text": tokenizer.decode(baseline_ids, skip_special_tokens=True),
        "current_text": tokenizer.decode(current_ids, skip_special_tokens=True),
        "baseline_prefix_around_mismatch": decode_token_slice(tokenizer, baseline_ids, mismatch),
        "current_prefix_around_mismatch": decode_token_slice(tokenizer, current_ids, mismatch),
    }


def run_accuracy_benchmark(config: BenchmarkConfig, results: dict[str, Any]) -> None:
    from myvllm.sampling_parameters import SamplingParams

    generated: dict[str, list[list[int]]] = {}
    tokenizer_for_decode = None
    for mode in config.quant_modes:
        engine = None
        try:
            engine, tokenizer = load_engine(config, mode, max_model_length=config.accuracy_max_model_length)
            tokenizer_for_decode = tokenizer
            prompts = [chat_prompt(tokenizer, prompt) for prompt in config.accuracy_prompts]
            params = SamplingParams(
                temperature=1e-6,
                max_tokens=config.accuracy_max_tokens,
                max_model_length=config.accuracy_max_model_length,
            )
            output = engine.generate(prompts, params)
            generated[mode] = output["token_ids"]
        except Exception as exc:  # noqa: BLE001
            record_failure(results, "accuracy", mode, exc)
        finally:
            if engine is not None:
                free_engine(engine)

    baseline = generated.get("auto")
    if not baseline or tokenizer_for_decode is None:
        return
    for mode in config.quant_modes:
        current = generated.get(mode)
        if not current:
            continue
        prompt_details = [
            token_diff_detail(tokenizer_for_decode, prompt, base_ids, cur_ids)
            for prompt, base_ids, cur_ids in zip(config.accuracy_prompts, baseline, current)
        ]
        prompt_rates = [detail["agreement"] for detail in prompt_details]
        avg = sum(prompt_rates) / len(prompt_rates) if prompt_rates else 1.0
        results["accuracy"][mode] = {
            "token_agreement_vs_fp16": avg,
            "prompt_agreements": prompt_rates,
            "num_prompts": len(prompt_rates),
            "details": prompt_details,
        }


def print_report(config: BenchmarkConfig, results: dict[str, Any]) -> None:
    print("\nKV Cache 量化 Benchmark")
    print(f"模型: {config.model_name_or_path}")
    print(f"格式: {QUANT_FORMAT}")

    if results.get("memory"):
        rows = []
        for mode, item in results["memory"].items():
            actual_kb = item.get("actual_total_block_bytes", item["block_bytes"]) / 1024
            rows.append([
                mode,
                item.get("actual_k_cache_dtype") or "unknown",
                f"{actual_kb:.1f}",
                f"{item.get('relative_to_auto_actual', float('nan')):.3f}x",
                item["max_cached_blocks"],
                f"{item['capacity_gain']:.2f}x",
            ])
        print_table(
            "Memory",
            ["模式", "实际K dtype", "实际block字节(KB)", "相对auto实际", "max_cached_blocks", "容量提升"],
            rows,
        )

    if results.get("latency"):
        rows = []
        for mode, item in results["latency"].items():
            rows.append([
                mode,
                f"{item['ttft_ms']:.1f}",
                f"{item['tpot_ms']:.2f}",
                f"{item['decode_tokens_per_s']:.1f}",
            ])
        print_table("Latency / Throughput", ["模式", "TTFT(ms)", "TPOT(ms)", "Decode吞吐(tok/s)"], rows)

    if results.get("concurrency"):
        headers = ["模式"] + [f"batch={bs} tok/s" for bs in config.concurrency_batch_sizes] + ["扩展比"]
        rows = []
        for mode, item in results["concurrency"].items():
            throughput = item["throughput_by_batch"]
            rows.append(
                [mode]
                + [f"{throughput.get(str(bs), float('nan')):.1f}" for bs in config.concurrency_batch_sizes]
                + [f"{item['scaleup']:.2f}x"]
            )
        print_table("Concurrency", headers, rows)

    if results.get("accuracy"):
        rows = []
        for mode, item in results["accuracy"].items():
            rate = item["token_agreement_vs_fp16"]
            if mode == "auto":
                note = "fp16 基线"
            elif rate >= 0.95:
                note = "精度几乎无损"
            elif rate >= 0.8:
                note = "轻微偏离"
            else:
                note = "精度显著退化"
            rows.append([mode, f"{rate:.3f}", note])
        print_table("Accuracy", ["模式", "token一致率(vs fp16)", "解读"], rows)

        detail_rows = []
        for mode, item in results["accuracy"].items():
            for prompt_id, detail in enumerate(item.get("details", [])):
                detail_rows.append([
                    mode,
                    prompt_id,
                    f"{detail['agreement']:.3f}",
                    detail["first_mismatch_index"] if detail["first_mismatch_index"] is not None else "一致",
                    detail["baseline_token_id"] if detail["baseline_token_id"] is not None else "-",
                    detail["current_token_id"] if detail["current_token_id"] is not None else "-",
                ])
        if detail_rows:
            print_table(
                "Accuracy Details",
                ["模式", "prompt", "agreement", "first_mismatch", "base_token", "cur_token"],
                detail_rows,
            )

    if results.get("failures"):
        rows = [[f["suite"], f["mode"], f["error"]] for f in results["failures"]]
        print_table("Failures", ["suite", "mode", "error"], rows)


def plot_results(config: BenchmarkConfig, results: dict[str, Any]) -> None:
    if not has_matplotlib():
        return

    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    modes = list(config.quant_modes)

    if results.get("memory"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(f"Memory ({QUANT_FORMAT})", fontsize=12)
        vals_kb = [results["memory"].get(m, {}).get("block_bytes", 0) / 1024 for m in modes]
        blocks = [results["memory"].get(m, {}).get("max_cached_blocks", 0) for m in modes]
        colors = [MODE_COLORS[m] for m in modes]
        ax1.bar(modes, vals_kb, color=colors)
        ax1.set_title("KV cache bytes per block (lower=better)")
        ax1.set_ylabel("KB / block")
        ax2.bar(modes, blocks, color=colors)
        ax2.set_title("Max KV blocks under same VRAM (higher=better)")
        ax2.set_ylabel("max_cached_blocks")
        for ax, vals in ((ax1, vals_kb), (ax2, blocks)):
            for i, value in enumerate(vals):
                ax.text(i, value, f"{value:.0f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(config.output_dir / "memory.png", dpi=120)
        plt.close(fig)

    if results.get("latency"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(f"Latency / Throughput ({QUANT_FORMAT})", fontsize=12)
        metrics = [
            ("ttft_ms", "TTFT (lower=better)", "ms"),
            ("tpot_ms", "TPOT (lower=better)", "ms / token"),
            ("decode_tokens_per_s", "Decode throughput (higher=better)", "tokens / s"),
        ]
        for ax, (key, title, ylabel) in zip(axes, metrics):
            vals = [results["latency"].get(m, {}).get(key, 0) for m in modes]
            ax.bar(modes, vals, color=[MODE_COLORS[m] for m in modes])
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            for i, value in enumerate(vals):
                ax.text(i, value, f"{value:.1f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(config.output_dir / "latency_throughput.png", dpi=120)
        plt.close(fig)

    if results.get("concurrency"):
        fig, ax = plt.subplots(figsize=(8, 5))
        batch_sizes = list(config.concurrency_batch_sizes)
        for mode in modes:
            item = results["concurrency"].get(mode)
            if not item:
                continue
            ys = [item["throughput_by_batch"].get(str(bs), float("nan")) for bs in batch_sizes]
            ax.plot(batch_sizes, ys, marker="o", color=MODE_COLORS[mode], label=mode)
        ax.set_xlabel("batch size (concurrent requests)")
        ax.set_ylabel("end-to-end throughput (tokens / s)")
        ax.set_title(f"Concurrent throughput vs batch size ({QUANT_FORMAT})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(config.output_dir / "concurrency_throughput.png", dpi=120)
        plt.close(fig)

    if results.get("accuracy"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        vals = [results["accuracy"].get(m, {}).get("token_agreement_vs_fp16", 0) for m in modes]
        ax.bar(modes, vals, color=[MODE_COLORS[m] for m in modes])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("token agreement vs fp16")
        ax.set_title("Generation accuracy (higher=better)")
        for i, value in enumerate(vals):
            ax.text(i, value + 0.01, f"{value:.3f}", ha="center")
        fig.tight_layout()
        fig.savefig(config.output_dir / "accuracy.png", dpi=120)
        plt.close(fig)

    plot_summary(config, results)


def plot_summary(config: BenchmarkConfig, results: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    modes = list(config.quant_modes)
    if not any(results.get(key) for key in ("memory", "latency", "accuracy", "concurrency")):
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"KV Cache Quantization Benchmark ({config.model_name_or_path}, {QUANT_FORMAT})", fontsize=14)
    colors = [MODE_COLORS[m] for m in modes]

    ax = axes[0][0]
    vals = [results.get("memory", {}).get(m, {}).get("max_cached_blocks", 0) for m in modes]
    ax.bar(modes, vals, color=colors)
    base = vals[0] or 1
    for i, value in enumerate(vals):
        ax.text(i, value, f"{value}\n({value / base:.2f}x)", ha="center", va="bottom")
    ax.set_title("Memory: max KV blocks (higher=better)")
    ax.set_ylabel("max_cached_blocks")

    ax = axes[0][1]
    vals = [results.get("latency", {}).get(m, {}).get("ttft_ms", 0) for m in modes]
    ax.bar(modes, vals, color=colors)
    for i, value in enumerate(vals):
        ax.text(i, value, f"{value:.0f}ms", ha="center", va="bottom")
    ax.set_title("Latency: TTFT (lower=better)")
    ax.set_ylabel("ms")

    ax = axes[1][0]
    vals = [results.get("latency", {}).get(m, {}).get("decode_tokens_per_s", 0) for m in modes]
    ax.bar(modes, vals, color=colors)
    for i, value in enumerate(vals):
        ax.text(i, value, f"{value:.0f}", ha="center", va="bottom")
    ax.set_title("Throughput: decode tokens/s (higher=better)")
    ax.set_ylabel("tokens / s")

    ax = axes[1][1]
    vals = [results.get("accuracy", {}).get(m, {}).get("token_agreement_vs_fp16", 0) for m in modes]
    ax.bar(modes, vals, color=colors)
    ax.set_ylim(0, 1.1)
    for i, value in enumerate(vals):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center")
    ax.set_title("Accuracy: token agreement vs fp16 (higher=better)")
    ax.set_ylabel("agreement")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(config.output_dir / "summary.png", dpi=120)
    plt.close(fig)


def parse_suites(raw: str) -> set[str]:
    valid = {"memory", "latency", "concurrency", "accuracy"}
    if raw.strip().lower() == "all":
        return valid
    selected = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = selected - valid
    if unknown:
        raise ValueError(f"未知 suite: {sorted(unknown)}; 可选: {sorted(valid)}")
    return selected


def run_selected_suites(config: BenchmarkConfig, selected: set[str]) -> dict[str, Any]:
    results = empty_results(config)
    if "memory" in selected:
        run_memory_benchmark(config, results)
    if "latency" in selected:
        run_latency_benchmark(config, results)
    if "concurrency" in selected:
        run_concurrency_benchmark(config, results)
    if "accuracy" in selected:
        run_accuracy_benchmark(config, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="KV Cache 量化 Benchmark")
    parser.add_argument("--suite", default="all", help="评测模块，逗号分隔: memory,latency,concurrency,accuracy 或 all")
    parser.add_argument("--output-dir", default=str(BenchmarkConfig.output_dir), help="结果输出目录")
    parser.add_argument("--no-plots", action="store_true", help="跳过 PNG 图输出")
    args = parser.parse_args()

    torch = require_torch()
    assert torch.cuda.is_available(), "本 benchmark 需要 CUDA GPU"

    config = BenchmarkConfig(output_dir=Path(args.output_dir))
    selected = parse_suites(args.suite)

    print(f"运行评测模块: {sorted(selected)}")
    print(f"结果输出目录: {config.output_dir}")

    results = run_selected_suites(config, selected)
    print_report(config, results)
    json_path = save_json_report(results, config.output_dir)
    print(f"\nJSON 结果已保存: {json_path}")
    if not args.no_plots:
        plot_results(config, results)
        print(f"PNG 图已保存到: {config.output_dir}")
    print("完成。")


if __name__ == "__main__":
    main()
