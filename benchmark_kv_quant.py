"""
KV Cache FP8 量化 Benchmark 
================================================

只保留推理服务领域 (vLLM / TensorRT-LLM / SGLang) 标准评测三大维度：

  1. Memory         显存：每 block KV 字节数 + max_cached_blocks 容量
  2. Latency/Throughput  延迟+吞吐：TTFT / TPOT / decode tokens/s
  3. Accuracy       精度：与 fp16 基线的 token 一致率（一句话证明精度无损）

所有结果【表格 + 图】同时输出到 benchmark_results/。
统一模型: Qwen3-0.6B

用法:
    python benchmark_kv_quant.py                # 全部
    python benchmark_kv_quant.py --suite memory # 仅显存
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

# matplotlib 可选；缺失则降级为仅文本表格
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception as _e:  # noqa: BLE001
    _HAS_MPL = False
    print(f"[警告] 未检测到 matplotlib，将仅输出文本表格。原因: {_e}")

RESULT_DIR = Path(__file__).parent / "benchmark_results"
RESULT_DIR.mkdir(exist_ok=True)

# 待对比的三种 KV cache 模式
QUANT_MODES = ["auto", "fp8_per_tensor", "fp8_per_token_head"]
# 用于柱状图配色，与各模块保持一致 (auto=灰, per_tensor=粉, per_token_head=青)
MODE_COLORS = ["#888", "#e8a", "#5a9"]
# FP8 数据格式标注（所有 FP8 模式统一使用 e4m3fn，数值上限 ±448）
FP8_FORMAT = "FP8 E4M3FN (max=448)"


# ======================================================================
# 通用工具：表格打印 + 保存 (md/csv 同步)
# ======================================================================
def render_table(headers, rows):
    """渲染等宽 ASCII 表格字符串（终端打印用）。"""
    cols = len(headers)
    widths = [len(str(headers[c])) for c in range(cols)]
    for row in rows:
        for c in range(cols):
            widths[c] = max(widths[c], len(str(row[c])))
    def fmt_row(cells):
        return "| " + " | ".join(str(cells[c]).ljust(widths[c]) for c in range(cols)) + " |"
    sep = "|" + "|".join("-" * (widths[c] + 2) for c in range(cols)) + "|"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])


def save_table(name, headers, rows):
    """打印 + 保存 csv（不保存 md，简化输出）。"""
    print(render_table(headers, rows))
    with open(RESULT_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  -> 表格已保存: {name}.csv")


# ======================================================================
# 引擎加载 / 释放工具
# ======================================================================
def build_config(kv_cache_dtype, max_model_length=1024):
    """Qwen3-0.6B 推理引擎配置，注入指定的 kv_cache_dtype。"""
    return {
        'max_num_sequences': 8,
        'max_num_batched_tokens': 8192,
        'max_cached_blocks': 2048,
        'block_size': 256,
        'world_size': 1,
        'model_name_or_path': 'Qwen/Qwen3-0.6B',
        'enforce_eager': True,
        'vocab_size': 151936,
        'hidden_size': 1024,
        'num_heads': 16,
        'head_dim': 128,
        'num_kv_heads': 8,
        'intermediate_size': 3072,
        'num_layers': 28,
        'tie_word_embeddings': True,
        'base': 1000000,
        'rms_norm_epsilon': 1e-6,
        'qkv_bias': False,
        'scale': 1,
        'max_position': 32768,
        'ffn_bias': False,
        'max_num_batch_tokens': 8192,
        'max_model_length': max_model_length,
        'gpu_memory_utilization': 0.85,
        'eos': 151645,
        'kv_cache_dtype': kv_cache_dtype,
    }


def load_engine(kv_cache_dtype, max_model_length=1024):
    """加载引擎前先彻底清理显存，避免顺序加载时误判可用显存。"""
    import gc
    from transformers import AutoTokenizer
    from myvllm.engine.llm_engine import LLMEngine as LLM

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    config = build_config(kv_cache_dtype, max_model_length)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
    engine = LLM(config=config)
    return engine, tokenizer


def free_engine(engine):
    """彻底释放引擎占用的显存（权重 + KV cache + scale 张量）。"""
    import gc
    # 先删模型强引用，再 exit()
    try:
        mr = getattr(engine, "model_runner", None)
        if mr is not None and hasattr(mr, "model"):
            del mr.model
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


def chat_prompt(tokenizer, user_text):
    """Qwen3 chat 模板封装。"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _gen_prompt_of_len(tokenizer, n_tokens):
    """用重复填充文本构造约 n_tokens 长的英文 prompt。"""
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "Paris is a city with a long history of art and culture. "
        "Machine learning models process large amounts of data. "
    )
    ids = tokenizer.encode(filler * max(1, n_tokens // 8))[:n_tokens]
    return tokenizer.decode(ids)


# ======================================================================
# 模块 1: 显存 (Memory)
# ======================================================================
def run_memory_benchmark():
    """
    显存评测 - 业界标准指标:
      - 每 block KV cache 字节数（理论值，固定结论）
      - max_cached_blocks: 相同显存下能容纳的 KV 块数 (= 实际可服务容量)

    KV cache 量化的核心价值就是显存：fp8 单元素 1 字节 (vs fp16 的 2)，
    在同样显存预算下能装更多 KV 块 -> 支持更长上下文 / 更大并发。
    """
    print("\n" + "=" * 70)
    print(f"模块 1: 显存 (Memory)   [{FP8_FORMAT}]")
    print("=" * 70)

    num_layers, num_kv_heads, head_dim, block_size = 28, 8, 128, 256
    headers = ["模式", "每block字节(KB)", "相对fp16", "max_cached_blocks", "容量提升"]
    rows = []
    blocks_by_mode = {}
    bytes_by_mode = {}

    baseline_blocks = None
    for mode in QUANT_MODES:
        # 理论字节数: data + (per_token_head 时) scale
        elem_bytes = 1 if mode != "auto" else 2
        data_bytes = block_size * 2 * num_layers * num_kv_heads * head_dim * elem_bytes
        scale_bytes = (block_size * 2 * num_layers * num_kv_heads * 4
                       if mode == "fp8_per_token_head" else 0)
        total_bytes = data_bytes + scale_bytes
        bytes_by_mode[mode] = total_bytes

        # 实测: 加载引擎拿到 max_cached_blocks (引擎按可用显存动态算)
        try:
            engine, _ = load_engine(mode, max_model_length=1024)
            blocks = engine.config.get('max_cached_blocks', 0)
            free_engine(engine)
        except Exception as e:  # noqa: BLE001
            blocks = 0
            print(f"  [警告] 加载 {mode} 失败: {e}")
        blocks_by_mode[mode] = blocks
        if baseline_blocks is None:
            baseline_blocks = blocks or 1

        rel_bytes = total_bytes / (block_size * 2 * num_layers * num_kv_heads * head_dim * 2)
        cap_gain = blocks / baseline_blocks if baseline_blocks > 0 else 0
        rows.append([mode, f"{total_bytes/1024:.1f}", f"{rel_bytes:.3f}x",
                     str(blocks), f"{cap_gain:.2f}x"])

    save_table("memory", headers, rows)

    if _HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(f"Memory ({FP8_FORMAT})", fontsize=12)
        modes = list(blocks_by_mode.keys())
        # 左: 每 block KV 字节 (越小越省)
        vals_kb = [bytes_by_mode[m] / 1024 for m in modes]
        ax1.bar(modes, vals_kb, color=MODE_COLORS)
        ax1.set_title("KV cache bytes per block (lower=better)")
        ax1.set_ylabel("KB / block")
        for i, v in enumerate(vals_kb):
            ax1.text(i, v, f"{v:.0f}", ha="center", va="bottom")
        # 右: max_cached_blocks (越大容量越好)
        vals_b = [blocks_by_mode[m] for m in modes]
        ax2.bar(modes, vals_b, color=MODE_COLORS)
        ax2.set_title("Max KV blocks under same VRAM (higher=better)")
        ax2.set_ylabel("max_cached_blocks")
        for i, v in enumerate(vals_b):
            ax2.text(i, v, str(v), ha="center", va="bottom")
        out = RESULT_DIR / "memory.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  -> 图已保存: {out.name}")

    return {
        "bytes": bytes_by_mode,
        "blocks": blocks_by_mode,
    }


# ======================================================================
# 模块 2: 延迟 + 吞吐 (Latency / Throughput)
# ======================================================================
def run_latency_throughput_benchmark():
    """
    业界标准延迟+吞吐评测:
      - TTFT (Time To First Token): prefill 完成、产出第一个 token 的延迟
      - TPOT (Time Per Output Token): decode 阶段平均每 token 延迟
      - Decode throughput (tokens/s): 解码阶段总吞吐
    固定 prompt 长度与生成 token 数，公平对比三种模式。
    """
    print("\n" + "=" * 70)
    print(f"模块 2: 延迟 + 吞吐 (TTFT / TPOT / Throughput)   [{FP8_FORMAT}]")
    print("=" * 70)

    from myvllm.sampling_parameters import SamplingParams

    prompt_len = 256
    gen_tokens = 64
    sp = SamplingParams(temperature=1e-6, max_tokens=gen_tokens, max_model_length=1024)

    headers = ["模式", "TTFT(ms)", "TPOT(ms)", "Decode吞吐(tok/s)"]
    rows = []
    detail = {}

    for mode in QUANT_MODES:
        engine, tokenizer = load_engine(mode, max_model_length=1024)
        prompt = chat_prompt(tokenizer, _gen_prompt_of_len(tokenizer, prompt_len))

        # 预热一次 (排除首次 triton kernel 编译开销)
        warm_sp = SamplingParams(temperature=1e-6, max_tokens=4, max_model_length=1024)
        engine.generate([prompt], warm_sp)

        # 真正计时
        engine.add_prompt(prompt, sp)
        torch.cuda.synchronize()
        t0 = time.time()
        # 第一次 step = prefill -> 产出首 token => TTFT
        engine.step()
        torch.cuda.synchronize()
        ttft_ms = (time.time() - t0) * 1000

        # 其余 step = decode
        decode_steps = 0
        t1 = time.time()
        while not engine.scheduler.is_finished():
            engine.step()
            decode_steps += 1
        torch.cuda.synchronize()
        decode_time = time.time() - t1
        tpot_ms = (decode_time / decode_steps * 1000) if decode_steps > 0 else float("nan")
        tps = (decode_steps / decode_time) if decode_time > 0 else float("nan")

        rows.append([mode, f"{ttft_ms:.1f}", f"{tpot_ms:.2f}", f"{tps:.1f}"])
        detail[mode] = {"ttft": ttft_ms, "tpot": tpot_ms, "tps": tps}
        free_engine(engine)

    save_table("latency_throughput", headers, rows)

    if _HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(f"Latency / Throughput ({FP8_FORMAT})", fontsize=12)
        modes = list(detail.keys())
        for ax, key, title, ylabel, lower_better in [
            (axes[0], "ttft", "TTFT (lower=better)", "ms", True),
            (axes[1], "tpot", "TPOT (lower=better)", "ms / token", True),
            (axes[2], "tps", "Decode throughput (higher=better)", "tokens / s", False),
        ]:
            vals = [detail[m][key] for m in modes]
            ax.bar(modes, vals, color=MODE_COLORS)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
        out = RESULT_DIR / "latency_throughput.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  -> 图已保存: {out.name}")

    return detail


# ======================================================================
# 模块 3: 精度 (Accuracy)
# ======================================================================
def run_accuracy_benchmark():
    """
    精度评测: 与 fp16(auto) 基线对比生成 token 序列的一致率。
    一致率越高 = 量化对模型行为破坏越小。这是简历上证明"量化无损"的核心数据。
    """
    print("\n" + "=" * 70)
    print(f"模块 3: 精度 (token 一致率 vs fp16)   [{FP8_FORMAT}]")
    print("=" * 70)

    from myvllm.sampling_parameters import SamplingParams

    # 4 个有代表性的 prompt: 中英文 + 知识/推理/事实/常识
    prompts_raw = [
        "Explain what a transformer is in one sentence.",
        "List the first 10 prime numbers.",
        "What is the capital of France?",
        "用一句话解释什么是梯度下降。",
    ]
    sp = SamplingParams(temperature=1e-6, max_tokens=48, max_model_length=512)

    results = {}
    for mode in QUANT_MODES:
        engine, tokenizer = load_engine(mode, max_model_length=512)
        prompts = [chat_prompt(tokenizer, p) for p in prompts_raw]
        out = engine.generate(prompts, sp)
        results[mode] = out['token_ids']
        free_engine(engine)

    baseline = results["auto"]
    headers = ["模式", "token一致率 (vs fp16)", "解读"]
    rows = []
    agreement = {}
    for mode in QUANT_MODES:
        rates = []
        for base_ids, cur_ids in zip(baseline, results[mode]):
            n = min(len(base_ids), len(cur_ids))
            if n == 0:
                rates.append(1.0)
                continue
            match = sum(1 for j in range(n) if base_ids[j] == cur_ids[j])
            rates.append(match / n)
        avg = sum(rates) / len(rates) if rates else 1.0
        agreement[mode] = avg
        if mode == "auto":
            note = "fp16 基线"
        elif avg >= 0.95:
            note = "精度几乎无损"
        elif avg >= 0.8:
            note = "轻微偏离"
        else:
            note = "精度显著退化"
        rows.append([mode, f"{avg:.3f}", note])
    save_table("accuracy", headers, rows)

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        modes = list(agreement.keys())
        vals = [agreement[m] for m in modes]
        ax.bar(modes, vals, color=MODE_COLORS)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("token agreement vs fp16")
        ax.set_title(f"Accuracy: token agreement vs fp16 (higher=better) [{FP8_FORMAT}]")
        ax.set_title("Generation accuracy (higher=better)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
        out = RESULT_DIR / "accuracy.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  -> 图已保存: {out.name}")

    return agreement


# ======================================================================
# 综合总览图 (一张图涵盖 显存/延迟/吞吐/精度 四个核心指标)
# ======================================================================
def plot_summary(mem_res, lat_res, acc_res):
    """业界标准三件套的一图总览。简历配图直接用这张。"""
    if not _HAS_MPL or not (mem_res and lat_res and acc_res):
        print("[提示] 跳过综合图（缺数据或无 matplotlib）")
        return

    modes = QUANT_MODES
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"KV Cache FP8 Quantization Benchmark (Qwen3-0.6B, {FP8_FORMAT})", fontsize=14)

    # 左上: 容量 (max_cached_blocks, 越高越好)
    ax = axes[0][0]
    vals = [mem_res["blocks"].get(m, 0) for m in modes]
    ax.bar(modes, vals, color=MODE_COLORS)
    base = vals[0] or 1
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}\n({v/base:.2f}x)", ha="center", va="bottom")
    ax.set_title("Memory: max KV blocks under same VRAM (higher=better)")
    ax.set_ylabel("max_cached_blocks")

    # 右上: TTFT (越低越好)
    ax = axes[0][1]
    vals = [lat_res[m]["ttft"] for m in modes]
    ax.bar(modes, vals, color=MODE_COLORS)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}ms", ha="center", va="bottom")
    ax.set_title("Latency: TTFT (lower=better)")
    ax.set_ylabel("ms")

    # 左下: Decode 吞吐 (越高越好)
    ax = axes[1][0]
    vals = [lat_res[m]["tps"] for m in modes]
    ax.bar(modes, vals, color=MODE_COLORS)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    ax.set_title("Throughput: decode tokens/s (higher=better)")
    ax.set_ylabel("tokens / s")

    # 右下: 精度 (与 fp16 一致率, 越高越好)
    ax = axes[1][1]
    vals = [acc_res.get(m, 0) for m in modes]
    ax.bar(modes, vals, color=MODE_COLORS)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1.1)
    ax.set_title("Accuracy: token agreement vs fp16 (higher=better)")
    ax.set_ylabel("agreement")

    out = RESULT_DIR / "summary.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n[综合总览图] 已保存: {out}")


# ======================================================================
# 主入口
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="KV Cache FP8 量化 Benchmark (精简版)")
    parser.add_argument(
        "--suite", default="all",
        help="评测模块, 逗号分隔: memory,latency,accuracy 或 all",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "本 benchmark 需要 CUDA GPU"

    suite = args.suite.strip().lower()
    if suite == "all":
        selected = {"memory", "latency", "accuracy"}
    else:
        selected = {s.strip() for s in suite.split(",")}

    print(f"运行评测模块: {sorted(selected)}")
    print(f"结果输出目录: {RESULT_DIR}")

    mem_res = lat_res = acc_res = None
    if "memory" in selected:
        mem_res = run_memory_benchmark()
    if "latency" in selected:
        lat_res = run_latency_throughput_benchmark()
    if "accuracy" in selected:
        acc_res = run_accuracy_benchmark()

    plot_summary(mem_res, lat_res, acc_res)

    print("\n完成。所有表格(.md/.csv)与图(.png)见 benchmark_results/")


if __name__ == "__main__":
    main()
