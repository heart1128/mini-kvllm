"""
KV Cache FP8 量化 Benchmark
================================

本脚本对 MinivLLM 的 KV cache FP8 量化进行多元化评测，目标是同时体现：
  - 量化的【优点】：显存占用下降
  - 量化的【精度问题】：数值误差、长上下文检索能力下降

评测分为四大模块：
  1. 数值精度微基准 (numeric)：直接对随机/含离群值张量做量化->反量化往返，
     对比 per_tensor 与 per_token_head 的 MSE / 最大误差 / 余弦相似度 / SNR。
  2. 端到端精度对比 (e2e)：在 Qwen3-0.6B 上 greedy decode，对比 fp16 基线与两种
     FP8 模式的 top-1 一致率与首次分歧位置。
  3. 性能 / 显存 (perf)：对比三种模式 KV cache 的显存占用。
  4. NIAH 测试 (needle-in-a-haystack)：长上下文检索压测，最能暴露量化精度损失。

所有评测【同时输出表格和图】：
  - 表格：终端打印 + 保存 .md / .csv 到 benchmark_results/
  - 图：matplotlib 保存 .png 到 benchmark_results/（缺失 matplotlib 时降级为纯文本）

统一使用模型：Qwen3-0.6B

用法：
    python benchmark_kv_quant.py --suite all          # 运行全部
    python benchmark_kv_quant.py --suite numeric       # 仅数值微基准（不需要加载模型）
    python benchmark_kv_quant.py --suite e2e,niah      # 指定多个
"""

import os
import sys
import csv
import math
import time
import argparse
from pathlib import Path

import torch

# 将 src 加入 import 路径，复用项目内的量化 kernel
sys.path.insert(0, str(Path(__file__).parent / "src"))

# matplotlib 为可选依赖：缺失时降级为纯文本输出，不中断运行
try:
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境下使用 Agg 后端
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception as _e:  # noqa: BLE001
    _HAS_MPL = False
    print(f"[警告] 未检测到 matplotlib，将仅输出文本表格（图表跳过）。原因: {_e}")

# 评测结果输出目录
RESULT_DIR = Path(__file__).parent / "benchmark_results"
RESULT_DIR.mkdir(exist_ok=True)

# FP8 e4m3fn 的数值上限（量化 scale = amax / 448）
FP8_E4M3_MAX = 448.0
# 三种待对比的量化模式
QUANT_MODES = ["auto", "fp8_per_tensor", "fp8_per_token_head"]


# ======================================================================
# 通用工具：表格打印与保存（同时输出 .md 与 .csv）
# ======================================================================
def render_table(headers, rows):
    """把表头和数据行渲染成等宽 ASCII 表格字符串（用于终端打印）。"""
    # 计算每列宽度：取表头与该列所有单元格的最大字符数
    cols = len(headers)
    widths = [len(str(headers[c])) for c in range(cols)]
    for row in rows:
        for c in range(cols):
            widths[c] = max(widths[c], len(str(row[c])))
    # 构造分隔线与每一行
    def fmt_row(cells):
        return "| " + " | ".join(str(cells[c]).ljust(widths[c]) for c in range(cols)) + " |"
    sep = "|" + "|".join("-" * (widths[c] + 2) for c in range(cols)) + "|"
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


def save_table(name, headers, rows):
    """把表格同时打印到终端，并保存为 markdown 与 csv 两种格式。"""
    table_str = render_table(headers, rows)
    print(table_str)
    # 保存 markdown
    md_path = RESULT_DIR / f"{name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(map(str, headers)) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(map(str, r)) + " |\n")
    # 保存 csv
    csv_path = RESULT_DIR / f"{name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  -> 表格已保存: {md_path.name}, {csv_path.name}")


# ======================================================================
# 模块 1：数值精度微基准（kernel 级，不依赖完整模型）
# ======================================================================
def quant_dequant_per_tensor(x):
    """per_tensor 量化->反量化往返。整个张量共享一个标量 scale。"""
    # scale 由全局最大绝对值决定
    amax = x.abs().amax().clamp(min=1e-8)
    scale = amax / FP8_E4M3_MAX
    # 量化：除以 scale 后裁剪到 FP8 范围再转 float8
    q = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    # 反量化：转回 float 再乘 scale
    return q.to(torch.float32) * scale


def quant_dequant_per_token_head(x):
    """
    per_token_head 量化->反量化往返。
    x 形状: (num_tokens, num_kv_heads, head_dim)
    对每个 (token, head) 的 head_dim 个元素独立计算 scale。
    """
    # 在 head_dim 维度上求每个 (token, head) 的最大绝对值
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # (T, H, 1)
    scale = amax / FP8_E4M3_MAX
    q = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q.to(torch.float32) * scale


def compute_error_metrics(orig, recon):
    """计算量化误差指标：MSE、最大绝对误差、余弦相似度、SNR(dB)。"""
    orig_f = orig.float().flatten()
    recon_f = recon.float().flatten()
    mse = torch.mean((orig_f - recon_f) ** 2).item()
    max_err = (orig_f - recon_f).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(orig_f, recon_f, dim=0).item()
    # 信噪比：信号功率 / 噪声功率，转分贝
    signal_power = torch.mean(orig_f ** 2).item()
    noise_power = mse + 1e-12
    snr_db = 10 * math.log10(signal_power / noise_power + 1e-12)
    return mse, max_err, cos, snr_db


def run_numeric_benchmark(device):
    """
    数值微基准：覆盖多种数据分布，体现 per_tensor 在离群值下误差大、
    per_token_head 更鲁棒的特性。
    """
    print("\n" + "=" * 70)
    print("模块 1: 数值精度微基准 (量化->反量化往返误差)")
    print("=" * 70)

    torch.manual_seed(0)
    T, H, D = 512, 8, 128  # tokens, kv_heads, head_dim

    # 构造多种数据分布
    distributions = {}
    # (a) 标准正态分布：理想情况
    distributions["normal"] = torch.randn(T, H, D, device=device)
    # (b) 含离群值：1% 的元素放大 50 倍，模拟激活离群值场景
    outlier = torch.randn(T, H, D, device=device)
    mask = torch.rand(T, H, D, device=device) < 0.01
    outlier[mask] *= 50.0
    distributions["outlier_1pct"] = outlier
    # (c) 重尾分布（拉普拉斯）：更接近真实 KV 分布
    laplace = torch.distributions.Laplace(0.0, 1.0).sample((T, H, D)).to(device)
    distributions["heavy_tail"] = laplace

    headers = ["分布", "模式", "MSE", "最大误差", "余弦相似度", "SNR(dB)"]
    rows = []
    # 用于画图：记录每个分布每种模式的 SNR
    plot_data = {m: [] for m in ["per_tensor", "per_token_head"]}
    dist_names = list(distributions.keys())

    for dist_name, x in distributions.items():
        for mode_name, fn in [
            ("per_tensor", quant_dequant_per_tensor),
            ("per_token_head", quant_dequant_per_token_head),
        ]:
            recon = fn(x)
            mse, max_err, cos, snr = compute_error_metrics(x, recon)
            rows.append([
                dist_name, mode_name,
                f"{mse:.3e}", f"{max_err:.4f}", f"{cos:.5f}", f"{snr:.2f}",
            ])
            plot_data[mode_name].append(snr)

    save_table("numeric_micro_benchmark", headers, rows)

    # 画图：分组柱状图 SNR 对比（越高越好）
    if _HAS_MPL:
        x_pos = range(len(dist_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([p - width / 2 for p in x_pos], plot_data["per_tensor"],
               width, label="per_tensor")
        ax.bar([p + width / 2 for p in x_pos], plot_data["per_token_head"],
               width, label="per_token_head")
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(dist_names)
        ax.set_ylabel("SNR (dB, higher is better)")
        ax.set_title("FP8 KV quant SNR by distribution")
        ax.legend()
        out = RESULT_DIR / "numeric_snr.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  -> 图已保存: {out.name}")

    # 返回数据供综合图使用：{dist_name: {mode: snr}}
    numeric_result = {dist_names[i]: {
        "per_tensor": plot_data["per_tensor"][i],
        "per_token_head": plot_data["per_token_head"][i],
    } for i in range(len(dist_names))}
    return numeric_result


# ======================================================================
# 端到端 / NIAH 共用：构建 LLM 引擎
# ======================================================================
def build_config(kv_cache_dtype, max_model_length=1024):
    """构造 Qwen3-0.6B 的引擎配置，注入指定的 kv_cache_dtype。"""
    return {
        'max_num_sequences': 8,
        'max_num_batched_tokens': 8192,
        'max_cached_blocks': 2048,
        'block_size': 256,
        'world_size': 1,
        'model_name_or_path': 'Qwen/Qwen3-0.6B',
        'enforce_eager': True,  # 量化路径未捕获 CUDA graph，使用 eager 模式
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
        'gpu_memory_utilization': 0.5,
        'eos': 151645,
        # 关键：量化模式
        'kv_cache_dtype': kv_cache_dtype,
    }


def load_engine(kv_cache_dtype, max_model_length=1024):
    """加载一个指定量化模式的 LLM 引擎，返回 (engine, tokenizer)。"""
    from transformers import AutoTokenizer
    from myvllm.engine.llm_engine import LLMEngine as LLM

    # 顺序加载多个引擎时，上一个的显存可能尚未完全回收。
    # 先做一次彻底清理 + 同步，避免 gpu_memory_utilization 误判可用显存
    # 导致后续引擎 "Not enough memory to hold at least one block"。
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    config = build_config(kv_cache_dtype, max_model_length)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
    engine = LLM(config=config)
    return engine, tokenizer


def free_engine(engine):
    """彻底释放引擎及其占用的显存（权重 + KV cache），供顺序评测复用显存。"""
    import gc
    # 先删模型与 KV cache 张量的强引用（必须在 exit() 删除 model_runner 之前）
    try:
        mr = getattr(engine, "model_runner", None)
        if mr is not None:
            if hasattr(mr, "model"):
                del mr.model
    except Exception:  # noqa: BLE001
        pass
    # 再正常退出（销毁进程组等）
    try:
        engine.exit()
    except Exception:  # noqa: BLE001
        pass
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def chat_prompt(tokenizer, user_text):
    """套用 Qwen3 chat 模板，生成可直接喂给模型的 prompt 字符串。"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ======================================================================
# 模块 2：端到端精度对比
# ======================================================================
def run_e2e_benchmark():
    """
    端到端精度：以 fp16(auto) 为基线，对比两种 FP8 模式 greedy decode 的输出。
    指标：top-1 token 一致率、首次分歧位置（divergence point）。
    """
    print("\n" + "=" * 70)
    print("模块 2: 端到端精度对比 (Qwen3-0.6B, greedy decode)")
    print("=" * 70)

    from myvllm.sampling_parameters import SamplingParams

    prompts_raw = [
        "Explain what a transformer model is in one paragraph.",
        "List the first 10 prime numbers.",
        "What is the capital of France and why is it famous?",
    ]
    # 近似 greedy: 项目禁止 temperature=0，用极低温度逼近贪心解码
    sp = SamplingParams(temperature=1e-6, max_tokens=64, max_model_length=1024)

    # 先用基线 (auto) 跑一遍，得到参考 token 序列
    results_by_mode = {}
    for mode in QUANT_MODES:
        engine, tokenizer = load_engine(mode, max_model_length=1024)
        prompts = [chat_prompt(tokenizer, p) for p in prompts_raw]
        out = engine.generate(prompts, sp)
        results_by_mode[mode] = out['token_ids']
        free_engine(engine)

    baseline = results_by_mode["auto"]
    headers = ["模式", "样本", "Top-1一致率", "首次分歧位置"]
    rows = []
    # 用于画图：每个模式的平均一致率
    avg_match = {}
    for mode in QUANT_MODES:
        match_rates = []
        for i, (base_ids, cur_ids) in enumerate(zip(baseline, results_by_mode[mode])):
            n = min(len(base_ids), len(cur_ids))
            # 逐位置比较，统计匹配数与首次分歧位置
            matches = 0
            divergence = n  # 默认无分歧
            for j in range(n):
                if base_ids[j] == cur_ids[j]:
                    matches += 1
                else:
                    divergence = j
                    break
            rate = matches / n if n > 0 else 1.0
            match_rates.append(rate)
            rows.append([mode, f"#{i}", f"{rate:.3f}", str(divergence)])
        avg_match[mode] = sum(match_rates) / len(match_rates) if match_rates else 1.0

    save_table("e2e_accuracy", headers, rows)

    # 画图：各模式平均 top-1 一致率柱状图
    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(7, 5))
        modes = list(avg_match.keys())
        ax.bar(modes, [avg_match[m] for m in modes])
        ax.set_ylabel("avg top-1 match vs fp16 baseline")
        ax.set_ylim(0, 1.05)
        ax.set_title("End-to-end token agreement (Qwen3-0.6B)")
        for i, m in enumerate(modes):
            ax.text(i, avg_match[m] + 0.01, f"{avg_match[m]:.3f}", ha="center")
        out = RESULT_DIR / "e2e_accuracy.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  -> 图已保存: {out.name}")

    # 返回供综合图使用：{mode: 平均 top-1 一致率}
    return dict(avg_match)


# ======================================================================
# 模块 3：性能 / 显存 benchmark
# ======================================================================
def run_perf_benchmark():
    """
    显存对比：计算三种模式下 KV cache 理论占用，并实测加载引擎后的峰值显存。
    fp8 数据为 1 字节/元素，fp16 为 2 字节；per_token_head 额外有 float32 scale 开销。
    """
    print("\n" + "=" * 70)
    print("模块 3: 性能 / 显存 benchmark")
    print("=" * 70)

    # 模型结构参数（与 build_config 保持一致）
    num_layers, num_kv_heads, head_dim, block_size = 28, 8, 128, 256

    headers = ["模式", "每block KV字节", "相对fp16", "实测峰值显存(MB)"]
    rows = []
    plot_modes, plot_bytes = [], []

    for mode in QUANT_MODES:
        elem_bytes = 1 if mode != "auto" else 2
        # data 部分: block_size * 2(K/V) * num_layers * num_kv_heads * head_dim * elem_bytes
        data_bytes = block_size * 2 * num_layers * num_kv_heads * head_dim * elem_bytes
        scale_bytes = 0
        if mode == "fp8_per_token_head":
            # 每个 (slot, kv_head) 一个 float32 scale（K/V 各一份）
            scale_bytes = block_size * 2 * num_layers * num_kv_heads * 4
        total = data_bytes + scale_bytes

        # 实测：加载引擎并记录峰值显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            engine, _ = load_engine(mode, max_model_length=512)
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            free_engine(engine)
        except Exception as e:  # noqa: BLE001
            peak_mb = float("nan")
            print(f"  [警告] 加载 {mode} 引擎失败，跳过实测显存: {e}")

        rel = total / (block_size * 2 * num_layers * num_kv_heads * head_dim * 2)
        rows.append([mode, str(total), f"{rel:.3f}", f"{peak_mb:.1f}"])
        plot_modes.append(mode)
        plot_bytes.append(total)

    save_table("perf_memory", headers, rows)

    # 画图：每 block KV cache 字节数对比
    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(plot_modes, plot_bytes)
        ax.set_ylabel("KV cache bytes per block")
        ax.set_title("KV cache memory per block by quant mode")
        out = RESULT_DIR / "perf_memory.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  -> 图已保存: {out.name}")

    # 返回供综合图使用：{mode: 每block相对fp16的字节比例}
    base = plot_bytes[0] if plot_bytes else 1
    return {plot_modes[i]: plot_bytes[i] / base for i in range(len(plot_modes))}


# ======================================================================
# 模块 4：NIAH (Needle In A Haystack) 测试
# ======================================================================
# 干扰文本段落（haystack 的填充材料），重复堆叠以凑够目标长度
HAYSTACK_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Paris is a city with a long history of art and culture. "
    "Machine learning models process large amounts of data. "
    "Mountains rise high above the surrounding plains. "
)


def make_niah_prompt(tokenizer, context_len, depth_ratio, secret_code):
    """
    构造 NIAH 测试 prompt。

    参数:
        context_len: 目标上下文长度（以 token 数计）
        depth_ratio: needle 插入的相对深度 (0.0~1.0)，决定它埋在 haystack 的哪个位置
        secret_code: 待检索的关键信息（随机数字串），作为"针"
    返回:
        套用 chat 模板后的 prompt 字符串。
    """
    # needle：用稀有的随机数字串避免被模型先验"猜中"
    needle = f"The secret access code is {secret_code}. Remember this number. "

    # 先生成足够长的填充文本，再按 token 数截断到目标 context_len
    filler = HAYSTACK_FILLER
    repeat = max(1, context_len // 8)
    full_filler = filler * repeat
    filler_ids = tokenizer.encode(full_filler)[:context_len]

    # 在 depth_ratio 对应的位置切开，插入 needle
    insert_pos = int(len(filler_ids) * depth_ratio)
    needle_ids = tokenizer.encode(needle)
    haystack_ids = filler_ids[:insert_pos] + needle_ids + filler_ids[insert_pos:]
    haystack_text = tokenizer.decode(haystack_ids)

    # 末尾追加问题，要求模型复述 needle
    question = "\n\nWhat is the secret access code mentioned above? Answer with just the number."
    user_text = haystack_text + question
    return chat_prompt(tokenizer, user_text)


def run_niah_benchmark():
    """
    NIAH 长上下文检索压测：context 长度 x 插入深度 网格，
    统计每种量化模式是否能正确复述 needle，输出准确率热力图。
    """
    print("\n" + "=" * 70)
    print("模块 4: NIAH (Needle In A Haystack) 长上下文检索测试")
    print("=" * 70)

    from myvllm.sampling_parameters import SamplingParams

    # 测试网格：context 长度 x 插入深度
    ctx_lens = [128, 256, 512, 768]
    depths = [0.1, 0.3, 0.5, 0.7, 0.9]
    sp = SamplingParams(temperature=1e-6, max_tokens=16, max_model_length=1024)

    # 为每个 (ctx_len, depth) 准备一个随机 secret code
    torch.manual_seed(42)
    codes = {}
    for cl in ctx_lens:
        for d in depths:
            codes[(cl, d)] = str(int(torch.randint(10000, 99999, (1,)).item()))

    # 结果网格：mode -> 2D 准确率矩阵 (len(ctx_lens) x len(depths))
    grids = {}
    for mode in QUANT_MODES:
        engine, tokenizer = load_engine(mode, max_model_length=1024)
        grid = [[0.0] * len(depths) for _ in range(len(ctx_lens))]
        for ci, cl in enumerate(ctx_lens):
            # 构造该 context 长度下所有深度的 prompt，批量生成
            prompts = [make_niah_prompt(tokenizer, cl, d, codes[(cl, d)]) for d in depths]
            out = engine.generate(prompts, sp)
            for di, d in enumerate(depths):
                answer = out['text'][di]
                # 判定：模型输出是否包含正确的 secret code
                grid[ci][di] = 1.0 if codes[(cl, d)] in answer else 0.0
        grids[mode] = grid
        free_engine(engine)

    # 表格输出：逐 (mode, ctx_len) 汇总各深度准确率与整体平均
    headers = ["模式", "ctx_len"] + [f"depth={d}" for d in depths] + ["平均"]
    rows = []
    for mode in QUANT_MODES:
        for ci, cl in enumerate(ctx_lens):
            row_vals = grids[mode][ci]
            avg = sum(row_vals) / len(row_vals)
            rows.append([mode, str(cl)] + [f"{v:.0f}" for v in row_vals] + [f"{avg:.2f}"])
    save_table("niah_accuracy", headers, rows)

    # 画图：每种模式一张热力图（行=ctx_len，列=depth，值=准确率）
    if _HAS_MPL:
        fig, axes = plt.subplots(1, len(QUANT_MODES), figsize=(5 * len(QUANT_MODES), 4))
        if len(QUANT_MODES) == 1:
            axes = [axes]
        for ax, mode in zip(axes, QUANT_MODES):
            data = grids[mode]
            im = ax.imshow(data, vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
            ax.set_title(mode)
            ax.set_xticks(range(len(depths)))
            ax.set_xticklabels([str(d) for d in depths])
            ax.set_yticks(range(len(ctx_lens)))
            ax.set_yticklabels([str(cl) for cl in ctx_lens])
            ax.set_xlabel("needle depth")
            ax.set_ylabel("context length")
            # 在每个格子标注准确率
            for ci in range(len(ctx_lens)):
                for di in range(len(depths)):
                    ax.text(di, ci, f"{data[ci][di]:.0f}", ha="center", va="center")
        fig.colorbar(im, ax=axes, fraction=0.025, label="retrieval accuracy")
        out = RESULT_DIR / "niah_heatmap.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> 图已保存: {out.name}")

    # 返回供综合图使用：{mode: 整体平均检索准确率}
    niah_avg = {}
    for mode in QUANT_MODES:
        flat = [v for row in grids[mode] for v in row]
        niah_avg[mode] = sum(flat) / len(flat) if flat else 0.0
    return niah_avg


# ======================================================================
# 综合总览图：把所有评测的核心结论画到同一张图上
# ======================================================================
def plot_summary(numeric_res, perf_res, e2e_res, niah_res):
    """
    把四类评测的核心指标整合到一张图（2x2 子图），方便一眼看懂结论：

    含义解读（都是“越靠右/越高 = 量化越好”）：
      - 左上 显存占比：相对 fp16 的 KV cache 字节数，越低越省显存（量化的优点）
      - 右上 数值精度 SNR：量化还原信号的信噪比(dB)，越高越准；
              重点看 outlier 分布上 per_token_head 明显高于 per_tensor
      - 左下 端到端一致率：与 fp16 基线逐 token 对比的 top-1 一致率，越接近 1 越好
      - 右下 NIAH 检索准确率：长上下文“大海捞针”能力，越接近 1 说明量化没损伤检索

    一句话结论：per_token_head 用几乎和 per_tensor 相同的显存，换来明显更好的精度。
    """
    if not _HAS_MPL:
        print("[提示] 未安装 matplotlib，跳过综合总览图。")
        return

    modes = QUANT_MODES  # ["auto", "fp8_per_tensor", "fp8_per_token_head"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("KV Cache FP8 Quantization Summary (Qwen3-0.6B)", fontsize=14)

    # ---- 左上：显存占比（相对 fp16，越低越好）----
    ax = axes[0][0]
    if perf_res:
        vals = [perf_res.get(m, float("nan")) for m in modes]
        bars = ax.bar(modes, vals, color=["#888", "#e8a", "#5a9"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}x", ha="center")
        ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.set_title("KV cache memory vs fp16 (lower=better)")
    ax.set_ylabel("relative bytes")

    # ---- 右上：数值精度 SNR（越高越好），仅两种量化模式 ----
    ax = axes[0][1]
    if numeric_res:
        dist_names = list(numeric_res.keys())
        xs = range(len(dist_names))
        w = 0.35
        pt = [numeric_res[d]["per_tensor"] for d in dist_names]
        pth = [numeric_res[d]["per_token_head"] for d in dist_names]
        ax.bar([x - w / 2 for x in xs], pt, w, label="per_tensor", color="#e8a")
        ax.bar([x + w / 2 for x in xs], pth, w, label="per_token_head", color="#5a9")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(dist_names)
        ax.legend()
    ax.set_title("Numeric SNR by distribution (higher=better)")
    ax.set_ylabel("SNR (dB)")

    # ---- 左下：端到端 top-1 一致率（越高越好）----
    ax = axes[1][0]
    if e2e_res:
        vals = [e2e_res.get(m, float("nan")) for m in modes]
        bars = ax.bar(modes, vals, color=["#888", "#e8a", "#5a9"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center")
        ax.set_ylim(0, 1.1)
    ax.set_title("End-to-end top-1 match vs fp16 (higher=better)")
    ax.set_ylabel("agreement")

    # ---- 右下：NIAH 检索准确率（越高越好）----
    ax = axes[1][1]
    if niah_res:
        vals = [niah_res.get(m, float("nan")) for m in modes]
        bars = ax.bar(modes, vals, color=["#888", "#e8a", "#5a9"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center")
        ax.set_ylim(0, 1.1)
    ax.set_title("NIAH retrieval accuracy (higher=better)")
    ax.set_ylabel("accuracy")

    out = RESULT_DIR / "summary.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n[综合总览图] 已保存: {out}")


# ======================================================================
# 主入口
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="KV Cache FP8 量化 Benchmark")
    parser.add_argument(
        "--suite", default="all",
        help="要运行的评测模块，逗号分隔: numeric,e2e,perf,niah 或 all",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "本 benchmark 需要 CUDA GPU"
    device = "cuda"

    suite = args.suite.strip().lower()
    if suite == "all":
        selected = {"numeric", "e2e", "perf", "niah"}
    else:
        selected = {s.strip() for s in suite.split(",")}

    print(f"运行评测模块: {sorted(selected)}")
    print(f"结果输出目录: {RESULT_DIR}")

    # 收集各模块返回值，最后整合成一张总览图
    numeric_res = perf_res = e2e_res = niah_res = None
    if "numeric" in selected:
        numeric_res = run_numeric_benchmark(device)
    if "perf" in selected:
        perf_res = run_perf_benchmark()
    if "e2e" in selected:
        e2e_res = run_e2e_benchmark()
    if "niah" in selected:
        niah_res = run_niah_benchmark()

    # 综合总览图：四类指标合并到一张图
    plot_summary(numeric_res, perf_res, e2e_res, niah_res)

    print("\n全部评测完成。所有表格(.md/.csv)与图(.png)见 benchmark_results/ 目录。")
    print("重点看 benchmark_results/summary.png：一张图涵盖显存/精度/一致率/检索四项结论。")


if __name__ == "__main__":
    main()
