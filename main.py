import sys, os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    # 调度器批处理配置
    'max_num_sequences': 16,              # 单批次最大序列数
    'max_num_batched_tokens': 1024,       # 单批次最大token数
    'max_cached_blocks': 1024,            # KV缓存最大block数
    'block_size': 256,                    # 每个block包含的token数（KV缓存分配单位）
    'world_size': 1,                      # 分布式训练/推理的进程数

    # 模型路径与运行模式
    'model_name_or_path': 'Qwen/Qwen3-0.6B',  # HuggingFace模型名称或本地路径
    'enforce_eager': True,                # True=禁用CUDA graph，False=启用（可能加速但占用更多显存）

    # 模型结构参数（应与对应模型一致）
    'vocab_size': 151936,                  # 词表大小
    'hidden_size': 1024,                   # 隐藏层维度
    'num_heads': 16,                       # 注意力头数（query）
    'head_dim': 128,                       # 每个注意力头的维度
    'num_kv_heads': 8,                     # key/value头数（用于GQA/MQA，< num_heads时可降低KV计算量）
    'intermediate_size': 3072,             # FFN中间层维度（mlp_hidden_size）
    'num_layers': 28,                      # transformer层数
    'tie_word_embeddings': True,          # 词嵌入权重与输出层权重共享

    # RoPE (Rotary Position Embedding) 配置
    'base': 1000000,                       # RoPE的base参数（theta）
    'max_position': 32768,                 # 最大位置索引（需 >= max_model_length）

    # Normalization 与 激活
    'rms_norm_epsilon': 1e-6,              # RMSNorm的epsilon（防止除零）

    # QKV 偏置（Qwen3为False）
    'qkv_bias': False,                     # QKV投影是否使用bias
    'ffn_bias': False,                     # FFN/MLP是否使用bias

    # 缩放因子（attention score乘以的系数）
    'scale': 1,                            # 注意力分数缩放，通常为 1.0 或 1/sqrt(head_dim)

    # 引擎资源与生成限制
    'max_num_batch_tokens': 4096,          # 调度器单次调度的最大token数（prompt+generation）
    'max_model_length': 128,               # 模型支持的最大序列长度（prompt + completion）
    'gpu_memory_utilization': 0.9,         # GPU显存使用比例（0~1）

    # kv cache量化
    # 'kv_cache_dtype': 'kivi_4bit',        # 或 'kivi_4bit'
    # 'kivi_group_size': 32,
    # 'kivi_residual_length': 128,
    'kv_cache_dtype': 'fp8_per_token_head',
    'enforce_eager': True,                # KIVI 不走 CUDA Graph

    # EOS token ID
    'eos': 151645,                         # End-of-Sequence token的ID
}

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    model_name = config.get('model_name_or_path', 'Qwen/Qwen3-0.6B')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path)
    llm = LLM(config=config)
    
    # max_tokens is the max number of generated tokens
    # max_model_length is the max total length including prompt
    # both should be set in SamplingParams and help to determine when to stop generation
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256, max_model_length=512)
    prompts = [
        "1+3等于多少",# * 15,
        "列出100以内的所有质数",# * 15,
        "谈谈你对人工智能对社会影响的看法",# * 15,
    ] #* 30
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    # outputs is a dict with 'text' and 'token_ids' keys
    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()