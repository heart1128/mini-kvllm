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
    'max_num_sequences': 16,  # Scheduler 单个 step 最多同时调度的请求/序列数，限制 batch 的 sequence 维度。
    'max_num_batched_tokens': 1024,  # Scheduler 单个 step 最多调度的新 token 数；chunk prefill 的 chunk 大小受它限制。
    'enable_chunked_prefill': True,  # 是否启用 chunked prefill；长 prompt 可拆成多轮 prefill，并允许和 decode 混合调度。
    'long_prefill_token_threshold': 512,  # 长 prompt 单个 step 最多调度的 prefill token 数；0 表示不额外限制，只受 max_num_batched_tokens 限制。
    'max_num_partial_prefills': 1,  # running 中最多允许多少个未完成 prompt 的 partial/chunked prefill；None 表示不限制。
    'max_long_partial_prefills': 1,  # running 中最多允许多少个长 prompt partial prefill；None 表示不限制，0 表示不允许新的长 partial prefill。
    'max_cached_blocks': 1024,  # BlockManager 可管理的 KV cache block 总数上限，影响最多能缓存多少上下文。
    'block_size': 256,  # 每个 KV cache block 容纳的 token 数；slot_mapping 会按 block_size 定位物理 KV slot。
    'world_size': 1,  # 张量并行/进程数；当前示例是单卡单进程。
    'model_name_or_path': 'Qwen/Qwen3-0.6B',  # HuggingFace 模型名或本地模型路径，用于加载 tokenizer 和权重。
    'enforce_eager': True,  # 是否强制 eager 执行；True 时不使用 CUDA graph decode，加调试更直观但可能慢一些。
    'vocab_size': 151936,  # 词表大小，必须和 HF checkpoint 一致；Qwen3-0.6B 使用 151936。
    'hidden_size': 1024,  # Transformer 隐层维度，也是 attention 输出投影前 num_heads * head_dim 的总宽度。
    'num_heads': 16,  # query attention head 数。
    'head_dim': 128,  # 每个 attention head 的维度；GQA 下 Q head_dim 仍为 hidden_size / num_heads。
    'num_kv_heads': 8,  # key/value head 数；小于 num_heads 表示使用 GQA，多个 Q head 共享一组 KV head。
    'intermediate_size': 3072,  # MLP/FFN 中间层维度。
    'num_layers': 28,  # Transformer decoder layer 层数。
    'tie_word_embeddings': True,  # 是否复用输入 embedding 和 lm_head 权重。
    'base': 1000000,  # RoPE theta/base；Qwen3 使用 1000000，位置编码必须和 checkpoint 配置一致。
    'rms_norm_epsilon': 1e-6,  # RMSNorm 的 epsilon，防止除零并保持和模型配置一致。
    'qkv_bias': False,  # Q/K/V projection 是否带 bias；Qwen3-0.6B 不使用 QKV bias。
    'scale': 1,  # attention 额外缩放系数；实际 attention scale 会再除以 sqrt(head_dim)。
    'max_position': 32768,  # RoPE 预计算支持的最大位置索引，应不小于 max_model_length。
    'ffn_bias': False,  # FFN/MLP projection 是否带 bias；Qwen3 不使用 MLP bias。
    'max_num_batch_tokens': 4096,  # ModelRunner warmup/图捕获使用的最大 token 数上限，和 scheduler 的 batched_tokens 不是同一个 key。
    'max_model_length': 128,  # 单个请求允许的最大总长度，通常包含 prompt token 和 generated token。
    'gpu_memory_utilization': 0.9,  # KV cache 分配时最多使用的 GPU 显存比例。
    'eos': 151645,  # EOS token id，必须和 tokenizer.eos_token_id 一致，用于判断生成结束。
}

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    model_name = config.get('model_name_or_path', 'Qwen/Qwen3-0.6B')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path)
    llm = LLM(config=config)
    
    # max_tokens is the max number of generated tokens
    # max_model_length is the max total length including prompt
    # both should be set in SamplingParams and help to determine when to stop generation
    sampling_params = SamplingParams(temperature=0.6, max_tokens=512, max_model_length=1024)
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