import sys, os
from pathlib import Path
import argparse

from transformers import AutoTokenizer
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description='MiniVLLM Inference Engine')

    # Scheduler batch config
    parser.add_argument('--max-num-sequences', type=int, default=16,
                        help='Max number of sequences per batch; limits the sequence dimension of a batch')
    parser.add_argument('--max-num-batched-tokens', type=int, default=1024,
                        help='Max number of new tokens per batch step; chunk size of chunk prefill is limited by this')
    parser.add_argument('--max-cached-blocks', type=int, default=1024,
                        help='Max number of KV cache blocks managed by BlockManager; affects how much context can be cached')
    parser.add_argument('--block-size', type=int, default=256,
                        help='Number of tokens per KV cache block; slot_mapping locates physical KV slots by block_size')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of distributed processes/tensor parallelism; current example is single-GPU single-process')

    # Chunked prefill config
    parser.add_argument('--enable-chunked-prefill', action='store_true', default=False,
                        help='Enable chunked prefill; long prompts can be split into multiple prefill rounds and mixed with decode')
    parser.add_argument('--disable-chunked-prefill', action='store_true',
                        help='Disable chunked prefill')
    parser.add_argument('--long-prefill-token-threshold', type=int, default=512,
                        help='Max prefill tokens per step for long prompts; 0 means no extra limit, only limited by max_num_batched_tokens')
    parser.add_argument('--max-num-partial-prefills', type=int, default=1,
                        help='Max concurrent partial/chunked prefills in running; None means unlimited')
    parser.add_argument('--max-long-partial-prefills', type=int, default=1,
                        help='Max concurrent long prompt partial prefills in running; None means unlimited, 0 means no new long partial prefills')

    # Model config
    parser.add_argument('--model-name-or-path', type=str, default='Qwen/Qwen3-0.6B',
                        help='HuggingFace model name or local path, used to load tokenizer and weights')
    parser.add_argument('--enforce-eager', action='store_true', default=True,
                        help='Force eager execution; CUDA graph decode is not used, more intuitive for debugging but may be slower')
    parser.add_argument('--use-cuda-graph', action='store_true',
                        help='Use CUDA graph for decoding; overrides --enforce-eager')
    parser.add_argument('--kv-cache-dtype', type=str, default='auto',
                        choices=['auto', 'fp8_per_tensor', 'fp8_per_token_head', 'int8_per_token_head', 'kivi_2bit', 'kivi_4bit'],
                        help='KV cache storage dtype or quantization mode')

    # Model architecture
    parser.add_argument('--vocab-size', type=int, default=151936,
                        help='Vocabulary size, must match HF checkpoint; Qwen3-0.6B uses 151936')
    parser.add_argument('--hidden-size', type=int, default=1024,
                        help='Transformer hidden dimension, also the total width of num_heads * head_dim before attention output projection')
    parser.add_argument('--num-heads', type=int, default=16,
                        help='Number of query attention heads')
    parser.add_argument('--head-dim', type=int, default=128,
                        help='Dimension per attention head; Q head_dim is still hidden_size / num_heads under GQA')
    parser.add_argument('--num-kv-heads', type=int, default=8,
                        help='Number of key/value heads; less than num_heads means GQA, multiple Q heads share one KV head group')
    parser.add_argument('--intermediate-size', type=int, default=3072,
                        help='MLP/FFN intermediate dimension')
    parser.add_argument('--num-layers', type=int, default=28,
                        help='Number of transformer decoder layers')
    parser.add_argument('--tie-word-embeddings', action='store_true', default=True,
                        help='Share input embedding and lm_head weights')

    # RoPE config
    parser.add_argument('--rope-base', type=int, default=1000000,
                        help='RoPE theta/base parameter; Qwen3 uses 1000000, positional encoding must match checkpoint config')
    parser.add_argument('--max-position', type=int, default=32768,
                        help='Max position index supported by RoPE precomputation; should be >= max_model_length')

    # Normalization
    parser.add_argument('--rms-norm-epsilon', type=float, default=1e-6,
                        help='RMSNorm epsilon, prevents division by zero and must match model config')

    # Biases
    parser.add_argument('--qkv-bias', action='store_true', default=False,
                        help='Use bias in Q/K/V projection; Qwen3-0.6B does not use QKV bias')
    parser.add_argument('--ffn-bias', action='store_true', default=False,
                        help='Use bias in FFN/MLP projection; Qwen3 does not use MLP bias')

    # Attention scaling
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Additional attention scaling factor; actual attention scale will also divide by sqrt(head_dim)')

    # Engine limits
    parser.add_argument('--max-num-batch-tokens', type=int, default=4096,
                        help='Max tokens for model runner warmup/graph capture; not the same as scheduler batched_tokens')
    parser.add_argument('--max-model-length', type=int, default=1024,
                        help='Max total length allowed per request, usually including prompt tokens and generated tokens')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='Maximum GPU memory ratio used for KV cache allocation (0~1)')

    # EOS token
    parser.add_argument('--eos-token-id', type=int, default=151645,
                        help='End-of-sequence token ID, must match tokenizer.eos_token_id, used to determine generation end')

    # Sampling params
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature; lower = more deterministic, higher = more random')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Max number of generated tokens (completion tokens only)')

    # Prompts
    parser.add_argument('--prompt', type=str, action='append',
                        help='Prompt(s) to generate (can be specified multiple times)')

    return parser.parse_args()


def build_config(args):
    config = {
        # Scheduler batch config
        'max_num_sequences': args.max_num_sequences,
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'max_cached_blocks': args.max_cached_blocks,
        'block_size': args.block_size,
        'world_size': args.world_size,

        # Chunked prefill config
        'enable_chunked_prefill': args.enable_chunked_prefill and not args.disable_chunked_prefill,
        'long_prefill_token_threshold': args.long_prefill_token_threshold,
        'max_num_partial_prefills': args.max_num_partial_prefills,
        'max_long_partial_prefills': args.max_long_partial_prefills,

        # Model config
        'model_name_or_path': args.model_name_or_path,
        'enforce_eager': args.enforce_eager and not args.use_cuda_graph,
        'kv_cache_dtype': args.kv_cache_dtype,

        # Model architecture
        'vocab_size': args.vocab_size,
        'hidden_size': args.hidden_size,
        'num_heads': args.num_heads,
        'head_dim': args.head_dim,
        'num_kv_heads': args.num_kv_heads,
        'intermediate_size': args.intermediate_size,
        'num_layers': args.num_layers,
        'tie_word_embeddings': args.tie_word_embeddings,

        # RoPE config
        'base': args.rope_base,
        'max_position': args.max_position,

        # Normalization
        'rms_norm_epsilon': args.rms_norm_epsilon,

        # Biases
        'qkv_bias': args.qkv_bias,
        'ffn_bias': args.ffn_bias,

        # Attention scaling
        'scale': args.scale,

        # Engine limits
        'max_num_batch_tokens': args.max_num_batch_tokens,
        'max_model_length': args.max_model_length,
        'gpu_memory_utilization': args.gpu_memory_utilization,

        # EOS token
        'eos': args.eos_token_id,
    }
    return config


def main():
    args = parse_args()
    config = build_config(args)

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    model_name = config['model_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path)
    llm = LLM(config=config)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_length=args.max_model_length,
    )

    if args.prompt:
        prompts = args.prompt
    else:
        prompts = [
            "1+3等于多少",
            "列出100以内的所有质数",
            "谈谈你对人工智能对社会影响的看法",
        ]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)
    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()
