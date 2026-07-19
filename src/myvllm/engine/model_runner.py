import math
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import ScheduledSequence
from myvllm.utils import *
from myvllm.utils.context import split_mixed_decode_prefill_metadata

class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        self.event = event

        # set distributed config
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
        torch.cuda.set_device(rank)

        # set model
        path_str = self.config['model_name_or_path']
        model_name = Path(path_str).name
        match model_name:
            case 'Qwen3-0.6B':
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    base=config['base'],
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            case 'Llama-3.2-1B-Instruct':
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    head_dim=config['head_dim'],
                    num_qo_heads=config['num_qo_heads'],
                    num_kv_heads=config['num_kv_heads'],
                    has_attn_bias=config['has_attn_bias'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    rope_base=config['rope_base'],
                    max_position_embeddings=config['max_position_embeddings'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    block_size=self.block_size,
                    tie_word_embeddings=config['tie_word_embeddings'],
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # Load weights in GPU (model moved to GPU before loading weights)
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step
        self._first_decode = False

        # KIVI: seq_id -> residual_slot 的稳定映射；每条运行中的序列分配一行 residual buffer
        # 在 prepare_prefill / prepare_decode 中按需 acquire / release
        self._kivi_seq_to_slot: dict[int, int] = {}
        self._kivi_free_slots: list[int] = []

        # warm up model so that we know peak memory usage
        self.warmup_model()
        # allocate kv cache
        self.allocate_kv_cache()
        # capture cuda graph for decoding
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.default_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__

    # only use read when rank != 0
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args) # Unpack args when calling
            if method_name == 'exit':
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)]
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # reserve some room for peak memory usage during model execution
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # find parameters to compute kv cache size
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads'] // self.world_size
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # ============ KV Cache 量化配置解析 ============
        # kv_cache_dtype 决定 KV cache 的存储精度与量化模式：
        #   "auto"               -> 不量化，使用默认精度 (fp16/bf16)
        #   "fp8_per_tensor"     -> FP8 量化，整个 cache 共享一组 (K/V 各一个) 标量 scale
        #   "fp8_per_token_head" -> FP8 量化，每个 (token, kv_head) 组合独立一个 scale (对齐 vLLM)
        #   "int8_per_token_head"-> INT8 量化，每个 (token, kv_head) 组合独立一个 scale (对齐 vLLM)
        #   "int4_per_token_head"-> INT4 量化，每两个 4-bit 值 pack 到 1 个 uint8 (模仿 vLLM)
        #   "int4_groupwise"     -> INT4 group-wise 量化，每个 token/head/group 独立 scale/zp
        #   "kivi_2bit"          -> KIVI 2-bit 量化 (K per-channel-per-block / V per-token-per-head + fp16 residual)
        #   "kivi_4bit"          -> KIVI 4-bit 量化 (同上, bit-width 不同)
        # 参考 vLLM 的 KVQuantMode，per_token_head 精度更高、对离群值更鲁棒。
        self.kv_cache_dtype = self.config.get('kv_cache_dtype', 'auto')
        is_int4_per_token_head = self.kv_cache_dtype == 'int4_per_token_head'
        is_int4_groupwise = self.kv_cache_dtype == 'int4_groupwise'
        is_per_token_head_quant = self.kv_cache_dtype in ('fp8_per_token_head', 'int8_per_token_head', 'int4_per_token_head')
        self.int4_group_size = int(self.config.get('kv_group_size', 32))
        self.int4_use_rht = bool(self.config.get('kv_use_rht', False))
        if is_int4_groupwise:
            assert head_dim % self.int4_group_size == 0, (
                f'head_dim {head_dim} must be divisible by kv_group_size {self.int4_group_size}'
            )
            if self.int4_use_rht:
                assert head_dim > 0 and (head_dim & (head_dim - 1)) == 0, (
                    f'RHT requires power-of-two head_dim, got {head_dim}'
                )
        # 是否启用普通 KV cache 量化 (不含 KIVI)
        self.kv_quant_enabled = self.kv_cache_dtype in (
            'fp8_per_tensor', 'fp8_per_token_head', 'int8_per_token_head',
            'int4_per_token_head', 'int4_groupwise',
        )
        # 是否启用 KIVI 量化路径
        self.kivi_enabled = self.kv_cache_dtype.startswith('kivi')
        # KIVI 超参数 (仅 KIVI 路径生效)
        self.kivi_bits = 2 if self.kv_cache_dtype == 'kivi_2bit' else (4 if self.kv_cache_dtype == 'kivi_4bit' else 0)
        self.kivi_group_size = int(self.config.get('kivi_group_size', 32))      # K 沿 head_dim 的分组宽度
        self.kivi_residual_length = int(self.config.get('kivi_residual_length', 32))  # fp16 buffer 容量

        cache_head_dim = head_dim
        # 按量化模式决定实际 cache dtype；不要在后续逻辑中再覆盖此值。
        if self.kivi_enabled:
            kv_data_dtype = torch.uint8
        elif is_int4_per_token_head or is_int4_groupwise:
            kv_data_dtype = torch.uint8
            cache_head_dim = (head_dim + 1) // 2
        elif self.kv_cache_dtype == 'int8_per_token_head':
            kv_data_dtype = torch.int8
        elif self.kv_cache_dtype in ('fp8_per_tensor', 'fp8_per_token_head'):
            kv_data_dtype = torch.float8_e4m3fn
        else:
            kv_data_dtype = self.default_dtype

        # 每个 cache 元素占用的字节数 (fp8=1, fp16/bf16=2)
        elem_bytes = kv_data_dtype.itemsize

        # check whether the current free memory can hold at least one block
        # compute the actual byte required of each block
        # data 部分: block_size * 2(K和V) * num_layers * num_kv_heads * cache_head_dim * elem_bytes
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * cache_head_dim * elem_bytes
        # per_token_head 模式额外需要为每个 (slot, kv_head) 存一个 float32 scale (K/V 各一份)
        # 额外开销 = block_size * 2(K和V) * num_layers * num_kv_heads * 4(float32)
        if is_per_token_head_quant:
            block_bytes += self.block_size * 2 * num_layers * num_kv_heads * 4
        elif is_int4_groupwise:
            n_groups = head_dim // self.int4_group_size
            block_bytes += self.block_size * 2 * num_layers * num_kv_heads * n_groups * 4
        if self.kivi_enabled:
            # KIVI 每个 block 额外开销 (按 fp32=4 bytes 计算):
            # - K scale + K zero: 2 * num_layers * block_size * num_kv_heads * (head_dim/group_size) * 4 bytes
            # - V scale         : num_layers * block_size * num_kv_heads * 4 bytes
            #   (V 只需 scale 不需 zero, 因 KIVI 中 V 用对称量化)
            assert head_dim % self.kivi_group_size == 0, \
                f"head_dim {head_dim} 必须能被 kivi_group_size {self.kivi_group_size} 整除"
            n_groups = head_dim // self.kivi_group_size
            block_bytes += num_layers * self.block_size * num_kv_heads * n_groups * 2 * 4   # K scale+zero
            block_bytes += num_layers * self.block_size * num_kv_heads * 4  # V scale
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterwards on rank-0) never allocates more blocks than any
            # rank can physically hold.
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # allocate max possible kv cache for the model, instead for each sequence
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values
        max_cached_blocks = self.config['max_cached_blocks']
        # KV cache data 张量：dtype 由 kv_data_dtype 决定 (fp8 量化时为 float8_e4m3fn)
        # 形状: (2, num_layers, max_cached_blocks, block_size, num_kv_heads, cache_head_dim)
        #       第 0 维 2 表示 K(=0) 和 V(=1)

        # 这里是分配kv cache的张量，后面分配scale的张量
        allocated_kv_cache = torch.zeros(
            2, num_layers, max_cached_blocks, self.block_size, num_kv_heads, cache_head_dim,
            dtype=kv_data_dtype, device=f'cuda:{self.rank}'
        )

        # ============ 为 KV cache 量化分配 scale 张量 ============
        # scale 始终用 float32 存储，反量化时: x_fp16 = x_quant.to(float) * scale
        allocated_k_scale = None
        allocated_v_scale = None
        # KIVI 量化使用的额外张量 (K 用非对称, 需 zero; V 用对称, scale 即可)
        allocated_k_zero = None
        allocated_k_residual = None
        allocated_v_residual = None
        if self.kv_cache_dtype == 'fp8_per_tensor':
            # per_tensor: 每层 K/V 各一个全局标量 scale
            # 形状: (2, num_layers, 1)，最后一维为 1 便于广播
            scale_buf = torch.ones(2, num_layers, 1, dtype=torch.float32, device=f'cuda:{self.rank}')
            allocated_k_scale = scale_buf[0]
            allocated_v_scale = scale_buf[1]
        elif is_per_token_head_quant:
            # per_token_head: 每个 (block, slot, kv_head) 一个 scale (对齐 vLLM 的 shape)
            # 形状: (2, num_layers, max_cached_blocks, block_size, num_kv_heads)
            scale_buf = torch.ones(
                2, num_layers, max_cached_blocks, self.block_size, num_kv_heads,
                dtype=torch.float32, device=f'cuda:{self.rank}'
            )
            allocated_k_scale = scale_buf[0]
            allocated_v_scale = scale_buf[1]
        elif is_int4_groupwise:
            # 每个 (block, slot, kv_head, group) 一个打包 scale/zp。
            n_groups = head_dim // self.int4_group_size
            scale_buf = torch.ones(
                2, num_layers, max_cached_blocks, self.block_size, num_kv_heads, n_groups,
                dtype=torch.float32, device=f'cuda:{self.rank}'
            )
            allocated_k_scale = scale_buf[0]
            allocated_v_scale = scale_buf[1]
        elif self.kivi_enabled:
            # ============ KIVI 量化的 scale / zero / residual 分配 ============
            # K: per-token-per-channel-group 非对称量化 (即每个 (block, slot, kv_head, group) 一组 scale/zero)
            #   shape: (num_layers, max_cached_blocks, block_size, num_kv_heads, head_dim/group_size)
            # BUG FIX: 原方案 shape 缺少 block_size 维 -> 同一块内不同 token 互相覆盖 scale/zero,
            #          decode 读到的几乎全是最后一个写入 token 的 scale, 导致输出乱码。
            n_groups = head_dim // self.kivi_group_size
            allocated_k_scale = torch.ones(
                num_layers, max_cached_blocks, self.block_size, num_kv_heads, n_groups,
                dtype=torch.float32, device=f'cuda:{self.rank}'
            )
            allocated_k_zero = torch.zeros(
                num_layers, max_cached_blocks, self.block_size, num_kv_heads, n_groups,
                dtype=torch.float32, device=f'cuda:{self.rank}'
            )
            # V: per-token-per-head 对称量化, 与 fp8_per_token_head 同布局
            #   shape: (num_layers, max_cached_blocks, block_size, num_kv_heads)
            allocated_v_scale = torch.ones(
                num_layers, max_cached_blocks, self.block_size, num_kv_heads,
                dtype=torch.float32, device=f'cuda:{self.rank}'
            )
            # residual buffer: 每条 (运行中) 序列的尾部 fp16 缓冲, 不参与 paging
            #   shape: (num_layers, max_num_seqs, residual_length, num_kv_heads, head_dim)
            # 用 max_num_sequences 作为 seq slot 数上限; 每条 seq 通过 prepare_xxx 传 residual_slots 拿到自己的行
            max_num_seqs = int(self.config.get('max_num_sequences', 16))
            self.kivi_max_num_seqs = max_num_seqs
            residual_dtype = self.default_dtype
            allocated_k_residual = torch.zeros(
                num_layers, max_num_seqs, self.kivi_residual_length, num_kv_heads, head_dim,
                dtype=residual_dtype, device=f'cuda:{self.rank}'
            )
            allocated_v_residual = torch.zeros(
                num_layers, max_num_seqs, self.kivi_residual_length, num_kv_heads, head_dim,
                dtype=residual_dtype, device=f'cuda:{self.rank}'
            )

        # 将 data / scale 张量按层下发到每个 attention 模块
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                # 透传量化模式，供 Attention.forward 选择量化/反量化路径
                module.kv_cache_dtype = self.kv_cache_dtype
                # 注入 scale 张量 (per_token_head: 对应层切片; per_tensor: 对应层切片; auto: None)
                if allocated_k_scale is not None:
                    module.k_scale = allocated_k_scale[layer_id]
                    module.v_scale = allocated_v_scale[layer_id]
                # KIVI 额外注入 zero / residual / 超参
                if self.kivi_enabled:
                    module.k_zero = allocated_k_zero[layer_id]
                    module.k_residual = allocated_k_residual[layer_id]
                    module.v_residual = allocated_v_residual[layer_id]
                module.kivi_bits = self.kivi_bits
                module.kivi_group_size = self.kivi_group_size
                module.kivi_residual_length = self.kivi_residual_length
                module.int4_group_size = self.int4_group_size
                module.int4_use_rht = self.int4_use_rht
                layer_id += 1

    # ============ KIVI residual buffer slot 管理 ============
    # 每条运行中的序列在 residual buffer 中独占一行；序列结束(从此 method 调用方再也看不到)时回收。
    # 简单实现: 用一个 set 跟踪当前批中的 seq_id, 把不在批中的 slot 回收回空闲池。
    def _kivi_get_slots_for_seqs(self, seqs: list[Sequence]) -> list[int]:
        if not getattr(self, 'kivi_enabled', False):
            return []
        # 初始化空闲池
        if not self._kivi_free_slots and not self._kivi_seq_to_slot:
            self._kivi_free_slots = list(range(self.kivi_max_num_seqs))
        current_ids = {seq.seq_id for seq in seqs}
        # 回收: 不再出现在批里的 seq 对应 slot 释放
        stale = [sid for sid in self._kivi_seq_to_slot if sid not in current_ids]
        for sid in stale:
            self._kivi_free_slots.append(self._kivi_seq_to_slot.pop(sid))
        # 分配
        slots: list[int] = []
        for seq in seqs:
            if seq.seq_id not in self._kivi_seq_to_slot:
                assert self._kivi_free_slots, "KIVI residual slot 不足: 请增大 max_num_sequences"
                self._kivi_seq_to_slot[seq.seq_id] = self._kivi_free_slots.pop()
            slots.append(self._kivi_seq_to_slot[seq.seq_id])
        return slots

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration: 
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # length: sum of all input_ids after prefix cache
        input_ids = []
        # length: sum of all input_ids after prefix cache
        slot_mappings = []
        # length: num_seqs
        seqlens_q = []
        # length: num_seqs
        seqlens_k = []
        # length: num_seqs + 1
        cu_seqlens_q = [0]
        # length: num_seqs + 1
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # pad block_tables
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        # ============ KIVI: 为每个 token 准备 residual_slot ============
        # prefill 阶段把 (num_tokens,) 的 residual_slot 与 token 序列对齐：
        #   每条 seq 的所有"新写入" token 共享同一个 residual_slot (= 该 seq 在 residual buffer 中的行号)
        residual_slots_tensor = None
        residual_lens_tensor = None
        if getattr(self, 'kivi_enabled', False):
            seq_slots = self._kivi_get_slots_for_seqs(seqs)
            per_token_slot: list[int] = []
            per_seq_residual_len: list[int] = []
            for seq, slot in zip(seqs, seq_slots):
                # 当前 seq 在本次 prefill 中要写入的 token 数
                new_n = len(seq.token_ids) - seq.num_cached_tokens
                per_token_slot.extend([slot] * new_n)
                # prefill 起始时 residual 长度为 num_cached_tokens (前缀命中可能不为 0)
                per_seq_residual_len.append(seq.num_cached_tokens)
            residual_slots_tensor = torch.tensor(per_token_slot, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            residual_lens_tensor = torch.tensor(per_seq_residual_len, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
            is_full_prefill=cu_seqlens_q[-1] == cu_seqlens_k[-1],
            residual_slots=residual_slots_tensor,
            residual_lens=residual_lens_tensor,
        )
        return input_ids


    def prepare_mixed(self, scheduled_items: list[ScheduledSequence]) -> torch.Tensor:
        input_ids = []
        slot_mappings = []
        seqlens_q = []
        seqlens_k = []
        context_lens = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        block_tables = []
        is_full_prefill_items = []
        query_lens = []
        is_decode_items = []
        seqs = [item.seq for item in scheduled_items]

        for item in scheduled_items:
            seq = item.seq
            if seq.num_computed_tokens < seq.num_prompt_tokens:
                # prefill chunk：只把本轮 scheduler 切出的 prompt 片段送进模型。
                # start 不能早于 prefix cache，也不能早于已经 forward 过的 chunk。
                start = max(seq.num_cached_tokens, seq.num_computed_tokens)
                end = start + item.num_scheduled_tokens
                input_ids.extend(seq.token_ids[start:end])
            else:
                # decode：模型只需要最新 token，但 attention 仍通过 block_tables/context_lens 读取完整 KV 历史。
                start = seq.num_tokens - 1
                end = seq.num_tokens
                input_ids.append(seq.last_token)

            # q 长度是本轮实际要算的新 token 数；k/context 长度是该 seq 当前可见的完整上下文长度。
            # pure full prefill 必须从 0 开始，且本轮 q_len 覆盖完整 context；decode/chunk/extend 都不是 full prefill。
            is_full_prefill_items.append(
                seq.num_computed_tokens < seq.num_prompt_tokens
                and start == 0
                and item.num_scheduled_tokens == end
            )
            seqlens_q.append(item.num_scheduled_tokens)
            seqlens_k.append(end)
            context_lens.append(end)
            query_lens.append(item.num_scheduled_tokens)
            is_decode_items.append(seq.num_computed_tokens >= seq.num_prompt_tokens)
            cu_seqlens_q.append(cu_seqlens_q[-1] + item.num_scheduled_tokens)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)

            # positions 是绝对位置，避免每个 chunk 的 RoPE 从 0 重新开始。
            # slot_mapping 告诉 attention 当前新 token 的 K/V 应写入哪个 paged KV cache slot。
            positions.extend(range(start, end))
            for token_idx in range(start, end):
                block_idx = token_idx // self.block_size
                block_offset = token_idx % self.block_size
                slot_mappings.append(seq.block_table[block_idx] * self.block_size + block_offset)

        max_num_blocks = max(len(seq.block_table) for seq in seqs)
        for seq in seqs:
            block_tables.append(seq.block_table + [-1] * (max_num_blocks - len(seq.block_table)))

        mixed_attention_split = split_mixed_decode_prefill_metadata(
            cu_seqlens_q, query_lens, is_decode_items
        )
        has_mixed_decode_prefill = (
            bool(mixed_attention_split.decode_token_indices)
            and bool(mixed_attention_split.prefill_token_indices)
        )
        if has_mixed_decode_prefill:
            mixed_attention_split.decode_token_indices_tensor = torch.tensor(
                mixed_attention_split.decode_token_indices,
                dtype=torch.long,
                pin_memory=True,
            ).cuda(non_blocking=True)
            mixed_attention_split.decode_seq_indices_tensor = torch.tensor(
                mixed_attention_split.decode_seq_indices,
                dtype=torch.long,
                pin_memory=True,
            ).cuda(non_blocking=True)
            mixed_attention_split.prefill_token_indices_tensor = torch.tensor(
                mixed_attention_split.prefill_token_indices,
                dtype=torch.long,
                pin_memory=True,
            ).cuda(non_blocking=True)
            mixed_attention_split.prefill_seq_indices_tensor = torch.tensor(
                mixed_attention_split.prefill_seq_indices,
                dtype=torch.long,
                pin_memory=True,
            ).cuda(non_blocking=True)
            mixed_attention_split.prefill_cu_seqlens_q_tensor = torch.tensor(
                mixed_attention_split.prefill_cu_seqlens_q,
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True)

        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        residual_slots_tensor = None
        residual_lens_tensor = None
        if getattr(self, 'kivi_enabled', False):
            seq_slots = self._kivi_get_slots_for_seqs(seqs)
            per_token_slot = []
            for item, slot in zip(scheduled_items, seq_slots):
                per_token_slot.extend([slot] * item.num_scheduled_tokens)
            residual_slots_tensor = torch.tensor(per_token_slot, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            residual_lens_tensor = torch.tensor(
                [item.seq.num_computed_tokens for item in scheduled_items],
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            positions=torch.tensor(positions, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            query_lens=torch.tensor(query_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            is_decode=torch.tensor(is_decode_items, dtype=torch.bool, pin_memory=True).cuda(non_blocking=True),
            mixed_attention_split=mixed_attention_split,
            is_full_prefill=all(is_full_prefill_items),
            residual_slots=residual_slots_tensor,
            residual_lens=residual_lens_tensor,
        )
        return input_ids


    # prepare input data for decoding
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        # KIVI: decode 阶段每条 seq 一个 token, residual_slots 形状 (num_seqs,)
        # residual_lens = 当前 seq 在 residual buffer 里 *已经* 写入了多少 token (不含本步)
        residual_slots_tensor = None
        residual_lens_tensor = None
        if getattr(self, 'kivi_enabled', False):
            seq_slots = self._kivi_get_slots_for_seqs(seqs)
            # 本步 token 写入前的 residual 长度 = (序列总长 - 1) % residual_length
            # 这里采用最简策略: 我们让 residual buffer 始终承接尾部 (序列长度 - paged_in_kv_tokens) 个 token,
            # 但为最小可跑实现, 直接用 (num_tokens - 1) 作为写入索引 (mod residual_length)
            residual_lens = [(len(seq) - 1) % self.kivi_residual_length for seq in seqs]
            residual_slots_tensor = torch.tensor(seq_slots, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            residual_lens_tensor = torch.tensor(residual_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
            residual_slots=residual_slots_tensor,
            residual_lens=residual_lens_tensor,
        )
        return input_ids    

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # finds smallest captured graph that fits the batch size
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'][:bs].fill_(-1)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # replay the graph
            graph.replay()
            logits = self.model.compute_logits(vars['outputs'][:bs])

        return logits


    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[ScheduledSequence] | list[Sequence], is_prefill: bool) -> list[int]:
        # prefill chunk才会给ScheduledSequence，没有chunk时给Sequence
        if seqs and isinstance(seqs[0], ScheduledSequence):
            scheduled_items = seqs
            raw_seqs = [item.seq for item in scheduled_items]
            decode_items = [item for item in scheduled_items if item.seq.num_computed_tokens >= item.seq.num_prompt_tokens]
            decode_only = bool(decode_items) and len(decode_items) == len(scheduled_items)
            if decode_only:
                # decode_only可以走decode的cuda graph加速
                input_ids = self.prepare_decode(raw_seqs)
                logits = self.run_model(input_ids, False)
                sample_seqs = raw_seqs
            else:
                # mixed batch 可能同时包含：长 prompt 的 prefill chunk、短 prompt 的完整 prefill、以及 decode token。
                # 统一走 eager prefill 路径，因为 CUDA graph decode path 只覆盖纯 decode batch。
                input_ids = self.prepare_mixed(scheduled_items)
                logits = self.run_model(input_ids, True)
                sample_indices = [
                    # 只有已经完成 prompt 的 item 才需要采样；未完成的中间 chunk 只写 KV，不产 completion token。
                    i for i, item in enumerate(scheduled_items)
                    if item.seq.num_computed_tokens + item.num_scheduled_tokens >= item.seq.num_prompt_tokens
                ]
                logits = logits[sample_indices] if sample_indices else None
                sample_seqs = [scheduled_items[i].seq for i in sample_indices]
        else:
            raw_seqs = seqs
            if is_prefill:
                input_ids = self.prepare_prefill(raw_seqs)
            else:
                input_ids = self.prepare_decode(raw_seqs)
            logits = self.run_model(input_ids, is_prefill)
            sample_seqs = raw_seqs

        # only sample when rank == 0
        token_ids = None
        if self.rank == 0 and logits is not None and sample_seqs:
            token_ids = self.sampler(logits, self.prepare_sample(sample_seqs))
        reset_context()
        return token_ids

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = self.config['max_num_seqs']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)
        # for decoding, input is always of shape (batch_size, 1)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # where to read KV values in the cache
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # output logits
        outputs = torch.zeros(max_bs, self.config['vocab_size'], device=f'cuda:{self.rank}')

        # graphs to be captured for different batch sizes
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
