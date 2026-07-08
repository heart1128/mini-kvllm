from collections import deque
from dataclasses import dataclass
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


@dataclass(frozen=True)
class ScheduledSequence:
    # 一次调度不再等价于“整个 seq 都要跑完”。
    # chunk prefill 下，一个长 prompt 只会被切出 num_scheduled_tokens 个 token 参与本轮 forward。
    seq: Sequence
    num_scheduled_tokens: int


class Scheduler:
    def __init__(
        self,
        max_num_sequences: int,
        max_num_batched_tokens: int,
        max_cached_blocks: int,
        block_size: int,
        eos: int,
        enable_chunked_prefill: bool = True,
        long_prefill_token_threshold: int | None = None,
        max_num_partial_prefills: int | None = None,
        max_long_partial_prefills: int | None = None,
    ):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        self.enable_chunked_prefill = enable_chunked_prefill
        self.long_prefill_token_threshold = long_prefill_token_threshold or 0
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_long_partial_prefills = max_long_partial_prefills
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[ScheduledSequence], bool]:
        if not self.enable_chunked_prefill:
            return self._schedule_without_chunked_prefill()
        return self._schedule_chunked_prefill()


    def _is_partial_prefill(self, seq: Sequence) -> bool:
        return 0 < seq.num_computed_tokens < seq.num_prompt_tokens


    def _is_long_prefill(self, seq: Sequence) -> bool:
        return (
            self.long_prefill_token_threshold > 0
            and seq.num_prompt_tokens > self.long_prefill_token_threshold
        )


    def _chunk_token_limit(self, seq: Sequence, remaining_budget: int) -> int:
        limits = [seq.num_prompt_tokens - seq.num_computed_tokens, remaining_budget]
        if self.long_prefill_token_threshold > 0 and self._is_long_prefill(seq):
            # 长 prompt 单个 step 最多吃 threshold 个 token，避免队首长 prefill 占满整个 batch budget。
            limits.append(self.long_prefill_token_threshold)
        return min(limits)


    def _can_admit_partial_prefill(self, seq: Sequence) -> bool:
        # 这个函数只用于 waiting admission 阶段：判断队首 waiting seq 如果本轮进入 running，
        # 会不会制造一个新的 partial/chunked prefill，以及是否超过 partial prefill 并发限制。
        # 已经在 running 中的 partial prefill 不会被这里拦住，它们仍会继续推进后续 chunk。

        # 1. 全局剩余的 budget 太小了，不够一次性把本次prefill计算完事True
        will_be_partial = self._chunk_token_limit(seq, self.max_num_batched_tokens) < (
            seq.num_prompt_tokens - seq.num_computed_tokens
        )
        # 如果本轮可以一次算完剩余 prompt，它就是 full prefill，不会留下 partial prefill 状态。
        # 这种请求不消耗 partial prefill 名额，可以直接 admission。

        # 2. 剩余的 budget 足够，可以一次性算完
        if not will_be_partial:
            return True

        # max_num_partial_prefills 限制所有 partial/chunked prefill 的总并发数，
        # 不区分 prompt 长短。达到上限时，新的 partial prefill 暂不从 waiting 进入 running。

        # 3. 上面过滤了不能一次性算完，这里过滤要chunk prefill的数量
        partial_prefills = sum(1 for running_seq in self.running if self._is_partial_prefill(running_seq))
        # 如果chunk prefill的数量大于设置的最大prefil chunk数量，就不能执行了
        if self.max_num_partial_prefills is not None and partial_prefills >= self.max_num_partial_prefills:
            return False

        # max_long_partial_prefills 只限制“长 prompt”的 partial prefill 并发数。
        # 短 prompt 即使会 partial，也不受这个 long-only 限制；它仍受上面的总数限制。
        if self._is_long_prefill(seq):
            long_partial_prefills = sum(
                1
                for running_seq in self.running
                if self._is_partial_prefill(running_seq) and self._is_long_prefill(running_seq)
            )
            if (
                self.max_long_partial_prefills is not None
                and long_partial_prefills >= self.max_long_partial_prefills
            ):
                return False

        return True


    def _schedule_chunked_prefill(self) -> tuple[list[ScheduledSequence], bool]:
        # 本轮真正要交给 ModelRunner 的调度结果。
        # 每个 item 只描述“这个 seq 本轮算多少 token”，不再隐含“整个 seq 都算完”。
        scheduled_sequences: list[ScheduledSequence] = []
        # 用于最后重排 running 队列：已经被本轮调度过的 seq 会放到队首，未调度的保持相对顺序排在后面。
        scheduled_set: set[Sequence] = set()
        # 本轮已经消耗的 token budget；不能超过 max_num_batched_tokens。
        current_scheduled_tokens = 0
        # 返回给 LLMEngine/ModelRunner 的标记：只要本轮包含 prefill chunk，就走 prefill/eager 路径。
        is_prefill = False
        # 发生 preempt 后，本轮不再 admission waiting 请求，避免刚抢占的请求又立刻被重新调度。
        preempted = False

        # 第一阶段：按 running 队列顺序调度，和 vLLM V1 一样不额外强制 decode-ready 优先。
        # 每个 seq 根据 num_computed_tokens 判断本轮是继续 prefill chunk，还是 decode 1 个 token。
        req_index = 0
        while req_index < len(self.running):
            # token 维度的上限：一个 batch 最多处理 max_num_batched_tokens 个新 token。
            if current_scheduled_tokens >= self.max_num_batched_tokens:
                break
            # sequence 维度的上限：一个 batch 最多同时包含 max_num_sequences 条序列。
            if len(scheduled_sequences) >= self.max_num_sequences:
                break

            seq = self.running[req_index]
            remaining_budget = self.max_num_batched_tokens - current_scheduled_tokens
            # 还没有完成prefill，是中间或者最后一个chunk prefill
            if seq.num_computed_tokens < seq.num_prompt_tokens:
                num_new_tokens = self._chunk_token_limit(seq, remaining_budget)
                if num_new_tokens <= 0:  # 没有足够的 budget 了，退出
                    break
                scheduled_sequences.append(ScheduledSequence(seq, num_new_tokens))
                scheduled_set.add(seq)
                current_scheduled_tokens += num_new_tokens
                is_prefill = True
                req_index += 1
                continue

            # decode 前需要确认最后一个 KV block 还能追加新 token；不能追加时做一次抢占/回收。
            if not self.block_manager.can_append(seq):
                # 简化策略：抢占 running 队尾请求，释放它的 KV blocks，并放回 waiting 队首。
                preempted_seq = self.running.pop()
                self.preempt(preempted_seq)
                preempted = True
                # 如果被抢占的正是当前 seq，当前 seq 已经不在 running 中，本轮停止扫描 running。
                if preempted_seq is seq:
                    break
                # 否则继续尝试当前 req_index 位置上的新 seq。
                continue

            # decode path：为新 token 追加 KV slot，本轮只调度 1 个 token。
            self.block_manager.append(seq)
            scheduled_sequences.append(ScheduledSequence(seq, 1))
            scheduled_set.add(seq)
            current_scheduled_tokens += 1
            req_index += 1

        # 第二阶段：用 RUNNING 消耗后剩下的 token budget admission WAITING 请求。
        # 与非 chunked prefill 不同，长 prompt 不要求完整放进 budget；能放多少就先算多少。
        # admission 后请求进入 running，后续未完成的 chunk 会在第一阶段继续推进。
        # preempted过的就不加新请求了，本来资源都不足
        while self.waiting and not preempted:
            if current_scheduled_tokens >= self.max_num_batched_tokens:
                break
            if len(scheduled_sequences) >= self.max_num_sequences:
                break
            # running 已经达到并发上限时，不再从 waiting 拉新请求。
            if len(self.running) >= self.max_num_sequences:
                break

            # 这里只看 waiting 队首，保持 FCFS；如果队首无法分配 KV block，就不跳过它调后面的请求。
            seq = self.waiting[0]
            seq.num_computed_tokens = max(seq.num_computed_tokens, seq.num_cached_tokens)
            # 限制长请求一直做prefill chunk，给短请求一点机会
            if not self._can_admit_partial_prefill(seq):
                break
            if not self.block_manager.can_allocate(seq):
                break

            # waiting 请求一旦分配 KV block 就进入 running。
            # 如果它是长 prompt，本轮只计算第一个 chunk；后续 chunk 由 running-first 阶段继续调度。
            seq = self.waiting.popleft()
            self.block_manager.allocate(seq)
            # prefix cache 命中的 token 已经有可复用 KV，不需要重复 forward。
            # allocate() 可能更新 num_cached_tokens，所以必须在这里之后再计算 chunk 大小。
            seq.num_computed_tokens = max(seq.num_computed_tokens, seq.num_cached_tokens)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

            remaining_budget = self.max_num_batched_tokens - current_scheduled_tokens
            # chunk 大小由“剩余 prompt token 数”、“剩余 batch budget”和长 prefill 阈值共同决定。
            num_new_tokens = self._chunk_token_limit(seq, remaining_budget)
            if num_new_tokens <= 0:
                continue
            scheduled_sequences.append(ScheduledSequence(seq, num_new_tokens))
            scheduled_set.add(seq)
            current_scheduled_tokens += num_new_tokens
            is_prefill = True

        # 第三阶段：重排 running 队列。
        # 本轮调度过的 seq 放在前面，下一轮能优先继续 decode 或继续 prefill chunk；
        # 未调度到的 running seq 保持原来的 FCFS 相对顺序，排在后面。
        if scheduled_sequences:
            scheduled_running = [item.seq for item in scheduled_sequences if item.seq in self.running]
            unscheduled_running = [seq for seq in self.running if seq not in scheduled_set]
            self.running = deque(scheduled_running + unscheduled_running)

        return scheduled_sequences, is_prefill


    def _schedule_without_chunked_prefill(self) -> tuple[list[ScheduledSequence], bool]:
        scheduled_sequences: list[ScheduledSequence] = []
        current_scheduled_tokens = 0
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            num_new_tokens = seq.num_prompt_tokens - max(seq.num_cached_tokens, seq.num_computed_tokens)
            if (
                self.block_manager.can_allocate(seq)
                and num_new_tokens + current_scheduled_tokens <= self.max_num_batched_tokens
            ):
                seq = self.waiting.popleft()
                self.block_manager.allocate(seq)
                # allocate() may update seq.num_cached_tokens when prefix cache hits.
                # Recompute the actual number of tokens to forward after allocation;
                # otherwise identical long prompts can be scheduled with the stale
                # full prompt length and prepare_mixed() will write past block_table.
                seq.num_computed_tokens = max(seq.num_computed_tokens, seq.num_cached_tokens)
                num_new_tokens = seq.num_prompt_tokens - seq.num_computed_tokens
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                if num_new_tokens > 0:
                    scheduled_sequences.append(ScheduledSequence(seq, num_new_tokens))
                    current_scheduled_tokens += num_new_tokens
            else:
                break
        if scheduled_sequences:
            return scheduled_sequences, True

        while self.running:
            seq = self.running.popleft()
            if not self.block_manager.can_append(seq):
                if self.running:
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)
                    break
                self.block_manager.append(seq)
                scheduled_sequences.append(ScheduledSequence(seq, 1))
                current_scheduled_tokens += 1

        if scheduled_sequences:
            self.running.extendleft(reversed([item.seq for item in scheduled_sequences]))

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[ScheduledSequence], token_ids: list[int]) -> None:
        token_iter = iter(token_ids or []) # 这里是所有序列的采样token（中间chunk不会有采样token）
        for item in seqs:
            seq = item.seq
            was_prefilling = seq.num_computed_tokens < seq.num_prompt_tokens
            # 本轮 forward 已经计算了 num_scheduled_tokens 个 token，先推进 computed 进度。
            # 如果 prompt 还没全部算完，本轮不会产生可追加的 completion token。
            seq.num_computed_tokens += item.num_scheduled_tokens
            # 中间chunk，只更新num_computed_tokens
            if was_prefilling and seq.num_computed_tokens < seq.num_prompt_tokens:
                continue

            # 只有 decode step，或最后一个 prefill chunk 完成 prompt 后，sampler 输出才会被 append。
            token_id = next(token_iter)
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos  # 遇到停止符
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens # 超出最大token限制
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length # 超出最大序列长度限制

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
