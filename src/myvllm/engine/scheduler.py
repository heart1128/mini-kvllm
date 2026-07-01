from collections import deque
from dataclasses import dataclass
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


@dataclass(frozen=True)
class ScheduledSequence:
    seq: Sequence
    num_scheduled_tokens: int


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int, enable_chunked_prefill: bool = True):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        self.enable_chunked_prefill = enable_chunked_prefill
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


    def _schedule_chunked_prefill(self) -> tuple[list[ScheduledSequence], bool]:
        scheduled_sequences: list[ScheduledSequence] = []
        scheduled_set: set[Sequence] = set()
        current_scheduled_tokens = 0
        is_prefill = False
        preempted = False

        # vLLM V1 style: schedule RUNNING requests first. Decode consumes one
        # token; an unfinished prompt consumes a chunk from the remaining budget.
        req_index = 0
        while req_index < len(self.running):
            if current_scheduled_tokens >= self.max_num_batched_tokens:
                break
            if len(scheduled_sequences) >= self.max_num_sequences:
                break

            seq = self.running[req_index]
            remaining_budget = self.max_num_batched_tokens - current_scheduled_tokens
            if seq.num_computed_tokens < seq.num_prompt_tokens:
                num_new_tokens = min(
                    seq.num_prompt_tokens - seq.num_computed_tokens,
                    remaining_budget,
                )
                if num_new_tokens <= 0:
                    break
                scheduled_sequences.append(ScheduledSequence(seq, num_new_tokens))
                scheduled_set.add(seq)
                current_scheduled_tokens += num_new_tokens
                is_prefill = True
                req_index += 1
                continue

            if not self.block_manager.can_append(seq):
                preempted_seq = self.running.pop()
                self.preempt(preempted_seq)
                preempted = True
                if preempted_seq is seq:
                    break
                continue

            self.block_manager.append(seq)
            scheduled_sequences.append(ScheduledSequence(seq, 1))
            scheduled_set.add(seq)
            current_scheduled_tokens += 1
            req_index += 1

        # Use leftover token budget for WAITING prefills. A long prompt is admitted
        # even when only a chunk fits, then continued from running in later steps.
        while self.waiting and not preempted:
            if current_scheduled_tokens >= self.max_num_batched_tokens:
                break
            if len(scheduled_sequences) >= self.max_num_sequences:
                break
            if len(self.running) >= self.max_num_sequences:
                break

            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break

            seq = self.waiting.popleft()
            self.block_manager.allocate(seq)
            seq.num_computed_tokens = max(seq.num_computed_tokens, seq.num_cached_tokens)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

            remaining_budget = self.max_num_batched_tokens - current_scheduled_tokens
            num_new_tokens = min(seq.num_prompt_tokens - seq.num_computed_tokens, remaining_budget)
            if num_new_tokens <= 0:
                continue
            scheduled_sequences.append(ScheduledSequence(seq, num_new_tokens))
            scheduled_set.add(seq)
            current_scheduled_tokens += num_new_tokens
            is_prefill = True

        # Keep unscheduled running requests behind the scheduled ones in FCFS order.
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
                seq.num_computed_tokens = max(seq.num_computed_tokens, seq.num_cached_tokens)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
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
        token_iter = iter(token_ids or [])
        for item in seqs:
            seq = item.seq
            was_prefilling = seq.num_computed_tokens < seq.num_prompt_tokens
            seq.num_computed_tokens += item.num_scheduled_tokens
            if was_prefilling and seq.num_computed_tokens < seq.num_prompt_tokens:
                continue

            token_id = next(token_iter)
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
