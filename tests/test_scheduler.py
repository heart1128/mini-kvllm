import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from collections import deque
from unittest.mock import MagicMock
from myvllm.engine.scheduler import Scheduler, ScheduledSequence
from myvllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(
    max_num_batched_tokens=100,
    max_num_sequences=10,
    max_cached_blocks=100,
    block_size=4,
    enable_chunked_prefill=True,
    long_prefill_token_threshold=0,
    max_num_partial_prefills=None,
    max_long_partial_prefills=None,
):
    return Scheduler(
        max_num_sequences=max_num_sequences,
        max_num_batched_tokens=max_num_batched_tokens,
        max_cached_blocks=max_cached_blocks,
        block_size=block_size,
        eos=0,
        enable_chunked_prefill=enable_chunked_prefill,
        long_prefill_token_threshold=long_prefill_token_threshold,
        max_num_partial_prefills=max_num_partial_prefills,
        max_long_partial_prefills=max_long_partial_prefills,
    )


def inject_running(scheduler: Scheduler, *seqs: Sequence):
    """Put sequences directly into the running queue, bypassing prefill."""
    for seq in seqs:
        seq.status = SequenceStatus.RUNNING
        seq.num_computed_tokens = seq.num_prompt_tokens
        scheduler.running.append(seq)


def all_tracked(scheduler: Scheduler, scheduled: list[ScheduledSequence]) -> set:
    """Return the set of all sequences the scheduler currently knows about."""
    return set(scheduler.running) | set(scheduler.waiting) | {item.seq for item in scheduled}


class TestBug2TokenLimitBreak:
    """
    Setup: 3 sequences in running, all can_append=True.
           max_num_batched_tokens=2 → only 2 fit per step.
    Expected after schedule(): seq_c should still be in running.
    Buggy behaviour: seq_c is popleft-ed, limit is hit, break fires,
                     seq_c is never restored → permanently lost.
    """

    def _run(self, scheduler: Scheduler):
        seq_a = Sequence([1, 2, 3])
        seq_b = Sequence([4, 5, 6])
        seq_c = Sequence([7, 8, 9])
        inject_running(scheduler, seq_a, seq_b, seq_c)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None

        scheduled, is_prefill = scheduler.schedule()

        return seq_a, seq_b, seq_c, scheduled, is_prefill

    def test_seq_count_is_correct(self):
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq_a, seq_b, seq_c, scheduled, is_prefill = self._run(scheduler)

        assert not is_prefill
        # Only 2 tokens fit in the batch
        assert len(scheduled) == 2

        # ---- THE BUG: seq_c disappears ----
        # After extendleft, running should be [seq_a, seq_b, seq_c]
        assert seq_c in scheduler.running, (
            "Bug 2: seq_c was popleft-ed and the break fired before it could be "
            "added to scheduled_sequences or put back into self.running → LOST"
        )

    def test_seq_count_limit_variant(self):
        """Same bug but triggered by max_num_sequences instead of token budget."""
        scheduler = make_scheduler(max_num_sequences=2, max_num_batched_tokens=100)
        seq_a, seq_b, seq_c, scheduled, is_prefill = self._run(scheduler)

        assert not is_prefill
        assert len(scheduled) == 2

        assert seq_c in scheduler.running, (
            "Bug 2 (seq-count variant): seq_c lost when len(scheduled_sequences) "
            ">= max_num_sequences caused the break"
        )

    def test_no_sequence_is_lost(self):
        """Total universe of sequences must be conserved."""
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq_a, seq_b, seq_c, scheduled, is_prefill = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        for seq in (seq_a, seq_b, seq_c):
            assert seq in tracked, f"seq {seq.seq_id} disappeared from the scheduler"


class TestBug1CanAppendFailure:
    """
    Setup: 2 sequences in running, can_append returns False for the first.
    Expected: seq_a is either put back into running (to retry later) or
              preempted into waiting; it must NOT disappear entirely.
    Buggy behaviour: seq_a is popleft-ed, can_append fails, the code does
                     self.preempt(self.running.pop()) which preempts seq_b,
                     but seq_a is never handled → lost.
    """

    def _run(self, scheduler: Scheduler):
        seq_a = Sequence([1, 2, 3])
        seq_b = Sequence([4, 5, 6])
        inject_running(scheduler, seq_a, seq_b)

        mock_bm = MagicMock()
        # First call (for seq_a): cannot append; subsequent calls: True
        mock_bm.can_append.side_effect = [False, True, True, True]
        mock_bm.append.return_value = None
        mock_bm.deallocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, is_prefill = scheduler.schedule()
        return seq_a, seq_b, scheduled, is_prefill

    def test_seq_a_not_lost(self):
        scheduler = make_scheduler()
        seq_a, seq_b, scheduled, is_prefill = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked, (
            "Bug 1: seq_a was popleft-ed, can_append returned False, "
            "self.preempt(self.running.pop()) preempted seq_b instead, "
            "and seq_a was never restored → LOST"
        )

    def test_total_conservation(self):
        """Neither seq must disappear."""
        scheduler = make_scheduler()
        seq_a, seq_b, scheduled, is_prefill = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked, f"seq_a disappeared"
        assert seq_b in tracked, f"seq_b disappeared"


class TestChunkedPrefillMixedScheduling:
    def test_running_decode_is_scheduled_before_waiting_prefill(self):
        scheduler = make_scheduler(max_num_batched_tokens=4, max_num_sequences=4)
        running = Sequence([1, 2, 3])
        running.num_computed_tokens = running.num_tokens
        waiting = Sequence([4, 5, 6, 7, 8])
        inject_running(scheduler, running)
        scheduler.add_sequence(waiting)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None
        scheduler.block_manager.can_allocate.return_value = True
        scheduler.block_manager.allocate.return_value = None

        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        assert [item.seq for item in scheduled] == [running, waiting]
        assert [item.num_scheduled_tokens for item in scheduled] == [1, 3]
        assert waiting.status == SequenceStatus.RUNNING
        assert waiting in scheduler.running

    def test_long_waiting_prompt_is_chunked_across_steps(self):
        scheduler = make_scheduler(max_num_batched_tokens=3, max_num_sequences=2)
        seq = Sequence([1, 2, 3, 4, 5, 6, 7])
        scheduler.add_sequence(seq)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_allocate.return_value = True
        scheduler.block_manager.allocate.return_value = None

        first, first_is_prefill = scheduler.schedule()
        assert first_is_prefill
        assert first == [ScheduledSequence(seq, 3)]

        scheduler.postprocess(first, [])
        assert seq.num_computed_tokens == 3
        assert seq in scheduler.running

        second, second_is_prefill = scheduler.schedule()
        assert second_is_prefill
        assert second == [ScheduledSequence(seq, 3)]

    def test_partial_prefill_postprocess_does_not_append_token(self):
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq = Sequence([1, 2, 3, 4])
        seq.status = SequenceStatus.RUNNING
        scheduled = [ScheduledSequence(seq, 2)]

        scheduler.postprocess(scheduled, [])

        assert seq.token_ids == [1, 2, 3, 4]
        assert seq.num_computed_tokens == 2
        assert seq.status == SequenceStatus.RUNNING

    def test_final_prefill_chunk_appends_sampled_token(self):
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq = Sequence([1, 2, 3, 4])
        seq.status = SequenceStatus.RUNNING
        seq.num_computed_tokens = 2
        scheduled = [ScheduledSequence(seq, 2)]

        scheduler.postprocess(scheduled, [9])

        assert seq.token_ids == [1, 2, 3, 4, 9]
        assert seq.num_computed_tokens == 4
        assert seq.status == SequenceStatus.RUNNING

    def test_disabled_chunked_prefill_does_not_split_long_waiting_prompt(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=3,
            max_num_sequences=2,
            enable_chunked_prefill=False,
        )
        seq = Sequence([1, 2, 3, 4, 5])
        scheduler.add_sequence(seq)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_allocate.return_value = True

        scheduled, is_prefill = scheduler.schedule()

        assert not is_prefill
        assert scheduled == []
        assert seq in scheduler.waiting
        scheduler.block_manager.allocate.assert_not_called()

    def test_disabled_chunked_prefill_keeps_prefill_only_when_prompt_fits(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=10,
            max_num_sequences=4,
            enable_chunked_prefill=False,
        )
        running = Sequence([1, 2, 3])
        waiting = Sequence([4, 5, 6])
        inject_running(scheduler, running)
        scheduler.add_sequence(waiting)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_allocate.return_value = True
        scheduler.block_manager.allocate.return_value = None
        scheduler.block_manager.can_append.return_value = True

        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        assert scheduled == [ScheduledSequence(waiting, 3)]
        assert running not in [item.seq for item in scheduled]
        scheduler.block_manager.append.assert_not_called()


class TestChunkedPrefillPolicy:
    def test_long_prefill_threshold_limits_running_chunk_size(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=10,
            long_prefill_token_threshold=3,
        )
        seq = Sequence(list(range(20)))
        seq.status = SequenceStatus.RUNNING
        seq.num_computed_tokens = 4
        scheduler.running.append(seq)

        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        assert scheduled == [ScheduledSequence(seq, 3)]

    def test_long_prefill_threshold_limits_waiting_admission_chunk_size(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=10,
            long_prefill_token_threshold=4,
        )
        seq = Sequence(list(range(20)))
        scheduler.add_sequence(seq)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_allocate.return_value = True
        scheduler.block_manager.allocate.return_value = None

        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        assert scheduled == [ScheduledSequence(seq, 4)]
        assert seq in scheduler.running

    def test_max_num_partial_prefills_blocks_new_partial_admission(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=8,
            long_prefill_token_threshold=4,
            max_num_partial_prefills=1,
        )
        running_partial = Sequence(list(range(20)))
        running_partial.status = SequenceStatus.RUNNING
        running_partial.num_computed_tokens = 4
        scheduler.running.append(running_partial)
        waiting = Sequence(list(range(20, 40)))
        scheduler.add_sequence(waiting)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_allocate.return_value = True

        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        assert scheduled == [ScheduledSequence(running_partial, 4)]
        assert waiting in scheduler.waiting
        assert waiting not in scheduler.running
        scheduler.block_manager.allocate.assert_not_called()

    def test_max_long_partial_prefills_blocks_new_long_partial_admission_only(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=16,
            long_prefill_token_threshold=4,
            max_long_partial_prefills=1,
        )
        running_long_partial = Sequence(list(range(20)))
        running_long_partial.status = SequenceStatus.RUNNING
        running_long_partial.num_computed_tokens = 4
        scheduler.running.append(running_long_partial)
        waiting_long = Sequence(list(range(100, 120)))
        scheduler.add_sequence(waiting_long)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_allocate.return_value = True

        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        assert scheduled == [ScheduledSequence(running_long_partial, 4)]
        assert waiting_long in scheduler.waiting
        assert waiting_long not in scheduler.running
        scheduler.block_manager.allocate.assert_not_called()

    def test_max_long_partial_prefills_allows_short_full_prefill_admission(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=16,
            long_prefill_token_threshold=4,
            max_long_partial_prefills=1,
        )
        running_long_partial = Sequence(list(range(20)))
        running_long_partial.status = SequenceStatus.RUNNING
        running_long_partial.num_computed_tokens = 4
        scheduler.running.append(running_long_partial)
        waiting_short = Sequence([100, 101, 102])
        scheduler.add_sequence(waiting_short)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_allocate.return_value = True
        scheduler.block_manager.allocate.return_value = None

        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        assert scheduled == [
            ScheduledSequence(running_long_partial, 4),
            ScheduledSequence(waiting_short, 3),
        ]
        assert waiting_short in scheduler.running

    def test_running_order_matches_vllm_without_forcing_decode_first(self):
        scheduler = make_scheduler(
            max_num_batched_tokens=5,
            long_prefill_token_threshold=4,
        )
        prefill = Sequence(list(range(20)))
        prefill.status = SequenceStatus.RUNNING
        prefill.num_computed_tokens = 4
        decode = Sequence([101, 102, 103])
        inject_running(scheduler, decode)
        scheduler.running = deque([prefill, decode])

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None

        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        assert scheduled == [ScheduledSequence(prefill, 4), ScheduledSequence(decode, 1)]


class TestSchedulerHappyPath:
    def test_prefill_scheduled_first(self):
        scheduler = make_scheduler(max_num_batched_tokens=100, max_cached_blocks=50)
        seq = Sequence([1, 2, 3, 4])
        scheduler.add_sequence(seq)

        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill
        assert seq in [item.seq for item in scheduled]
        assert seq in scheduler.running

    def test_all_running_seqs_scheduled_when_budget_allows(self):
        scheduler = make_scheduler(max_num_batched_tokens=10)
        seq_a = Sequence([1])
        seq_b = Sequence([2])
        inject_running(scheduler, seq_a, seq_b)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None

        scheduled, is_prefill = scheduler.schedule()
        assert not is_prefill
        assert len(scheduled) == 2
        # Both should be back in running after schedule()
        assert seq_a in scheduler.running
        assert seq_b in scheduler.running

    def test_preempt_only_seq_when_cant_append_and_running_empty(self):
        """Original else-branch: running=[seq], can_append=False → preempt seq → waiting."""
        scheduler = make_scheduler()
        seq = Sequence([1, 2])
        inject_running(scheduler, seq)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = False
        scheduler.block_manager.deallocate.return_value = None

        scheduled, is_prefill = scheduler.schedule()
        assert not is_prefill
        assert len(scheduled) == 0
        assert seq in scheduler.waiting
        assert seq.status == SequenceStatus.WAITING
