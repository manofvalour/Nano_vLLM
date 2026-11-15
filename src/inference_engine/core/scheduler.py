from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference_engine.core.block_manager import BlockManager

from inference_engine.core.sequence import (
    SchedulerOutputs,
    Sequence,
    SequenceGroup,
    SequenceStatus,
)


class Scheduler:
    """scheduler for waiting, running, and preempted sequences.

    The default policy design is that the running decode sequences are
    admitted first, then waiting prompts are added as bounded prefill chunks.
    """
    def __init__(
        self,
        *,
        block_size: int = 16,
        max_num_batched_tokens: int = 4096,
        max_num_seqs: int = 256,
        max_prefill_chunk: int = 2048,
        enable_chunked_prefill: bool = True,
        enable_preemption: bool = True,
        block_manager: BlockManager | None = None,
    ) -> None:
        self.block_size = int(block_size)
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.max_num_seqs = int(max_num_seqs)
        self.max_prefill_chunk = int(max_prefill_chunk)
        self.enable_chunked_prefill = bool(enable_chunked_prefill)
        self.enable_preemption = bool(enable_preemption)
        self.block_manager = block_manager

        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be > 0")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be > 0")
        if self.max_prefill_chunk <= 0:
            raise ValueError("max_prefill_chunk must be > 0")

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self._groups: list[SequenceGroup] = []

    def _add_sequence_group(self, seq_group: SequenceGroup | list[Sequence] | Sequence) -> None:
        """Register a sequence group or sequence for scheduling."""

        if seq_group is None:
            return

        if isinstance(seq_group, Sequence):
            seq_group = SequenceGroup(request_id=seq_group.seq_id, sequences=[seq_group])
        elif isinstance(seq_group, list):
            if not seq_group:
                return
            seq_group = SequenceGroup(
                request_id=f"group-{len(self._groups)}",
                sequences=list(seq_group),
            )

        if seq_group in self._groups:
            return

        self._groups.append(seq_group)
        for seq in seq_group.sequences:
            if seq.is_finished:
                continue
            if seq.status == SequenceStatus.RUNNING:
                if seq not in self.running:
                    self.running.append(seq)
                continue
            if (seq.status == SequenceStatus.WAITING and seq not in self.waiting) or (
                seq.status == SequenceStatus.SWAPPED and seq not in self.waiting
            ):
                self.waiting.append(seq)

    def seq_group(self, seq_group: SequenceGroup | list[Sequence] | Sequence) -> None:
        """Compatibility wrapper for callers that use a ``seq_group`` method name."""

        self._add_sequence_group(seq_group)

    def _remaining_prompt_tokens(self, seq: Sequence) -> int:
        """Tokens still needing prefill for the current request."""
        if seq is None:
            return 0
        computed = max(0, int(getattr(seq, "num_computed_tokens", 0)))
        return max(0, seq.prompt_len - computed)

    def _prefill_chunk_tokens(self, seq: Sequence, remaining_budget: int) -> int:
        """Return the capped prefill chunk size for a single waiting sequence."""
        remaining_prompt = self._remaining_prompt_tokens(seq)
        if remaining_prompt <= 0 or remaining_budget <= 0:
            return 0

        cap = self.max_prefill_chunk if self.enable_chunked_prefill else remaining_prompt
        return max(1, min(remaining_prompt, cap, remaining_budget))

    def _can_allocate_for_prefill(self, seq: Sequence, *, chunk_tokens: int | None = None) -> bool:
        if self.block_manager is None:
            return True
        if chunk_tokens is None:
            chunk_tokens = self._remaining_prompt_tokens(seq)
        needed = 0 if chunk_tokens <= 0 else max(1, math.ceil(chunk_tokens / self.block_size))
        return self.block_manager.can_allocate(needed)

    def _allocation_target(self, seq: Sequence, *, chunk_tokens: int | None = None) -> int:
        if chunk_tokens is None:
            chunk_tokens = self._remaining_prompt_tokens(seq)
        return 0 if chunk_tokens <= 0 else max(1, math.ceil(chunk_tokens / self.block_size))

    def _allocate_blocks_for_prefill(self, seq: Sequence, *, chunk_tokens: int | None = None) -> list[int]:
        if seq is None:
            return []

        if chunk_tokens is None:
            chunk_tokens = self._remaining_prompt_tokens(seq)
        if chunk_tokens <= 0:
            return list(seq.block_table)

        needed = self._allocation_target(seq, chunk_tokens=chunk_tokens)
        if self.block_manager is not None:
            current_needed = len(seq.block_table)
            additional = max(0, needed - current_needed)
            if additional > 0:
                allocated = self.block_manager.allocate(additional)
                seq.block_table.extend(int(block_id) for block_id in allocated.tolist())
            return list(seq.block_table)

        return list(seq.block_table)

    def _schedule_running(self, outputs: SchedulerOutputs) -> None:
        """Promote currently-running sequences to the decode batch."""

        for seq in list(self.running):
            if seq.is_finished:
                self.running.remove(seq)
                continue

            if seq.status == SequenceStatus.RUNNING:
                outputs.decode_seqs.append(seq)
                outputs.num_batched_tokens += max(1, seq.total_len)
                continue

            if seq.status == SequenceStatus.SWAPPED:
                seq.status = SequenceStatus.RUNNING
                outputs.swapped_in_seqs.append(seq)
                outputs.num_batched_tokens += max(1, seq.total_len)

    def _schedule_waiting(self, outputs: SchedulerOutputs) -> None:
        """Admit waiting sequences into prefill or chunked-prefill work."""

        if len(self.running) + len(outputs.decode_seqs) >= self.max_num_seqs:
            return

        candidates = sorted(
            [seq for seq in self.waiting if not seq.is_finished],
            key=lambda seq: (seq.priority, seq.arrival_time, seq.seq_id),
        )
        for seq in candidates:
            if seq.is_finished:
                self.waiting.remove(seq)
                continue

            if seq.status not in {SequenceStatus.WAITING, SequenceStatus.SWAPPED}:
                self.waiting.remove(seq)
                continue

            if outputs.num_batched_tokens >= self.max_num_batched_tokens:
                break

            remaining_prompt = self._remaining_prompt_tokens(seq)
            if remaining_prompt <= 0:
                self.waiting.remove(seq)
                seq.status = SequenceStatus.RUNNING
                if seq not in self.running:
                    self.running.append(seq)
                outputs.decode_seqs.append(seq)
                outputs.num_batched_tokens += max(1, seq.total_len)
                continue

            remaining_budget = max(0, self.max_num_batched_tokens - outputs.num_batched_tokens)
            chunk_tokens = self._prefill_chunk_tokens(seq, remaining_budget)
            if chunk_tokens <= 0:
                break

            if not self._can_allocate_for_prefill(seq, chunk_tokens=chunk_tokens):
                break

            self._allocate_blocks_for_prefill(seq, chunk_tokens=chunk_tokens)
            seq.status = SequenceStatus.RUNNING
            if seq not in self.running:
                self.running.append(seq)
            if self.enable_chunked_prefill and remaining_prompt > chunk_tokens:
                outputs.chunked_prefill_seqs.append(seq)
            else:
                outputs.prefill_seqs.append(seq)
            outputs.num_batched_tokens += chunk_tokens
            self.waiting.remove(seq)

    def _handle_preemption(self, outputs: SchedulerOutputs) -> None:
        """Preempt lower-priority running work when higher-priority work is waiting."""

        if not self.enable_preemption:
            return

        if not self.waiting or not self.running:
            return

        waiting_priorities = [seq.priority for seq in self.waiting if not seq.is_finished]
        running_priorities = [seq.priority for seq in self.running if not seq.is_finished]
        if not waiting_priorities or not running_priorities:
            return

        highest_waiting = max(waiting_priorities)
        lowest_running = min(running_priorities)
        if highest_waiting <= lowest_running:
            return

        for seq in sorted(
            [s for s in self.running if not s.is_finished],
            key=lambda item: (item.priority, item.arrival_time, item.seq_id),
        ):
            if seq.priority >= highest_waiting:
                continue

            seq.reset_for_preemption()
            seq.status = SequenceStatus.WAITING
            self.running.remove(seq)
            if seq not in self.waiting:
                self.waiting.append(seq)
            outputs.preempted_seqs.append(seq)

    def schedule(self) -> SchedulerOutputs:
        """Return the next batch of sequences to run."""

        outputs = SchedulerOutputs()
        self._handle_preemption(outputs)
        self._schedule_running(outputs)
        if outputs.num_batched_tokens < self.max_num_batched_tokens:
            self._schedule_waiting(outputs)
        return outputs

    def post_process(self, outputs: SchedulerOutputs | None = None) -> SchedulerOutputs:
        """Finalize scheduling results and clear finished work."""

        if outputs is None:
            outputs = SchedulerOutputs()

        scheduled = [
            *outputs.prefill_seqs,
            *outputs.decode_seqs,
            *outputs.chunked_prefill_seqs,
            *outputs.preempted_seqs,
            *outputs.swapped_in_seqs,
            *outputs.swapped_out_seqs,
            *outputs.ignored_seqs,
        ]

        for seq in scheduled:
            if seq.is_finished:
                if seq in self.running:
                    self.running.remove(seq)
                if seq in self.waiting:
                    self.waiting.remove(seq)
                continue
            if seq.status == SequenceStatus.RUNNING:
                seq.num_computed_tokens = max(seq.num_computed_tokens, seq.prompt_len)

        return outputs

    def has_unfinished_sequence(self) -> bool:
        return any(not seq.is_finished
                   for group in self._groups
                   for seq in group.sequences)

    def get_num_unfinished_sequence(self) -> int:
        return sum(
            1
            for group in self._groups
            for seq in group.sequences
            if not seq.is_finished
        )

    def __len__(self) -> int:
        return len(self._groups)
