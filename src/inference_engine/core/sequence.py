from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any


class SequenceStatus(StrEnum, Enum):
    """Lifecycle states for a generation request."""

    WAITING = "WAITING"
    RUNNING = "RUNNING"
    SWAPPED = "SWAPPED"
    FINISHED_STOPPED = "FINISHED_STOPPED" #generated EOS token
    FINISHED_LENGTH_CAPPED = "FINISHED_LENGTH_CAPPED" #reached the max_token
    FINISHED_ABORTED = "FINISHED_ABORTED" #generation failed
    FINISHED_IGNORED = "FINISHED_IGNORED" #generating ignored either because of preemption


_FINISHED_STATES = {
    SequenceStatus.FINISHED_STOPPED,
    SequenceStatus.FINISHED_LENGTH_CAPPED,
    SequenceStatus.FINISHED_ABORTED,
    SequenceStatus.FINISHED_IGNORED,
}


@dataclass
class Sequence:
    """Represents a single generation sequence.

    The implementation exposes the lifecycle and bookkeeping fields described
    in the implementation plan while staying compatible with dict-based and
    object-based sampling configs.
    """

    seq_id: int
    prompt: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    sampling_params: Any | None = None
    block_table: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    num_cached_tokens: int = 0
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    cumulative_logprob: float = 0.0
    finish_reason: str | None = None
    priority: int = 0
    arrival_time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        #prompt
        if self.prompt_token_ids is None:
            self.prompt_token_ids = []
        elif not isinstance(self.prompt_token_ids, list):
            self.prompt_token_ids = list(self.prompt_token_ids)
        self.prompt_token_ids = [int(token) for token in self.prompt_token_ids]

        #block
        if self.block_table is None:
            self.block_table = []
        elif not isinstance(self.block_table, list):
            self.block_table = list(self.block_table)
        self.block_table = [int(block_id) for block_id in self.block_table]

        # generated tokens
        if self.output_token_ids is None:
            self.output_token_ids = []
        elif not isinstance(self.output_token_ids, list):
            self.output_token_ids = list(self.output_token_ids)
        self.output_token_ids = [int(token) for token in self.output_token_ids]

        #logprob of output tokens
        if self.output_logprobs is None:
            self.output_logprobs = []
        elif not isinstance(self.output_logprobs, list):
            self.output_logprobs = list(self.output_logprobs)
        self.output_logprobs = [float(lp) for lp in self.output_logprobs]

        self.num_cached_tokens = max(0, self.num_cached_tokens)
        self.num_computed_tokens = max(0, self.num_computed_tokens)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def is_prefill_done(self) -> bool:
        return self.num_computed_tokens >= self.prompt_len

    @property
    def is_finished(self) -> bool:
        return self.status in _FINISHED_STATES

    @property
    def num_new_tokens(self) -> int:
        return self.output_len

    def _sampling_param(self, key: str, default: Any = None) -> Any:
        if self.sampling_params is None:
            return default
        if isinstance(self.sampling_params, dict):
            return self.sampling_params.get(key, default)
        return getattr(self.sampling_params, key, default)

    def _stop_token_ids(self) -> set[int]:
        stop = self._sampling_param("stop_token_ids", None)
        if stop is None:
            stop = self._sampling_param("stop", None)
        if stop is None:
            return set()
        if isinstance(stop, Iterable) and not isinstance(stop, (str, bytes)):
            return {int(token_id) for token_id in stop}
        return {int(stop)}

    def append_token(self, token_id: int, logprob: float | None = None) -> None:
        """Append a newly generated token to the sequence."""

        self.output_token_ids.append(int(token_id))
        if logprob is not None:
            self.output_logprobs.append(float(logprob))
            self.cumulative_logprob += float(logprob)

    def maybe_append_block(self, block_id: int | None) -> bool:
        """Append a block ID to the block table if it is new."""

        if block_id is None:
            return False

        block_id = int(block_id)
        if not self.block_table or self.block_table[-1] != block_id:
            self.block_table.append(block_id)
            return True
        return False

    def check_finished(self) -> bool:
        """Set a finished status when generation should stop."""

        if self.is_finished:
            return True

        stop_token_ids = self._stop_token_ids()
        if self.output_token_ids and self.output_token_ids[-1] in stop_token_ids:
            self.status = SequenceStatus.FINISHED_STOPPED
            self.finish_reason = "stop_token"
            return True

        max_tokens = self._sampling_param("max_tokens", None)
        if max_tokens is not None and self.num_new_tokens >= int(max_tokens):
            self.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            self.finish_reason = "max_length"
            return True

        return False

    def reset_for_preemption(self) -> None:
        """Move a running sequence back into the waiting pool."""

        self.status = SequenceStatus.WAITING
        self.num_cached_tokens = max(0, self.num_cached_tokens)
        self.finish_reason = None

    def __len__(self) -> int:
        return self.total_len

    def __repr__(self) -> str:
        return (
            f"Sequence(seq_id={self.seq_id}, status={self.status.value}, "
            f"prompt_len={self.prompt_len}, output_len={self.output_len}, "
            f"block_table={self.block_table})"
        )


@dataclass
class SequenceGroup:
    """Group of sequences that share the same request metadata."""

    request_id: Any
    sequences: list[Sequence] = field(default_factory=list)
    sampling_params: Any | None = None
    arrival_time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.sequences is None:
            self.sequences = []
        self.sequences = list(self.sequences)

    @property
    def is_finished(self) -> bool:
        return all(sequence.is_finished for sequence in self.sequences) if self.sequences else False

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    def add_sequence(self, sequence: Sequence) -> None:
        self.sequences.append(sequence)

    def __len__(self) -> int:
        return len(self.sequences)


@dataclass
class SchedulerOutputs:
    """Container for the scheduler's requested work for the next engine step."""

    prefill_seqs: list[Sequence] = field(default_factory=list)
    decode_seqs: list[Sequence] = field(default_factory=list)
    chunked_prefill_seqs: list[Sequence] = field(default_factory=list)
    preempted_seqs: list[Sequence] = field(default_factory=list)
    swapped_in_seqs: list[Sequence] = field(default_factory=list)
    swapped_out_seqs: list[Sequence] = field(default_factory=list)
    ignored_seqs: list[Sequence] = field(default_factory=list)
    num_batched_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "prefill_seqs",
            "decode_seqs",
            "chunked_prefill_seqs",
            "preempted_seqs",
            "swapped_in_seqs",
            "swapped_out_seqs",
            "ignored_seqs",
        ):
            value = getattr(self, field_name)
            if value is None:
                setattr(self, field_name, [])


__all__ = [
    "SchedulerOutputs",
    "Sequence",
    "SequenceGroup",
    "SequenceStatus",
]
