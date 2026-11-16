from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol
import time

@dataclass(frozen=True)
class Request:
    req_id:str
    prompt_token_count:int
    max_new_tokens:int
    priority:int=0
    slo_ttft_ms:float | None=None

@dataclass(frozen=True)
class WorkloadSnapshot:
    remaining_prefill_tokens:int
    remaining_decode_tokens:int
    priority:int

@dataclass(frozen=True)
class ThroughputEstimate:
    length_bucket:int
    prefill_tokens_per_sec:float
    decode_tokens_per_sec:float

@dataclass(frozen=True)
class SchedulingSnapshot:
    waiting:tuple[WorkloadSnapshot, ...]
    running:tuple[WorkloadSnapshot, ...]
    throughput_by_bucket: tuple[ThroughputEstimate, ...]
    captured_at:float

@dataclass(frozen=True)
class AdmissionDecision:
    allowed:bool
    reason:str|None=None
    estimated_wait_ms: float |None=None
    required_blocks:int=0

class WaitEstimatorProtocol(Protocol):
    def estimated_wait(
            self,
            req:Request,
            snapshot:SchedulingSnapshot,
    )->float:
        """Return the estimated wait before the request
        can start prefill, in ms"""

        ...

class SchedulingSnapshotProvider(Protocol):
    def create_admission_snapshot(self)->SchedulingSnapshot:
        """Return an immutable view of the curreint scheduling state"""
        ...

class ThroughputTracker:
    """Uses a rolling exponential average of prefill and decode throughput,
    keyed by the sequence-length bucket"""
    def __init__(self,
                 *,
                 alpha:float=0.2,
                 default_prefill_tps:float = 1000.0,
                 default_decode_tps:float = 500.0
        )->None:
        if  not 0.0<alpha<=1.0:
            raise ValueError("alpha must be between [0,1]")

        self.alpha=alpha
        self.default_prefill_tps=default_prefill_tps
        self.default_decode_tps=default_decode_tps

        self._rates: dict[int, tuple[float,float]]={}

    @staticmethod
    def length_bucket(token_count:int)->int:
        if token_count<=0:
            return 0
        return 1<<(token_count -1).bit_length()

    def update(
            self,
            sequence_length:int,
            *,
            prefill_tokens_per_sec:float,
            decode_tokens_per_sec:float,
    )-> None:
        if prefill_tokens_per_sec <= 0 or decode_tokens_per_sec<=0:
            raise ValueError("Throughput values must be positive")
        bucket = self.length_bucket(sequence_length)
        previous = self._rates.get(bucket)

        if previous is None:
            self._rates[bucket] = (
                prefill_tokens_per_sec,
                decode_tokens_per_sec,
            )
            return
        
        old_prefill, old_decode = previous
        a = self.alpha

        self._rates[bucket]= (
            a*prefill_tokens_per_sec + (1-a)*old_prefill,
            a*decode_tokens_per_sec + (1-a)*old_decode,
        )

    def snapshot(self) -> tuple[ThroughputEstimate, ...]:
        estimates = [
            ThroughputEstimate(
                length_bucket=bucket,
                prefill_tokens_per_sec=prefill_tps,
                decode_tokens_per_sec=decode_tps,
            )
            for bucket,(prefill_tps, decode_tps) in self._rates.items()
        ]

        if not estimates:
            estimates.append(
                ThroughputEstimate(
                    length_bucket=0,
                    prefill_tokens_per_sec=self.default_prefill_tps,
                    decode_tokens_per_sec=self.default_decode_tps,
                )
            )
        return tuple(sorted(estimates,
                            key = lambda item: item.length_bucket))


class WaitEstimator:
    def _select_throughput(
            self,
            req:Request,
            snapshot:SchedulingSnapshot,
    )-> ThroughputEstimate:
        if not snapshot.throughput_by_bucket:
            raise ValueError("Snapshot has no throughput estimates")

        target = ThroughputTracker.length_bucket(req.prompt_token_count)
        return min(
            snapshot.throughput_by_bucket,
            key=lambda estimate: abs(estimate.length_bucket - target),
        )

    def estimated_wait(
            self,
            req:Request,
            snapshot:SchedulingSnapshot,
    )-> float:
        rates = self._select_throughput(req,snapshot)

        prefill_tps = rates.prefill_tokens_per_sec
        decode_tps = rates.decode_tokens_per_sec

        if prefill_tps<=0 or decode_tps<=0:
            raise ValueError("the throughput estimates must be positive")

        # Running requests contribute outstanding work.
        work_ahead = list(snapshot.running)

        # Snapshot order preserves FIFO order within a priority class.
        # Lower number means higher priority.
        for item in snapshot.waiting:
            if item.priority <= req.priority:
                work_ahead.append(item)
        prefill_seconds = sum(
            item.remaining_prefill_tokens/prefill_tps
            for item in work_ahead
        )

        decode_seconds = sum(
            item.remaining_decode_tokens/decode_tps
            for item in work_ahead
        )

        return (prefill_seconds+decode_seconds)*1000.0

 # Implementing the Admission policy
class BlockCapacity(Protocol):
    @property
    def num_free_block(self)->int:
        ...

@dataclass(frozen=True)
class AdmissionConfig:
    block_size:int=16

class AdmissionPolicy:
    def __init__(self,
                 wait_estimator:WaitEstimator,
                 config:AdmissionConfig | None=None,
                 )->None:
        self.wait_estimator = wait_estimator
        self.config = config or AdmissionConfig()

        if self.config.block_size <=0:
            raise ValueError("block_size must be positive")

    def required_blocks(self, req:Request)->int:
        total_tokens = req.prompt_token_count + req.max_new_tokens

        if total_tokens <=0:
            return 0

        return math.ceil(total_tokens/self.config.block_size)

    def should_admit(
            self,
            req:Request,
            snapshot:SchedulingSnapshot,
            block_manager:BlockCapacity,
    )->AdmissionDecision:
        if req.prompt_token_count<0:
            return AdmissionDecision(False, reason="invalid prompt token count")

        if req.max_new_tokens < 0:
            return AdmissionDecision(False, reason="invalid max_new_tokens")

        if req.slo_ttft_ms is not None and req.slo_ttft_ms<0:
            return AdmissionDecision(False, reason="invalid TTFT SLO")

        estimated_wait_ms = self.wait_estimator.estimated_wait(
            req, snapshot,
        )

        required = self.required_blocks(req)

        if (
            req.slo_ttft_ms is not None
            and estimated_wait_ms>req.slo_ttft_ms
        ):
            return AdmissionDecision(
                allowed=False,
                reason = "estimated wait exceeds TTFT SLO",
                estimated_wait_ms=estimated_wait_ms,
                required_blocks=required
            )

        if block_manager.num_free_block<required:
            return AdmissionDecision(
                allowed=False,
                reason="KV-block budget exceeded",
                estimated_wait_ms=estimated_wait_ms,
                required_blocks=required
            )

        return AdmissionDecision(
            allowed=True,
            estimated_wait_ms=estimated_wait_ms,
            required_blocks=required
        )


class SequenceWorkview(Protocol):
    prompt_len:int
    num_computed_tokens:int
    remaining_decode_tokens:int
    priority:int

class SchedulerSnapshotMixin:
    def _snapshot_sequence(
        self,
        seq:SequenceWorkview,
    )->WorkloadSnapshot:
        remaining_prefill = max(
            0, seq.prompt_len-seq.num_computed_tokens
        )

        return WorkloadSnapshot(
            remaining_prefill_tokens=remaining_prefill,
            remaining_decode_tokens=max(
                0,seq.remaining_decode_tokens
            ),
            priority= seq.priority
        )

    def create_admission_snapshot(self)->SchedulingSnapshot:
        waiting = tuple(self._snapshot_sequence(seq)
        for seq in self.waiting)

        running = tuple(
            self._snapshot_sequence(seq)
            for seq in self.running
        )

        return SchedulingSnapshot(
            waiting=waiting,
            running=running,
            throughput_by_bucket=self.throughput_tracker.snapshot(),
            captured_at=time.monotonic()
        )

class Frontend:
    def __init__(self,
                 admission_policy:AdmissionPolicy,
                 snapshot_provider:SchedulingSnapshotProvider,
                 block_manager:BlockCapacity,
    )->None:
        self.admission_policy = admission_policy
        self.snapshot_provider = snapshot_provider
        self.block_manager = block_manager

    def submit(self,req:Request)-> AdmissionDecision:
        snapshot = (
            self.snapshot_provider.create_admission_snapshot()
        )

        decision = self.admission_policy.should_admit(
            req, snapshot, self.block_manager
        )

        return decision