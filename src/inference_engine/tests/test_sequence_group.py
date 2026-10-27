import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference_engine.core.sequence import Sequence, SequenceGroup, SequenceStatus


def make_seq(seq_id: int = 1, status: SequenceStatus = SequenceStatus.WAITING) -> Sequence:
    return Sequence(seq_id=seq_id, prompt_token_ids=[1, 2], status=status)


def test_sequence_group_defaults():
    group = SequenceGroup(request_id="req-1")

    assert group.request_id == "req-1"
    assert group.sequences == []
    assert group.sampling_params is None
    assert group.num_sequences == 0
    assert len(group) == 0
    assert group.is_finished is False
    assert group.arrival_time > 0


def test_sequence_group_normalizes_sequences():
    empty = SequenceGroup(request_id="req-empty", sequences=None)
    assert empty.sequences == []

    from_tuple = SequenceGroup(request_id="req-tuple", sequences=(make_seq(1), make_seq(2)))
    assert isinstance(from_tuple.sequences, list)
    assert len(from_tuple) == 2

    source = [make_seq(3)]
    copied = SequenceGroup(request_id="req-copy", sequences=source)
    source.append(make_seq(4))
    assert len(copied) == 1


def test_sequence_group_add_sequence_updates_counts():
    group = SequenceGroup(request_id="req-1", sequences=[])
    first = make_seq(1)

    group.add_sequence(first)

    assert group.sequences == [first]
    assert group.num_sequences == 1
    assert len(group) == 1
    assert group.is_finished is False

    group.add_sequence(make_seq(2))
    assert group.num_sequences == 2
    assert len(group) == 2


def test_sequence_group_is_finished_only_when_all_sequences_finish():
    running = make_seq(1, status=SequenceStatus.RUNNING)
    stopped = make_seq(2, status=SequenceStatus.FINISHED_STOPPED)
    group = SequenceGroup(request_id="req-1", sequences=[running, stopped])

    assert group.is_finished is False

    running.status = SequenceStatus.FINISHED_LENGTH_CAPPED
    assert group.is_finished is True


@pytest.mark.parametrize(
    "status",
    [
        SequenceStatus.FINISHED_STOPPED,
        SequenceStatus.FINISHED_LENGTH_CAPPED,
        SequenceStatus.FINISHED_ABORTED,
        SequenceStatus.FINISHED_IGNORED,
    ],
)
def test_sequence_group_recognizes_all_finished_states(status):
    group = SequenceGroup(request_id="req-1", sequences=[make_seq(1, status=status)])
    assert group.is_finished is True


@pytest.mark.parametrize(
    "status",
    [SequenceStatus.WAITING, SequenceStatus.RUNNING, SequenceStatus.SWAPPED],
)
def test_sequence_group_unfinished_states_keep_group_open(status):
    group = SequenceGroup(request_id="req-1", sequences=[make_seq(1, status=status)])
    assert group.is_finished is False


def test_sequence_group_adding_sequence_after_completion_reopens_group():
    finished = make_seq(1, status=SequenceStatus.FINISHED_STOPPED)
    group = SequenceGroup(request_id="req-1", sequences=[finished])
    assert group.is_finished is True

    group.add_sequence(make_seq(2))
    assert group.is_finished is False


def test_sequence_group_preserves_metadata():
    params = {"max_tokens": 8, "temperature": 0.0}
    sequences = [make_seq(1)]
    group = SequenceGroup(
        request_id="req-42",
        sequences=sequences,
        sampling_params=params,
        arrival_time=123.5,
    )

    assert group.request_id == "req-42"
    assert group.sequences == sequences
    assert group.sampling_params is params
    assert group.arrival_time == 123.5


def test_sequence_group_equality_tracks_sequences_and_metadata():
    sequence = make_seq(1)
    first = SequenceGroup(
        request_id="req-1",
        sequences=[sequence],
        sampling_params={"max_tokens": 4},
        arrival_time=1.0,
    )
    same = SequenceGroup(
        request_id="req-1",
        sequences=[sequence],
        sampling_params={"max_tokens": 4},
        arrival_time=1.0,
    )
    different = SequenceGroup(
        request_id="req-2",
        sequences=[sequence],
        sampling_params={"max_tokens": 4},
        arrival_time=1.0,
    )

    assert first == same
    assert first != different


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
