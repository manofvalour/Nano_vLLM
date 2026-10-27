import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inference_engine.core.sequence import Sequence, SequenceStatus


def make_seq(
    *,
    prompt_token_ids: list[int] | None = None,
    sampling_params: object = None,
    **kwargs,
) -> Sequence:
    return Sequence(
        seq_id=kwargs.pop("seq_id", 1),
        prompt_token_ids=prompt_token_ids if prompt_token_ids is not None else [1, 2],
        sampling_params=sampling_params,
        **kwargs,
    )


def test_sequence_empty_prompt_and_prefill_lifecycle():
    seq = make_seq(prompt_token_ids=[], sampling_params={"max_tokens": 4})

    assert seq.prompt_len == 0
    assert seq.output_len == 0
    assert seq.total_len == 0
    assert seq.is_prefill_done is True
    assert seq.is_finished is False
    assert seq.num_new_tokens == 0
    assert seq.check_finished() is False


def test_sequence_normalizes_token_and_block_inputs():
    seq = Sequence(
        seq_id=1,
        prompt_token_ids=(1, "2", 3.0),
        block_table=(4, 5),
        output_token_ids=(7,),
        output_logprobs=(0.5, "1.5"),
    )

    assert seq.prompt_token_ids == [1, 2, 3]
    assert seq.block_table == [4, 5]
    assert seq.output_token_ids == [7]
    assert seq.output_logprobs == [0.5, 1.5]

    empty = Sequence(
        seq_id=2,
        prompt_token_ids=None,
        block_table=None,
        output_token_ids=None,
        output_logprobs=None,
    )
    assert empty.prompt_token_ids == []
    assert empty.block_table == []
    assert empty.output_token_ids == []
    assert empty.output_logprobs == []


def test_sequence_clamps_cached_and_computed_tokens():
    seq = Sequence(seq_id=1, num_cached_tokens=-4, num_computed_tokens=-1)

    assert seq.num_cached_tokens == 0
    assert seq.num_computed_tokens == 0


def test_sequence_properties_track_prompt_and_output():
    seq = make_seq(prompt_token_ids=[1, 2, 3])
    seq.append_token(10)
    seq.append_token(11)
    seq.num_computed_tokens = 2

    assert seq.prompt_len == 3
    assert seq.output_len == 2
    assert seq.total_len == 5
    assert len(seq) == 5
    assert seq.is_prefill_done is False

    seq.num_computed_tokens = 3
    assert seq.is_prefill_done is True


@pytest.mark.parametrize(
    "status",
    [
        SequenceStatus.FINISHED_STOPPED,
        SequenceStatus.FINISHED_LENGTH_CAPPED,
        SequenceStatus.FINISHED_ABORTED,
        SequenceStatus.FINISHED_IGNORED,
    ],
)
def test_sequence_finished_states(status):
    seq = make_seq(status=status)
    assert seq.is_finished is True


@pytest.mark.parametrize(
    "status",
    [SequenceStatus.WAITING, SequenceStatus.RUNNING, SequenceStatus.SWAPPED],
)
def test_sequence_unfinished_states(status):
    seq = make_seq(status=status)
    assert seq.is_finished is False


def test_sequence_append_token_tracks_logprobs():
    seq = make_seq()

    seq.append_token(5, logprob=-0.25)
    seq.append_token(6, logprob=-0.75)

    assert seq.output_token_ids == [5, 6]
    assert seq.output_logprobs == [-0.25, -0.75]
    assert seq.cumulative_logprob == pytest.approx(-1.0)
    assert seq.num_new_tokens == 2


def test_sequence_append_token_without_logprob_only_appends_token():
    seq = make_seq()
    seq.append_token(5, logprob=-0.5)
    seq.append_token(6)

    assert seq.output_token_ids == [5, 6]
    assert seq.output_logprobs == [-0.5]
    assert seq.cumulative_logprob == pytest.approx(-0.5)


def test_sequence_maybe_append_block_deduplicates_last_block():
    seq = make_seq(block_table=[10, 11])

    assert seq.maybe_append_block(None) is False
    assert seq.maybe_append_block(11) is False
    assert seq.maybe_append_block(12) is True
    assert seq.maybe_append_block(12) is False
    assert seq.maybe_append_block(10) is True
    assert seq.block_table == [10, 11, 12, 10]


def test_sequence_stop_tokens_and_length_caps():
    stop_seq = make_seq(sampling_params={"stop_token_ids": [7, 9], "max_tokens": 5})
    stop_seq.append_token(5)
    assert stop_seq.check_finished() is False

    stop_seq.append_token(9)
    assert stop_seq.check_finished() is True
    assert stop_seq.status == SequenceStatus.FINISHED_STOPPED
    assert stop_seq.finish_reason == "stop_token"

    length_seq = make_seq(sampling_params={"max_tokens": 2})
    length_seq.append_token(20)
    assert length_seq.check_finished() is False

    length_seq.append_token(21)
    assert length_seq.check_finished() is True
    assert length_seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED
    assert length_seq.finish_reason == "max_length"


def test_sequence_stop_token_via_stop_alias():
    seq = make_seq(sampling_params={"stop": [3, 4]})

    seq.append_token(3)
    assert seq.check_finished() is True
    assert seq.status == SequenceStatus.FINISHED_STOPPED
    assert seq.finish_reason == "stop_token"


@pytest.mark.parametrize("stop_token_ids", [{4, 5}, 4, (4, 5)])
def test_sequence_stop_token_ids_accept_multiple_container_types(stop_token_ids):
    seq = make_seq(sampling_params={"stop_token_ids": stop_token_ids})
    seq.append_token(4)

    assert seq.check_finished() is True
    assert seq.status == SequenceStatus.FINISHED_STOPPED


def test_sequence_stop_token_takes_precedence_over_length_cap():
    seq = make_seq(sampling_params={"stop_token_ids": [9], "max_tokens": 1})
    seq.append_token(9)

    assert seq.check_finished() is True
    assert seq.status == SequenceStatus.FINISHED_STOPPED
    assert seq.finish_reason == "stop_token"


def test_sequence_zero_max_tokens_finishes_immediately():
    seq = make_seq(sampling_params={"max_tokens": 0})

    assert seq.check_finished() is True
    assert seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED
    assert seq.finish_reason == "max_length"


def test_sequence_check_finished_is_idempotent():
    seq = make_seq(status=SequenceStatus.FINISHED_ABORTED)
    seq.finish_reason = "aborted"

    assert seq.check_finished() is True
    assert seq.check_finished() is True
    assert seq.status == SequenceStatus.FINISHED_ABORTED
    assert seq.finish_reason == "aborted"


def test_sequence_supports_object_sampling_params():
    params = SimpleNamespace(stop_token_ids=[8], max_tokens=3)
    seq = make_seq(sampling_params=params)

    seq.append_token(8)
    assert seq.check_finished() is True
    assert seq.status == SequenceStatus.FINISHED_STOPPED


def test_sequence_without_sampling_params_never_finishes():
    seq = make_seq(sampling_params=None)
    for token_id in range(5):
        seq.append_token(token_id)
        assert seq.check_finished() is False
    assert seq.finish_reason is None


def test_sequence_reset_for_preemption():
    seq = make_seq(
        prompt_token_ids=[7, 8],
        sampling_params={"max_tokens": 4},
        status=SequenceStatus.RUNNING,
        block_table=[1, 2],
        num_cached_tokens=5,
        num_computed_tokens=2,
    )
    seq.output_token_ids = [11]
    seq.finish_reason = "max_length"

    seq.reset_for_preemption()

    assert seq.status == SequenceStatus.WAITING
    assert seq.finish_reason is None
    assert seq.num_cached_tokens == 5
    assert seq.num_computed_tokens == 2
    assert seq.block_table == [1, 2]
    assert seq.output_token_ids == [11]
    assert seq.is_finished is False


def test_sequence_repr_includes_lifecycle_fields():
    seq = make_seq(seq_id=42, prompt_token_ids=[1, 2], block_table=[3])
    seq.append_token(9)

    representation = repr(seq)

    assert "Sequence(" in representation
    assert "seq_id=42" in representation
    assert "status=WAITING" in representation
    assert "prompt_len=2" in representation
    assert "output_len=1" in representation
    assert "block_table=[3]" in representation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
