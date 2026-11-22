"""Tests for the loop limits shared by the RAG types that search more than once."""

import pytest

from rag_titan.rag_types.loops import (
    MAX_STEPS,
    MODEL_DONE,
    NO_NEW_CHUNKS,
    LoopLimit,
)


def chunks(*ids):
    return [{"id": chunk_id, "text": chunk_id} for chunk_id in ids]


def test_the_cap_stops_the_loop_and_says_why():
    limit = LoopLimit(max_steps=2)

    rounds = 0
    while limit.next_step():
        rounds += 1

    assert rounds == 2
    assert limit.summary() == {"steps": 2, "max_steps": 2, "stop_reason": MAX_STEPS}
    assert limit.next_step() is False  # a stopped loop stays stopped


def test_zero_steps_never_runs():
    limit = LoopLimit(max_steps=0)

    assert limit.next_step() is False
    assert limit.summary() == {"steps": 0, "max_steps": 0, "stop_reason": MAX_STEPS}


def test_new_chunks_returns_only_unseen_ones():
    limit = LoopLimit(max_steps=3)
    limit.remember(chunks("a", "b"))

    fresh = limit.new_chunks(chunks("b", "c", "c", "d"))

    assert [chunk["id"] for chunk in fresh] == ["c", "d"]
    assert limit.stop_reason is None


def test_no_new_chunks_stops_the_loop():
    limit = LoopLimit(max_steps=5)
    limit.next_step()
    limit.new_chunks(chunks("a"))

    limit.next_step()
    fresh = limit.new_chunks(chunks("a"))

    assert fresh == []
    assert limit.stop_reason == NO_NEW_CHUNKS
    assert limit.next_step() is False
    assert limit.summary()["steps"] == 2


def test_an_empty_search_is_no_progress_too():
    limit = LoopLimit(max_steps=5)

    assert limit.new_chunks([]) == []
    assert limit.stop_reason == NO_NEW_CHUNKS


def test_the_model_can_stop_the_loop_and_the_first_reason_wins():
    limit = LoopLimit(max_steps=5)
    limit.next_step()

    limit.stop(MODEL_DONE)
    limit.new_chunks([])  # would be "no new chunks", but the loop already stopped

    assert limit.stop_reason == MODEL_DONE
    assert limit.next_step() is False


def test_an_unknown_stop_reason_is_refused():
    with pytest.raises(ValueError, match="Unknown stop reason"):
        LoopLimit(max_steps=1).stop("tired")


@pytest.mark.parametrize("bad", [-1, 1.5, "3", None, True])
def test_bad_max_steps_raise_a_clear_error(bad):
    with pytest.raises(ValueError, match="max_retries must be a whole number of 0 or more"):
        LoopLimit(bad, name="max_retries")
