"""Tests for the two RRF options: list weights and keeping every list's #1 (keep_top)."""

import pytest

from rag_titan.chunking import WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.pipelines import IndexingPipeline
from rag_titan.rag_types import Trace
from rag_titan.retrieval import DenseRetriever, fuse_ranked, rrf_fuse, search_and_fuse
from rag_titan.vector_stores import MemoryStore


def result(chunk_id, score=0.5, text=None):
    """A search result as a retriever returns it (no offsets, all from a.md)."""
    return {"id": chunk_id, "source": "a.md", "text": text or f"text of {chunk_id}",
            "score": score}


def ids(matches):
    return [match["id"] for match in matches]


# three chunks found by both lists, and x: only first in q2
LISTS = {"q1": [result("a"), result("b"), result("c"), result("d")],
         "q2": [result("x"), result("a"), result("b"), result("c")]}


def test_no_weights_and_weights_of_one_give_the_old_results():
    old = fuse_ranked(LISTS, top_k=3)

    assert fuse_ranked(LISTS, top_k=3, weights={"q1": 1, "q2": 1.0}) == old
    assert fuse_ranked(LISTS, top_k=3, weights={"q1": 1}) == old   # a name left out counts 1
    assert fuse_ranked(LISTS, top_k=3, keep_top=False) == old
    assert rrf_fuse(LISTS, weights={"q2": 1}) == rrf_fuse(LISTS)


def test_a_weight_changes_the_order_by_hand():
    # equal RRF without weights, so the higher relevance (b) wins the tie
    lists = {"q1": [result("a", 0.1), result("b", 0.9)], "q2": [result("b", 0.9), result("a", 0.1)]}
    assert [chunk["id"] for chunk, _ in rrf_fuse(lists)] == ["b", "a"]

    fused = rrf_fuse(lists, weights={"q1": 2})

    assert [chunk["id"] for chunk, _ in fused] == ["a", "b"]
    trace = fused[0][1]
    assert trace["rrf"] == pytest.approx(2 / 61 + 1 / 62)
    assert trace["q1_weight"] == 2.0 and "q2_weight" not in trace   # 1 is not shown


def test_scores_still_top_out_at_one():
    lists = {"q1": [result("a"), result("b")], "q2": [result("a"), result("b")]}

    fused = fuse_ranked(lists, top_k=2, weights={"q1": 3})

    assert fused[0]["score"] == pytest.approx(1.0)           # first in every list
    assert fused[1]["score"] == pytest.approx((4 / 62) / (4 / 61))


@pytest.mark.parametrize("weights, message", [
    ({"q1": 0}, "above 0"),
    ({"q1": -1}, "above 0"),
    ({"q1": True}, "above 0"),
    ({"q1": "2"}, "above 0"),
    ({"q1": float("nan")}, "above 0"),
    ({"q1": float("inf")}, "above 0"),
    ({"zz": 1}, "unknown lists"),
    ([1, 2], "must be a dict"),
])
def test_bad_weights_give_a_clear_error(weights, message):
    with pytest.raises(ValueError, match=message):
        fuse_ranked(LISTS, top_k=3, weights=weights)


def test_keep_top_brings_back_a_list_s_first_chunk():
    assert ids(fuse_ranked(LISTS, top_k=3)) == ["a", "b", "c"]    # x is out: one vote only

    kept = fuse_ranked(LISTS, top_k=3, keep_top=True)

    assert ids(kept) == ["a", "b", "x"]                           # x replaced c, still best first
    assert kept[2]["score_trace"]["kept_as_top_of"] == ["q2"]
    assert "kept_as_top_of" not in kept[0]["score_trace"]         # a made it anyway
    assert "kept_as_top_of" not in fuse_ranked(LISTS, top_k=3)[0]["score_trace"]


def test_with_more_lists_than_places_the_best_first_chunks_win():
    lists = {"q1": [result("a"), result("x")], "q2": [result("b"), result("x")],
             "q3": [result("c"), result("x")]}
    assert ids(fuse_ranked(lists, top_k=2)) == ["x", "a"]         # x: three second places

    # a, b and c tie; the tie goes to the chunk seen first, as without keep_top
    assert ids(fuse_ranked(lists, top_k=2, keep_top=True)) == ["a", "b"]


def test_a_first_chunk_dropped_as_a_near_copy_is_not_brought_back():
    same = "Every employee gets twenty paid leave days each year"
    lists = {"q1": [result("a", text=same), result("c")], "q2": [result("b", text=same),
                                                                result("c")]}

    kept = fuse_ranked(lists, top_k=2, keep_top=True)

    assert ids(kept) == ["c", "a"]                                # b repeats a: not forced back


def test_keep_top_must_be_true_or_false():
    with pytest.raises(ValueError, match="keep_top must be True or False"):
        fuse_ranked(LISTS, top_k=3, keep_top="yes")


@pytest.fixture
def retriever(tmp_path):
    (tmp_path / "hr.md").write_text("You get 20 paid leave days per year. Sick leave needs "
                                    "a note from the doctor.", encoding="utf-8")
    (tmp_path / "it.md").write_text("Remote work needs a laptop. Error E-4022 means the "
                                    "printer is offline.", encoding="utf-8")
    store = MemoryStore()
    IndexingPipeline(WordChunker(6, 0), HashEmbedder(64), store).run(
        [tmp_path / "hr.md", tmp_path / "it.md"])
    return DenseRetriever(HashEmbedder(64), store)


def fuse_step(trace):
    return [step for step in trace.steps if step["step"] == "fuse_queries"][0]


def test_search_and_fuse_passes_the_options_on(retriever):
    trace = Trace()
    search_and_fuse(retriever, ["leave", "  ", "laptop"], 3, trace, weights=[2, 5, 1],
                    keep_top=True)

    used = fuse_step(trace)["input"]
    # the empty query is dropped together with its weight
    assert used["weights"] == {"query_1": 2.0, "query_2": 1.0}
    assert used["keep_top"] is True


def test_a_default_run_records_the_same_step_as_before(retriever):
    trace = Trace()
    search_and_fuse(retriever, ["leave", "laptop"], 3, trace)

    assert "weights" not in fuse_step(trace)["input"]
    assert "keep_top" not in fuse_step(trace)["input"]


@pytest.mark.parametrize("weights, keep_top, message", [
    ([2], False, "one number per query"),
    ({"query_1": 2}, False, "one number per query"),
    ([2, 0], False, "above 0"),
    (None, 1, "keep_top must be True or False"),
])
def test_bad_options_fail_before_any_search(retriever, weights, keep_top, message):
    trace = Trace()
    with pytest.raises(ValueError, match=message):
        search_and_fuse(retriever, ["leave", "laptop"], 3, trace, weights=weights,
                        keep_top=keep_top)
    assert trace.steps == []


def test_one_query_ignores_the_options(retriever):
    trace = Trace()
    search_and_fuse(retriever, ["leave", ""], 2, trace, weights=[3, 1], keep_top=True)

    assert [step["step"] for step in trace.steps] == ["retrieve"]
