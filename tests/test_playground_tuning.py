"""
Tests for tuned playground runs: any top_k / top_n / reranker / attempts / budget,
stopping cleanly where the recording ends, and Parent-Child indexes in the browser.
"""

import pytest
from conftest import FakeEmbedder
from test_playground import DOCUMENTS, QUESTIONS, MoodyLLM

from rag_titan.factory import RAG_TYPES
from rag_titan.llm import ReplayLLM
from rag_titan.playground import (
    PRESET_CHUNKING,
    LivePlayground,
    Playground,
    check_presets,
    grid_tunings,
    record_presets,
)
from rag_titan.playground.tuning import check_tuning, default_tuning
from rag_titan.rag_types import Trace
from rag_titan.utils.not_recorded import NotRecorded


@pytest.fixture(scope="module")
def presets():
    """A small recording of every RAG type with its defaults."""
    return record_presets(DOCUMENTS, QUESTIONS, PRESET_CHUNKING, MoodyLLM(), FakeEmbedder())


def without_replies(presets):
    """The same recording with no model replies: every answer is "not recorded"."""
    return {**presets, "replies": {}}


def test_trace_keeps_the_step_that_failed():
    trace = Trace()

    def broken(query):
        raise NotRecorded("no reply")

    with pytest.raises(NotRecorded):
        trace.record("generate", broken, query="q")
    assert trace.steps[-1]["step"] == "generate"
    assert trace.steps[-1]["output"] == {"error": "NotRecorded: no reply"}


def test_a_missing_reply_is_not_recorded_and_still_a_value_error():
    with pytest.raises(NotRecorded, match="No recorded reply"):
        ReplayLLM({}).ask("system", "user")
    assert issubclass(NotRecorded, ValueError)


def test_recorded_settings_replay_and_others_stop_at_the_answer(presets):
    playground = Playground(presets)

    same = playground.run("word", "simple", "q1")
    assert same["recorded"] and same["stopped_at"] is None and same["answer"]

    # more chunks than the documents have: the model gets the same context, so it replays
    assert playground.run("word", "simple", "q1", tuning={"top_k": 6, "top_n": 4})["recorded"]

    tuned = Playground(without_replies(presets)).run("word", "simple", "q1",
                                                     tuning={"top_k": 1, "top_n": 1})
    assert not tuned["recorded"] and tuned["stopped_at"] == "generate"
    assert tuned["answer"] is None and tuned["answered"] is None
    # the context the model would read is still scored
    assert tuned["recall"] in (0.0, 1.0) and len(tuned["sources"]) == 1
    assert "error" in tuned["steps"][-1]["output"]


def test_an_unrecorded_llm_rerank_stops_at_the_rerank(presets):
    run = Playground(without_replies(presets)).run("word", "simple", "q1",
                                                   tuning={"reranker": "llm"})
    assert run["stopped_at"] == "rerank" and run["recall"] is None


def test_parent_child_runs_on_its_own_index(presets):
    run = Playground(without_replies(presets)).run("word", "parent_child", "q1",
                                                   tuning={"parent_budget": 200})

    assert [s["step"] for s in run["steps"]] == ["retrieve", "rerank", "expand", "generate"]
    parents = run["steps"][2]["output"]
    assert parents and all("children" in parent for parent in parents)
    assert run["stopped_at"] == "generate" and run["recall"] is not None


def test_evaluate_tuned_sums_up_every_question(presets):
    summary = Playground(without_replies(presets)).evaluate_tuned("word", "simple", "dense",
                                                                  {"top_n": 1})

    assert summary["questions"] == 2 and summary["recorded"] == 0
    assert summary["recall"] is not None and summary["answered"] is None
    assert all("steps" not in row and row["id"] for row in summary["rows"])
    assert Playground(presets).evaluate_tuned("word", "simple")["recorded"] == 2


def test_info_draws_every_rag_type_and_offers_the_knobs(presets):
    info = Playground(presets).info()

    assert set(info["flows"]) == set(RAG_TYPES)
    expand = [n["id"] for n in info["flows"]["parent_child"]["dense"]["nodes"]]
    assert expand == ["retrieve", "rerank", "expand", "generate"]
    assert info["rerankers"] == ["keyword", "none", "mmr", "llm"]
    assert info["defaults"] == default_tuning() and info["parent_sizes"] == [1500]
    assert info["rag_types"] == RAG_TYPES


@pytest.mark.parametrize("tuning, message", [
    ({"top_kk": 3}, "Unknown tuning"),
    ({"reranker": "cross_encoder"}, "cannot run in the browser"),
    ({"reranker": "cohere"}, "Unknown reranker"),
    ({"parent_size": 900}, "was not recorded"),
    ("top_k=3", "must be a dict"),
])
def test_bad_tuning_gives_a_clear_error(tuning, message):
    with pytest.raises(ValueError, match=message):
        check_tuning(tuning, [1500])


def test_numbers_are_checked_by_the_rag_type(presets):
    with pytest.raises(ValueError, match="top_n .3. cannot be more than top_k .2."):
        Playground(presets).run("word", "simple", "q1", tuning={"top_k": 2, "top_n": 3})


def test_default_tuning_is_each_type_s_defaults():
    defaults = default_tuning()
    assert defaults["simple"] == {"top_k": 5, "top_n": 2}
    assert defaults["adaptive"]["adaptive_max_attempts"] == 2
    assert defaults["parent_child"] == {"top_k": 8, "top_n": 4, "parent_budget": 3000,
                                        "parent_size": 1500}


def test_live_mode_builds_parents_for_parent_child(presets):
    live = LivePlayground(presets, chat=lambda messages: "an answer",
                          embed=lambda texts: FakeEmbedder().embed_many(texts),
                          model="fake", embed_model="fake-embed")
    run = live.run("word", "parent_child", "how many leave days")
    assert [s["step"] for s in run["steps"]] == ["retrieve", "rerank", "expand", "generate"]


GRID = {"simple": {"top_k": [1, 5], "reranker": ["keyword", "none"]},
        "parent_child": {"parent_size": [1500, 60], "parent_budget": [100, 3000]}}


def test_grid_tunings_are_one_knob_at_a_time():
    labels = [label for label, _ in grid_tunings("simple", GRID, "keyword")]
    assert labels == ["default", "top_k=1", "reranker=none"]       # defaults are not repeated
    assert grid_tunings("adaptive", GRID, "keyword") == [("default", None)]


def test_a_recorded_grid_replays_every_knob_value():
    presets = record_presets(DOCUMENTS, QUESTIONS, PRESET_CHUNKING, MoodyLLM(), FakeEmbedder(),
                             grid=GRID)
    playground = Playground(presets)

    assert presets["parent_sizes"] == [1500, 60] and presets["grid"] == GRID
    assert set(presets["expected"]["word"]["dense"]["parent_child"]) == {
        "default", "parent_size=60", "parent_budget=100"}
    for tuning in ({"top_k": 1}, {"reranker": "none"}):
        assert playground.run("word", "simple", "q1", "hybrid", tuning)["recorded"]
    small = playground.run("word", "parent_child", "q1", tuning={"parent_size": 60})
    assert small["recorded"] and small["steps"][2]["step"] == "expand"
    assert check_presets(presets) == []


QUERY_GRID = {"hyde": {"hyde_with_question": [True, False]},
              "rag_fusion": {"fusion_queries": [1, 3]}}


def test_the_query_transformation_knobs_record_and_replay():
    presets = record_presets(DOCUMENTS, QUESTIONS, PRESET_CHUNKING, MoodyLLM(), FakeEmbedder(),
                             grid=QUERY_GRID)
    playground = Playground(presets)

    alone = playground.run("word", "hyde", "q1", tuning={"hyde_with_question": False})
    assert alone["recorded"] and "fuse_queries" not in [s["step"] for s in alone["steps"]]
    one = playground.run("word", "rag_fusion", "q1", tuning={"fusion_queries": 1})
    assert one["recorded"] and one["steps"][0]["step"] == "write_queries"
    assert check_presets(presets) == []


def test_the_query_transformation_defaults_for_the_pages():
    defaults = default_tuning()
    assert defaults["hyde"] == {"top_k": 5, "top_n": 2, "hyde_with_question": True,
                                "question_weight": 1.0, "rrf_keep_top": False}
    assert defaults["rag_fusion"] == {"top_k": 5, "top_n": 2, "fusion_queries": 3,
                                      "question_weight": 1.0, "rrf_keep_top": False}
    assert check_tuning({"hyde_with_question": False, "fusion_queries": 4}, [1500]) == {
        "hyde_with_question": False, "fusion_queries": 4}
