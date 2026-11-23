"""The playground's RRF knobs: question weight and keep-top (HyDE, RAG-Fusion), hybrid weights."""

import json
from pathlib import Path

import pytest

from rag_titan.playground import Playground
from rag_titan.playground.recording import PRESET_GRID, grid_tunings
from rag_titan.playground.tuning import RETRIEVER_DEFAULTS, TUNING_KEYS, check_tuning

PRESETS = Path(__file__).parent.parent / "docs" / "assets" / "playground" / "presets.json"


@pytest.fixture(scope="module")
def playground():
    if not PRESETS.exists():
        pytest.skip("no recorded presets yet")
    return Playground(json.loads(PRESETS.read_text(encoding="utf-8")))


def step(run, name):
    return [s for s in run["steps"] if s["step"] == name][0]


def test_the_pages_get_the_knobs_and_their_defaults(playground):
    info = playground.info()

    assert info["retriever_defaults"] == {"hybrid_dense_weight": 1.0, "hybrid_bm25_weight": 1.0,
                                          "rrf_keep_top": False}
    assert info["defaults"]["hyde"]["question_weight"] == 1.0
    assert set(RETRIEVER_DEFAULTS) | {"question_weight"} <= set(TUNING_KEYS)
    assert check_tuning({"question_weight": 2.0, "rrf_keep_top": True}, [1500]) == {
        "question_weight": 2.0, "rrf_keep_top": True}


def test_a_hybrid_weight_reaches_the_fused_scores(playground):
    run = playground.run("paragraph", "simple", "q_001", "hybrid", {"hybrid_dense_weight": 2.0})

    chunks = step(run, "fuse")["output"]
    assert chunks and all(chunk["score_trace"].get("dense_weight") == 2.0 for chunk in chunks
                          if "dense_rank" in chunk["score_trace"])


def test_the_hybrid_knobs_change_nothing_with_another_retriever(playground):
    plain = playground.run("paragraph", "simple", "q_001", "dense")
    tuned = playground.run("paragraph", "simple", "q_001", "dense", {"hybrid_dense_weight": 3.0,
                                                                     "rrf_keep_top": True})

    assert tuned["steps"][0]["output"] == plain["steps"][0]["output"]


def test_hyde_s_question_weight_reaches_the_fuse_step(playground):
    run = playground.run("word", "hyde", "q_005", "dense", {"question_weight": 2.0})

    assert step(run, "fuse_queries")["input"]["weights"] == {"query_1": 1.0, "query_2": 2.0}


def test_keep_top_brings_back_a_search_s_best_chunk(playground):
    # RAG-Fusion with top_k 2: the furniture-money chunk is one search's #1 only
    tuning = {"top_k": 2, "top_n": 2}
    plain = playground.run("word", "rag_fusion", "q_006", "dense", tuning)
    kept = playground.run("word", "rag_fusion", "q_006", "dense", {**tuning, "rrf_keep_top": True})

    fused = step(kept, "fuse_queries")["output"]
    assert any("kept_as_top_of" in chunk["score_trace"] for chunk in fused)
    assert kept["recall"] > plain["recall"]


def test_a_bad_weight_gives_a_clear_error(playground):
    with pytest.raises(ValueError, match="question_weight must be a number above 0"):
        playground.run("word", "hyde", "q_005", "dense", {"question_weight": 0})


@pytest.mark.parametrize("rag_type", ["simple", "hyde", "rag_fusion"])
def test_every_new_knob_value_has_recorded_answers(playground, rag_type):
    new = [(label, tuning) for label, tuning in grid_tunings(rag_type, PRESET_GRID, "keyword")
           if tuning and set(tuning) & (set(RETRIEVER_DEFAULTS) | {"question_weight"})]
    assert new

    for _, tuning in new:
        for retriever in ("dense", "hybrid"):
            for question in ("q_005", "q_006"):
                run = playground.run("word", rag_type, question, retriever, tuning)
                assert run["recorded"], (rag_type, tuning, retriever, question)
