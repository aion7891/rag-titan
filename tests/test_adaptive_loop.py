"""
Adaptive RAG's retries run under the shared LoopLimit: the result says how many
searches ran and why they stopped, and every run is the same as before.
"""

import json
from pathlib import Path

import pytest
from conftest import FakeLLM

from rag_titan import AdaptiveRAG
from rag_titan.generation import GroundedGenerator
from rag_titan.playground import check_presets
from rag_titan.prompts import NO_ANSWER
from rag_titan.rag_types.loops import EVIDENCE_GOOD, MAX_STEPS

PRESETS = Path(__file__).parent.parent / "docs" / "assets" / "playground" / "presets.json"


def adaptive(parts, replies, **options):
    llm = FakeLLM(replies)
    rag = AdaptiveRAG(llm, parts["retriever"], parts["reranker"], GroundedGenerator(llm),
                      **options)
    return rag.run("how many leave days?"), llm


def names(result):
    return [step["step"] for step in result["steps"]]


def test_enough_evidence_stops_the_loop_after_one_search(parts):
    result, _ = adaptive(parts, ["simple", "yes", "20 days [chunk_0]"])

    assert result["answer"] == "20 days [chunk_0]"
    assert result["loop"] == {"steps": 1, "max_steps": 2, "stop_reason": EVIDENCE_GOOD}
    assert names(result) == ["route", "retrieve", "rerank", "check", "generate"]


def test_enough_evidence_on_the_second_search(parts):
    result, _ = adaptive(parts, ["simple", "no", "leave days", "yes", "20 days [chunk_0]"])

    assert result["loop"] == {"steps": 2, "max_steps": 2, "stop_reason": EVIDENCE_GOOD}
    assert names(result).count("rewrite") == 1


def test_giving_up_keeps_the_old_steps_and_calls(parts):
    result, llm = adaptive(parts, ["simple", "no", "leave days", "no"])

    assert result["answer"] == NO_ANSWER
    assert names(result) == ["route", "retrieve", "rerank", "check", "rewrite",
                             "retrieve", "rerank", "check", "give_up"]
    assert len(llm.calls) == 4  # route, check, rewrite, check: no rewrite after the last
    loop = {"steps": 2, "max_steps": 2, "stop_reason": MAX_STEPS}
    assert result["loop"] == loop
    assert result["steps"][-1]["input"] == {"attempts": 2, "loop": loop}


def test_one_attempt_never_rewrites(parts):
    result, _ = adaptive(parts, ["simple", "no"], max_attempts=1)

    assert "rewrite" not in names(result)
    assert result["loop"] == {"steps": 1, "max_steps": 1, "stop_reason": MAX_STEPS}


def test_a_direct_reply_has_no_loop(parts):
    result, _ = adaptive(parts, ["direct", "Hello there!"])

    assert result["answer"] == "Hello there!" and "loop" not in result


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_bad_max_attempts_still_raise(parts, bad):
    with pytest.raises(ValueError, match="max_attempts"):
        adaptive(parts, [], max_attempts=bad)


def test_every_recorded_adaptive_run_replays_the_same():
    if not PRESETS.exists():
        pytest.skip("no recorded presets yet")
    presets = json.loads(PRESETS.read_text(encoding="utf-8"))
    presets["rag_types"] = ["adaptive"]
    presets["expected"] = {chunking: {retriever: {"adaptive": types["adaptive"]}
                                      for retriever, types in by_retriever.items()}
                           for chunking, by_retriever in presets["expected"].items()}

    # answers, steps and sources of all 540 runs recorded with 0.14.0
    assert check_presets(presets) == []
