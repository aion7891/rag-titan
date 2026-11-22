"""The playground replays Corrective and Iterative RAG from the recording, with their loops."""

import json
from pathlib import Path

import pytest

from rag_titan.playground import Playground
from rag_titan.rag_types.loops import STOP_REASONS

PRESETS = Path(__file__).parent.parent / "docs" / "assets" / "playground" / "presets.json"


@pytest.fixture(scope="module")
def playground():
    if not PRESETS.exists():
        pytest.skip("no recorded presets yet")
    return Playground(json.loads(PRESETS.read_text(encoding="utf-8")))


@pytest.mark.parametrize("rag_type, decision", [("corrective", "grade"), ("iterative", "think")])
def test_both_types_replay_with_their_loop(playground, rag_type, decision):
    run = playground.run("paragraph", rag_type, "q_001")

    assert run["recorded"] and run["answer"]
    assert run["loop"]["stop_reason"] in STOP_REASONS
    assert decision in [step["step"] for step in run["steps"]]
    assert json.loads(json.dumps(run)) == run          # plain JSON for the browser


def test_simple_rag_has_no_loop(playground):
    assert playground.run("paragraph", "simple", "q_001")["loop"] is None


def test_the_knobs_change_the_run(playground):
    no_refine = playground.run("paragraph", "corrective", "q_001",
                               tuning={"crag_refine": False})
    search_once = playground.run("paragraph", "iterative", "q_001",
                                 tuning={"iterative_max_steps": 0})

    assert "refine" not in [step["step"] for step in no_refine["steps"]]
    assert [step["step"] for step in search_once["steps"]] == ["retrieve", "rerank", "generate"]
