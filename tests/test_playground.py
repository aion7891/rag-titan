"""Tests for the playground presets: record, replay, check, and the committed recording."""

import json
from pathlib import Path

import pytest
from conftest import FakeEmbedder

from rag_titan.llm import BaseLLM, RecordingLLM
from rag_titan.playground import (
    PRESET_CHUNKING,
    PRESET_KEYS,
    build_parts,
    check_presets,
    index_documents,
    preset_settings,
    record_presets,
    replay_parts,
)

PRESETS = Path(__file__).parent.parent / "docs" / "assets" / "playground" / "presets.json"

DOCUMENTS = [
    {"source": "hr.md", "text": "You get 20 leave days every year. Salary is paid monthly."},
    {"source": "it.md", "text": "Remote work needs a laptop. The office opens at nine."},
]
QUESTIONS = [
    {"id": "q1", "question": "how many leave days", "required_evidence_ids": ["hr.md"]},
    {"id": "q2", "question": "remote work laptop", "required_evidence_ids": ["it.md"]},
]


class MoodyLLM(BaseLLM):
    """A model that never answers the same twice, like a real one with sampling."""

    def __init__(self):
        super().__init__("moody")
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        replies = ["simple", "complex", "yes", "no", "leave days per year"]
        reply = replies[self.calls % len(replies)] + f" #{self.calls}"
        if self.calls % 3 == 0:
            reply = ["simple", "yes", "no"][self.calls % 2]
        return {"content": reply, "tool_calls": [], "message": {}}


def test_preset_settings_keep_only_what_a_preset_needs():
    settings = preset_settings(PRESET_CHUNKING["paragraph"])
    assert list(settings) == PRESET_KEYS
    assert settings["chunker"] == "paragraph" and settings["chunk_size"] == 300


def test_index_documents_needs_no_files():
    parts = build_parts(preset_settings({"chunk_size": 5, "chunk_overlap": 0}),
                        FakeEmbedder(), MoodyLLM())
    assert index_documents(parts, DOCUMENTS) == parts["store"].count() > 2


def test_recording_llm_asks_each_prompt_once():
    llm = MoodyLLM()
    recorder = RecordingLLM(llm)
    first = recorder.ask("route", "how many leave days")
    again = recorder.ask("route", "how many leave days")

    assert first == again and llm.calls == 1 and len(recorder.replies) == 1


def test_a_recording_replays_the_same_even_with_a_moody_model():
    presets = record_presets(DOCUMENTS, QUESTIONS, PRESET_CHUNKING, MoodyLLM(), FakeEmbedder())

    assert set(presets["expected"]) == set(PRESET_CHUNKING)
    assert set(presets["expected"]["word"]) == {"simple", "adaptive"}
    assert check_presets(json.loads(json.dumps(presets))) == []   # survives a JSON round trip


def test_check_finds_a_changed_run():
    presets = record_presets(DOCUMENTS, QUESTIONS, PRESET_CHUNKING, MoodyLLM(), FakeEmbedder())
    presets["expected"]["word"]["simple"]["q1"]["answer"] = "something else"

    problems = check_presets(presets)
    assert len(problems) == 1 and problems[0].startswith("word / simple / q1")


@pytest.mark.skipif(not PRESETS.exists(), reason="no recorded presets yet")
def test_the_committed_recording_still_replays():
    """Fails when prompts, RAG types or chunkers changed: run
    `python tools/playground/record_traces.py` to record again."""
    presets = json.loads(PRESETS.read_text(encoding="utf-8"))

    assert check_presets(presets) == []
    parts = replay_parts(presets, "paragraph")
    assert parts["retriever"].has_documents()
