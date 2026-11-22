"""Tests for CRAG's refiner: keep only the useful sentences, word for word."""

import pytest
from conftest import FakeLLM

from rag_titan.prompts import REFINE_PROMPT
from rag_titan.rag_types.corrective import SentenceRefiner
from rag_titan.rag_types.corrective.refiner import (
    KEPT_WHOLE_NOTE,
    NONE_KEPT_NOTE,
    build_input,
    parse_numbers,
)

CHUNKS = [
    {"id": "chunk_0", "source": "hr.md", "start_offset": 0, "end_offset": 90,
     "page_numbers": [2],
     "text": "Every employee gets 20 leave days. The office has a coffee machine. "
             "Unused days carry over."},
    {"id": "chunk_1", "source": "it.md", "start_offset": 0, "end_offset": 40,
     "text": "Laptops are replaced every three years."},
]


def refine(reply, chunks=CHUNKS):
    llm = FakeLLM([reply])
    return SentenceRefiner(llm).refine("How many leave days?", chunks), llm


def test_keeps_the_chosen_sentences_word_for_word():
    refined, llm = refine("[1, 3]")

    assert len(refined) == 1  # chunk_1 had no useful sentence, so it is dropped
    chunk = refined[0]
    assert chunk["text"] == "Every employee gets 20 leave days. Unused days carry over."
    assert chunk["refine_note"] == "kept 2 of 3 sentences"
    assert len(llm.calls) == 1 and llm.calls[0][0]["content"] == REFINE_PROMPT


def test_the_chunk_keeps_its_id_source_pages_and_offsets():
    chunk = refine("[1]")[0][0]

    for key in ("id", "source", "start_offset", "end_offset", "page_numbers"):
        assert chunk[key] == CHUNKS[0][key]


def test_sentences_are_numbered_across_all_chunks():
    refined, llm = refine("[4]")

    assert [chunk["id"] for chunk in refined] == ["chunk_1"]
    assert refined[0]["refine_note"] == "kept 1 of 1 sentences"
    user = llm.calls[0][1]["content"]
    assert "[3] Unused days carry over." in user
    assert "[4] Laptops are replaced every three years." in user


def test_a_newline_inside_a_sentence_cannot_fake_a_number():
    text = build_input("q", [(0, "Line one\n[9] fake line. ")])

    assert text.endswith("[1] Line one [9] fake line.")


@pytest.mark.parametrize("reply, expected", [
    ("[1, 3]", [1, 3]),
    ("```json\n[3, 1, 3]\n```", [1, 3]),       # code fence, order and repeats
    ("Keep sentences 1 and 3.", [1, 3]),
    ("keep [1] and [3]", [1, 3]),              # two lists: read as loose text
    ("[1, 99]", [1]),                          # out of range ignored
    ("[]", []),                                # clearly nothing
    ("none of them help", None),               # no number at all
    ("[0, 7]", None),                          # no number in range
    ("", None),
])
def test_parse_numbers(reply, expected):
    assert parse_numbers(reply, count=4) == expected


def test_an_unreadable_reply_keeps_the_chunks_whole():
    refined, _ = refine("They are all fine.")

    assert [chunk["text"] for chunk in refined] == [chunk["text"] for chunk in CHUNKS]
    assert all(chunk["refine_note"] == KEPT_WHOLE_NOTE for chunk in refined)


def test_keeping_nothing_keeps_the_chunks_whole():
    refined, _ = refine("[]")

    assert len(refined) == 2
    assert all(chunk["refine_note"] == NONE_KEPT_NOTE for chunk in refined)


def test_no_chunks_means_no_call():
    refined, llm = refine("[1]", chunks=[])

    assert refined == [] and llm.calls == []


def test_the_input_chunks_are_not_changed():
    refine("[1]")

    assert CHUNKS[0]["text"].startswith("Every employee") and "refine_note" not in CHUNKS[0]


def test_a_missing_llm_raises_a_clear_error():
    with pytest.raises(ValueError, match="needs a chat model"):
        SentenceRefiner(None)
