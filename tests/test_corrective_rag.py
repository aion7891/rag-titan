"""Tests for Corrective RAG: grade the chunks, repair weak evidence, then answer."""

import pytest
from correction_fakes import ListRetriever, ScriptLLM, assert_follows_flow, names

from rag_titan import CorrectiveRAG
from rag_titan.chunking import WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline
from rag_titan.prompts import (
    GROUNDED_ANSWER_PROMPT,
    LLM_RERANK_PROMPT,
    NO_ANSWER,
)
from rag_titan.rag_types import describe_steps
from rag_titan.rag_types.loops import EVIDENCE_GOOD, MAX_STEPS, NO_NEW_CHUNKS
from rag_titan.retrieval import DenseRetriever
from rag_titan.retrieval.rerankers import NoReranker
from rag_titan.vector_stores import MemoryStore

QUESTION = "How many leave days do I get?"


def chunk(chunk_id, text="Staff get 20 leave days. The office has a gym.", score=0.5):
    return {"id": chunk_id, "source": "hr.md", "text": text, "score": score}


def corrective(llm, retriever, **options):
    return CorrectiveRAG(llm, retriever, NoReranker(), GroundedGenerator(llm), **options)


def test_good_evidence_is_kept_refined_and_answered():
    llm = ScriptLLM(['{"1": 9, "2": 1}'])
    retriever = ListRetriever([chunk("a"), chunk("b")])
    rag = corrective(llm, retriever)

    result = rag.run(QUESTION)

    assert names(result) == ["retrieve", "grade", "rerank", "refine", "generate"]
    assert result["answer"] == "20 days"
    # b was graded 1 (< lower 3): never kept; a's off-topic sentence was removed
    assert [source["id"] for source in result["sources"]] == ["a"]
    assert result["sources"][0]["text"] == "Staff get 20 leave days."
    assert result["loop"] == {"steps": 0, "max_steps": 1, "stop_reason": EVIDENCE_GOOD}
    assert retriever.queries == [QUESTION]
    assert result["steps"][1]["output"]["verdict"] == "correct"
    assert_follows_flow(rag, result)


def test_incorrect_evidence_is_thrown_away_and_searched_again():
    llm = ScriptLLM(['{"1": 0, "2": 1}', '{"1": 9}'])
    retriever = ListRetriever([chunk("a"), chunk("b")], [chunk("c")])
    rag = corrective(llm, retriever)

    result = rag.run(QUESTION)

    assert names(result) == ["retrieve", "grade", "rewrite", "retrieve", "grade",
                             "rerank", "refine", "generate"]
    assert [source["id"] for source in result["sources"]] == ["c"]
    # the rewritten query searches; grading and answering keep the original question
    assert retriever.queries == [QUESTION, "annual leave allowance"]
    grade_inputs = [user for system, user in llm.calls if system == LLM_RERANK_PROMPT]
    assert all(user.startswith(f"QUESTION:\n{QUESTION}") for user in grade_inputs)
    assert result["loop"]["stop_reason"] == EVIDENCE_GOOD and result["loop"]["steps"] == 1
    assert_follows_flow(rag, result)


def test_ambiguous_evidence_keeps_the_good_part_and_adds_the_new_search():
    llm = ScriptLLM(['{"1": 5, "2": 1}', '{"1": 8}'])
    retriever = ListRetriever([chunk("a"), chunk("b")], [chunk("c")])
    rag = corrective(llm, retriever, refine=False)

    result = rag.run(QUESTION)

    assert names(result) == ["retrieve", "grade", "rewrite", "retrieve", "grade",
                             "rerank", "generate"]
    # kept: a (5) and c (8), best grade first; b (1) is dropped
    assert [source["id"] for source in result["sources"]] == ["c", "a"]
    assert [source["grade"] for source in result["sources"]] == [8.0, 5.0]
    assert_follows_flow(rag, result)


def test_gives_up_honestly_when_the_retries_run_out():
    llm = ScriptLLM(['{"1": 0}', '{"1": 1}'])
    retriever = ListRetriever([chunk("a")], [chunk("b")])
    rag = corrective(llm, retriever)

    result = rag.run(QUESTION)

    assert names(result) == ["retrieve", "grade", "rewrite", "retrieve", "grade", "give_up"]
    assert result["answer"] == NO_ANSWER and result["sources"] == []
    assert result["loop"] == {"steps": 1, "max_steps": 1, "stop_reason": MAX_STEPS}
    assert result["steps"][-1]["input"] == {"loop": result["loop"]}
    assert GROUNDED_ANSWER_PROMPT not in llm.prompts()  # no answer from bad evidence
    assert_follows_flow(rag, result)


def test_a_search_with_nothing_new_stops_the_loop():
    llm = ScriptLLM(['{"1": 5}'])
    retriever = ListRetriever([chunk("a")])  # the second search finds "a" again
    rag = corrective(llm, retriever, max_retries=3, refine=False)

    result = rag.run(QUESTION)

    assert names(result) == ["retrieve", "grade", "rewrite", "retrieve", "grade",
                             "rerank", "generate"]
    # the second grade had nothing to grade: no model call for it
    assert llm.prompts().count(LLM_RERANK_PROMPT) == 1
    assert result["steps"][4]["output"]["note"] == "there were no chunks to grade"
    assert result["loop"] == {"steps": 1, "max_steps": 3, "stop_reason": NO_NEW_CHUNKS}
    assert [source["id"] for source in result["sources"]] == ["a"]


def test_zero_retries_grades_only():
    llm = ScriptLLM(['{"1": 0}'])
    rag = corrective(llm, ListRetriever([chunk("a")]), max_retries=0)

    result = rag.run(QUESTION)

    assert names(result) == ["retrieve", "grade", "give_up"]
    assert result["loop"] == {"steps": 0, "max_steps": 0, "stop_reason": MAX_STEPS}


def test_the_fallback_retriever_is_used_to_search_again():
    llm = ScriptLLM(['{"1": 0}', '{"1": 9}'])
    own, web = ListRetriever([chunk("a")]), ListRetriever([chunk("w")])
    rag = corrective(llm, own, fallback_retriever=web)

    result = rag.run(QUESTION)

    assert own.queries == [QUESTION] and web.queries == ["annual leave allowance"]
    assert [source["id"] for source in result["sources"]] == ["w"]


def test_unreadable_grades_keep_everything_and_search_again():
    llm = ScriptLLM(["they look fine", '{"1": 9}'])
    retriever = ListRetriever([chunk("a"), chunk("b")], [chunk("c")])
    rag = corrective(llm, retriever, refine=False, top_n=3)

    result = rag.run(QUESTION)

    assert result["steps"][1]["output"]["verdict"] == "ambiguous"
    assert [source["id"] for source in result["sources"]] == ["c", "a", "b"]


def test_top_n_limits_what_the_model_reads():
    llm = ScriptLLM(['{"1": 9, "2": 8, "3": 7}'])
    rag = corrective(llm, ListRetriever([chunk("a"), chunk("b"), chunk("c")]),
                     refine=False, top_n=2)

    assert [source["id"] for source in rag.run(QUESTION)["sources"]] == ["a", "b"]


def test_empty_question_and_empty_index():
    llm = ScriptLLM([])
    assert corrective(llm, ListRetriever([])).run("  ")["answer"] == "Please enter a question."

    class Empty(ListRetriever):
        def has_documents(self):
            return False

    assert corrective(llm, Empty([])).run(QUESTION)["answer"] == \
        "Please index your documents first."
    assert llm.calls == []


@pytest.mark.parametrize("options, message", [
    ({"refine": "yes"}, "refine must be True or False"),
    ({"max_retries": -1}, "max_retries must be a whole number of 0 or more"),
    ({"top_k": 2, "top_n": 3}, r"top_n \(3\) cannot be more than top_k \(2\)"),
    ({"upper": 2, "lower": 5}, r"lower \(5\) cannot be more than upper \(2\)"),
])
def test_bad_settings_raise_a_clear_error(options, message):
    with pytest.raises(ValueError, match=message):
        corrective(ScriptLLM([]), ListRetriever([]), **options)


def test_works_end_to_end_on_a_real_index(tmp_path):
    (tmp_path / "hr.md").write_text("Every full-time employee gets 20 paid leave days per "
                                    "calendar year. Sick days need a doctor's note.",
                                    encoding="utf-8")
    store = MemoryStore()
    IndexingPipeline(WordChunker(8, 0), HashEmbedder(64), store).run([tmp_path / "hr.md"])
    llm = ScriptLLM(['{"1": 9, "2": 2}'], refine="[1]", answer="20 days [chunk_0]")
    rag = corrective(llm, DenseRetriever(HashEmbedder(64), store), top_k=2)

    result = rag.run(QUESTION)

    assert result["answer"] == "20 days [chunk_0]"
    assert len(result["sources"]) == 1
    assert "grade" in describe_steps(result["steps"])
    assert_follows_flow(rag, result)
