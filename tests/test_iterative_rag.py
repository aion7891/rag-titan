"""Tests for Iterative RAG: search, think about what is missing, search again, answer once."""

import pytest
from correction_fakes import ThinkLLM, assert_follows_flow, names

from rag_titan import IterativeRAG
from rag_titan.chunking import WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline
from rag_titan.prompts import GROUNDED_ANSWER_PROMPT, NEXT_STEP_PROMPT, NO_ANSWER
from rag_titan.rag_types import describe_steps
from rag_titan.rag_types.iterative.next_step import NO_DECISION_NOTE
from rag_titan.rag_types.loops import MAX_STEPS, MODEL_DONE, NO_NEW_CHUNKS
from rag_titan.retrieval import BM25Retriever
from rag_titan.retrieval.rerankers import NoReranker
from rag_titan.vector_stores import MemoryStore

QUESTION = "Who approves gear for a new starter in Berlin?"
BERLIN = {"id": "berlin", "source": "offices.md", "score": 0.9,
          "text": "In Berlin, IT requests go to the Facilities team."}
LEAD = {"id": "lead", "source": "teams.md", "score": 0.8,
        "text": "The Facilities team is led by Ana Weber, who approves all gear."}
OTHER = {"id": "other", "source": "teams.md", "score": 0.1, "text": "The gym opens at 7."}


class MapRetriever:
    """Returns the chunks prepared for each query (nothing for an unknown query)."""

    def __init__(self, results):
        self.results = results
        self.queries = []

    def retrieve(self, query, top_k):
        self.queries.append(query)
        return self.results.get(query, [])[:top_k]

    def has_documents(self):
        return True


def iterative(steps, results, **options):
    llm = ThinkLLM(steps, answer="Ana Weber approves it [lead].")
    retriever = MapRetriever(results)
    rag = IterativeRAG(llm, retriever, NoReranker(), GroundedGenerator(llm), **options)
    return rag, llm, retriever


def test_the_second_hop_is_found_from_the_first():
    rag, llm, retriever = iterative(
        ["Berlin requests go to Facilities.\nSEARCH: Facilities team lead", "DONE"],
        {QUESTION: [BERLIN], "Facilities team lead": [LEAD]})

    result = rag.run(QUESTION)

    assert names(result) == ["retrieve", "rerank", "think", "retrieve", "rerank", "think",
                             "generate"]
    assert retriever.queries == [QUESTION, "Facilities team lead"]
    # both hops reach the model, in the order they were found
    assert [source["id"] for source in result["sources"]] == ["berlin", "lead"]
    assert result["answer"] == "Ana Weber approves it [lead]."
    assert result["loop"] == {"steps": 2, "max_steps": 3, "stop_reason": MODEL_DONE}
    assert_follows_flow(rag, result)


def test_think_sees_the_question_the_searches_and_what_was_found():
    rag, llm, _ = iterative(["SEARCH: Facilities team lead", "DONE"],
                            {QUESTION: [BERLIN], "Facilities team lead": [LEAD]})
    rag.run(QUESTION)

    thinks = [user for system, user in llm.calls if system == NEXT_STEP_PROMPT]
    assert thinks[0].startswith(f"QUESTION:\n{QUESTION}\n\nSEARCHES SO FAR:\n- {QUESTION}\n\n")
    assert "Facilities team." in thinks[0] and "Ana Weber" not in thinks[0]
    assert "- Facilities team lead" in thinks[1] and "Ana Weber" in thinks[1]


def test_the_trace_keeps_what_each_think_step_saw():
    rag, _, _ = iterative(["SEARCH: Facilities team lead", "DONE"],
                          {QUESTION: [BERLIN], "Facilities team lead": [LEAD]})
    result = rag.run(QUESTION)

    thinks = [step for step in result["steps"] if step["step"] == "think"]
    assert [len(step["input"]["chunks"]) for step in thinks] == [1, 2]
    assert [len(step["input"]["searches"]) for step in thinks] == [1, 2]


def test_each_search_is_reranked_against_its_own_query():
    rag, _, _ = iterative(["SEARCH: Facilities team lead", "DONE"],
                          {QUESTION: [BERLIN, OTHER, LEAD], "Facilities team lead": [LEAD]},
                          top_n=1)
    result = rag.run(QUESTION)

    reranks = [step for step in result["steps"] if step["step"] == "rerank"]
    assert [step["input"]["query"] for step in reranks] == [QUESTION, "Facilities team lead"]
    # top_n=1 per search; "lead" was seen (not kept) in search 1, so search 2 finds nothing new
    assert [source["id"] for source in result["sources"]] == ["berlin"]
    assert result["loop"]["stop_reason"] == NO_NEW_CHUNKS


def test_the_cap_stops_a_model_that_never_says_done():
    results = {QUESTION: [BERLIN], "a": [LEAD], "b": [OTHER]}
    rag, _, retriever = iterative(["SEARCH: a", "SEARCH: b", "SEARCH: c"], results, max_steps=2)

    result = rag.run(QUESTION)

    assert retriever.queries == [QUESTION, "a", "b"]
    assert names(result).count("think") == 2
    assert result["loop"] == {"steps": 2, "max_steps": 2, "stop_reason": MAX_STEPS}
    assert names(result)[-1] == "generate"
    assert_follows_flow(rag, result)


def test_a_search_that_finds_nothing_new_stops_the_loop():
    rag, _, _ = iterative(["SEARCH: berlin again"],
                          {QUESTION: [BERLIN], "berlin again": [BERLIN]})

    result = rag.run(QUESTION)

    assert names(result) == ["retrieve", "rerank", "think", "retrieve", "rerank", "generate"]
    assert result["steps"][4]["output"] == []
    assert result["loop"]["stop_reason"] == NO_NEW_CHUNKS
    assert_follows_flow(rag, result)


def test_zero_steps_searches_once_like_simple_rag():
    rag, llm, _ = iterative(["SEARCH: x"], {QUESTION: [BERLIN]}, max_steps=0)

    result = rag.run(QUESTION)

    assert names(result) == ["retrieve", "rerank", "generate"]
    assert NEXT_STEP_PROMPT not in [system for system, _ in llm.calls]


def test_nothing_found_gives_up_honestly():
    rag, llm, _ = iterative(["DONE"], {})

    result = rag.run(QUESTION)

    assert names(result) == ["retrieve", "rerank", "give_up"]
    assert result["answer"] == NO_ANSWER and result["sources"] == []
    assert GROUNDED_ANSWER_PROMPT not in [system for system, _ in llm.calls]
    assert_follows_flow(rag, result)


def test_an_unusable_think_reply_stops_with_a_note():
    rag, _, _ = iterative(["I am not sure what to do."], {QUESTION: [BERLIN]})

    result = rag.run(QUESTION)

    assert result["steps"][2]["output"]["note"] == NO_DECISION_NOTE
    assert result["loop"]["stop_reason"] == MODEL_DONE
    assert "done (the reply had no SEARCH" in describe_steps(result["steps"])


def test_empty_question_and_empty_index():
    rag, llm, _ = iterative(["DONE"], {})
    assert rag.run("  ")["answer"] == "Please enter a question."

    rag.retriever.has_documents = lambda: False
    assert rag.run(QUESTION)["answer"] == "Please index your documents first."
    assert llm.calls == []


@pytest.mark.parametrize("options, message", [
    ({"max_steps": -1}, "max_steps must be a whole number of 0 or more"),
    ({"max_steps": True}, "max_steps must be a whole number"),
    ({"top_k": 1, "top_n": 2}, r"top_n \(2\) cannot be more than top_k \(1\)"),
])
def test_bad_settings_raise_a_clear_error(options, message):
    with pytest.raises(ValueError, match=message):
        iterative(["DONE"], {}, **options)


def test_works_end_to_end_on_a_real_index(tmp_path):
    (tmp_path / "offices.md").write_text(BERLIN["text"], encoding="utf-8")
    (tmp_path / "teams.md").write_text(LEAD["text"], encoding="utf-8")
    store = MemoryStore()
    IndexingPipeline(WordChunker(20, 0), HashEmbedder(64), store).run(
        [tmp_path / "offices.md", tmp_path / "teams.md"])
    llm = ThinkLLM(["SEARCH: Facilities team led", "DONE"], answer="Ana Weber [chunk_1]")
    rag = IterativeRAG(llm, BM25Retriever(store), NoReranker(), GroundedGenerator(llm),
                       top_k=1, top_n=1)

    result = rag.run("Berlin IT requests: who approves gear?")

    # one chunk per search (top_k=1): the second search found the other file
    assert sorted(source["source"] for source in result["sources"]) == ["offices.md", "teams.md"]
    assert_follows_flow(rag, result)
