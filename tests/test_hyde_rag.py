"""Tests for HyDE RAG: search with a made-up answer (and the question), answer from real chunks."""

import pytest
from conftest import FakeLLM, PromptLLM

import rag_titan.factory as factory
from rag_titan import HyDERAG, RAGTitan, cli, get_settings
from rag_titan.chunking import WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline
from rag_titan.prompts import HYDE_PROMPT
from rag_titan.rag_types.rag_registry import rag_options
from rag_titan.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from rag_titan.retrieval.rerankers import NoReranker
from rag_titan.vector_stores import MemoryStore

PASSAGE = "Employees receive twenty paid leave days every calendar year."


@pytest.fixture
def store(tmp_path):
    (tmp_path / "hr.md").write_text("Every full-time employee gets 20 paid leave days per "
                                    "calendar year. Sick days need a doctor's note.",
                                    encoding="utf-8")
    (tmp_path / "it.md").write_text("Remote work needs a laptop and a headset from IT.",
                                    encoding="utf-8")
    store = MemoryStore()
    IndexingPipeline(WordChunker(8, 0), HashEmbedder(64), store).run(
        [tmp_path / "hr.md", tmp_path / "it.md"])
    return store


def hyde(store, replies, retriever=None, **options):
    llm = FakeLLM(replies)
    retriever = retriever or DenseRetriever(HashEmbedder(64), store)
    return HyDERAG(llm, retriever, NoReranker(), GroundedGenerator(llm), **options), llm


def names(result):
    return [step["step"] for step in result["steps"]]


def follows_flow(rag, result):
    flow, steps = rag.flow(), names(result)
    edges = {tuple(edge) for edge in flow["edges"]}
    return all(pair in edges for pair in zip(steps, steps[1:]))


def test_searches_the_passage_and_the_question_then_fuses(store):
    rag, llm = hyde(store, [PASSAGE, "You get 20 days [chunk_0]."])
    result = rag.run("How many days off do I get?")

    assert names(result) == ["hypothesize", "retrieve", "retrieve", "fuse_queries",
                             "rerank", "generate"]
    searches = [s["input"]["query"] for s in result["steps"] if s["step"] == "retrieve"]
    assert searches == [PASSAGE, "How many days off do I get?"]
    assert llm.calls[0][0]["content"] == HYDE_PROMPT
    assert result["answer"] == "You get 20 days [chunk_0]."
    assert follows_flow(rag, result)


def test_rerank_and_answer_use_the_original_question_and_real_chunks(store):
    rag, llm = hyde(store, [PASSAGE, "the answer"])
    result = rag.run("How many days off do I get?")

    rerank, generate = result["steps"][-2], result["steps"][-1]
    assert rerank["input"]["query"] == generate["input"]["query"] == "How many days off do I get?"
    stored = {chunk["id"] for chunk in store.all_chunks()}
    assert result["sources"] and all(chunk["id"] in stored for chunk in result["sources"])
    # the made-up passage reaches the model only as a search, never as context
    assert PASSAGE not in llm.calls[1][1]["content"]


def test_without_the_question_it_searches_the_passage_only(store):
    rag, _ = hyde(store, [PASSAGE, "the answer"], with_question=False)
    result = rag.run("How many days off do I get?")

    assert names(result) == ["hypothesize", "retrieve", "rerank", "generate"]
    assert result["steps"][1]["input"]["query"] == PASSAGE
    assert follows_flow(rag, result)


def test_an_empty_passage_falls_back_to_the_question(store, caplog):
    rag, _ = hyde(store, ["   ", "the answer"])
    result = rag.run("How many days off do I get?")

    assert names(result) == ["hypothesize", "retrieve", "rerank", "generate"]
    assert result["steps"][1]["input"]["query"] == "How many days off do I get?"
    assert "wrote no passage" in caplog.text


def test_with_hybrid_each_search_shows_its_steps(store):
    hybrid = HybridRetriever(DenseRetriever(HashEmbedder(64), store, dedupe_overlap=None),
                             BM25Retriever(store, dedupe_overlap=None))
    rag, _ = hyde(store, [PASSAGE, "the answer"], retriever=hybrid)
    result = rag.run("How many days off do I get?")

    one = ["vector_search", "keyword_search", "fuse"]
    assert names(result) == ["hypothesize", *one, *one, "fuse_queries", "rerank", "generate"]
    assert follows_flow(rag, result)


def test_empty_question_and_empty_index(store):
    rag, _ = hyde(store, [PASSAGE])
    assert rag.answer("  ") == "Please enter a question."
    empty = HyDERAG(FakeLLM(), DenseRetriever(HashEmbedder(64), MemoryStore()), NoReranker(), None)
    assert empty.answer("leave?") == "Please index your documents first."


@pytest.mark.parametrize("options, message", [
    ({"with_question": "yes"}, "with_question"),
    ({"top_k": 2, "top_n": 3}, "top_n"),
    ({"top_k": 0}, "top_k"),
])
def test_bad_options_raise(options, message):
    with pytest.raises(ValueError, match=message):
        HyDERAG(FakeLLM(), None, None, None, **options)


def test_setting_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HYDE_WITH_QUESTION", "false")
    assert rag_options("hyde", get_settings()) == {"with_question": False,
                                                    "question_weight": 1.0, "keep_top": False}
    monkeypatch.delenv("HYDE_WITH_QUESTION")
    assert get_settings()["hyde_with_question"] is True

    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hr.md").write_text("You get 20 paid leave days.", encoding="utf-8")
    assert RAGTitan(rag_type="hyde", settings={"hyde_with_question": False}).rag.with_question \
        is False
    cli.main(["ask", "--files", "hr.md", "--question", "leave days?", "--rag-type", "hyde",
              "--no-hyde-with-question", "--trace"])
    out = capsys.readouterr().out
    assert "hypothesize" in out and "fuse_queries" not in out


def test_the_log_summarizes_the_fused_lists(store):
    from rag_titan.rag_types import describe_steps

    rag, _ = hyde(store, [PASSAGE, "the answer"])
    log = describe_steps(rag.run("How many days off do I get?")["steps"])
    assert "lists=query_1 " in log and ", query_2 " in log and "{'id'" not in log
