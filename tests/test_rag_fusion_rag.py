"""Tests for RAG-Fusion: search the question and several rewordings, fuse with RRF."""

import pytest
from conftest import FakeLLM, PromptLLM

import rag_titan.factory as factory
from rag_titan import RAGFusionRAG, RAGTitan, cli, get_settings
from rag_titan.chunking import WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline
from rag_titan.prompts import MULTI_QUERY_PROMPT
from rag_titan.rag_types.rag_fusion_rag import parse_queries
from rag_titan.rag_types.rag_registry import rag_options
from rag_titan.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from rag_titan.retrieval.rerankers import NoReranker
from rag_titan.vector_stores import MemoryStore

QUESTION = "My notebook will not start"
QUERIES = "laptop does not turn on\ncomputer power problem\nlaptop battery dead"


@pytest.fixture
def store(tmp_path):
    (tmp_path / "it.md").write_text("If your laptop does not turn on, charge it for an hour. "
                                    "A dead battery is replaced by IT.", encoding="utf-8")
    (tmp_path / "hr.md").write_text("You get 20 paid leave days per year.", encoding="utf-8")
    store = MemoryStore()
    IndexingPipeline(WordChunker(8, 0), HashEmbedder(64), store).run(
        [tmp_path / "it.md", tmp_path / "hr.md"])
    return store


def fusion(store, replies, retriever=None, **options):
    llm = FakeLLM(replies)
    retriever = retriever or DenseRetriever(HashEmbedder(64), store)
    return RAGFusionRAG(llm, retriever, NoReranker(), GroundedGenerator(llm), **options), llm


def names(result):
    return [step["step"] for step in result["steps"]]


def follows_flow(rag, result):
    edges = {tuple(edge) for edge in rag.flow()["edges"]}
    steps = names(result)
    return all(pair in edges for pair in zip(steps, steps[1:]))


def test_searches_the_question_and_every_rewording_then_fuses(store):
    rag, llm = fusion(store, [QUERIES, "Charge it for an hour [chunk_0]."])
    result = rag.run(QUESTION)

    assert names(result) == ["write_queries", "retrieve", "retrieve", "retrieve", "retrieve",
                             "fuse_queries", "rerank", "generate"]
    searched = [s["input"]["query"] for s in result["steps"] if s["step"] == "retrieve"]
    assert searched == [QUESTION, *QUERIES.splitlines()]
    system, user = llm.calls[0][0]["content"], llm.calls[0][1]["content"]
    assert system == MULTI_QUERY_PROMPT and "Number of queries: 3" in user and QUESTION in user
    assert result["steps"][1 + 4]["output"][0]["score_trace"].keys() >= {"query_1_rank"}
    assert result["steps"][-1]["input"]["query"] == QUESTION      # the answer: original question
    assert follows_flow(rag, result)


def test_no_usable_rewording_searches_the_question_only(store, caplog):
    rag, _ = fusion(store, ["Here are some queries:\n\n" + QUESTION, "the answer"])
    result = rag.run(QUESTION)

    assert names(result) == ["write_queries", "retrieve", "rerank", "generate"]
    assert result["steps"][0]["output"] == []
    assert "searching with the question only" in caplog.text
    assert follows_flow(rag, result)


def test_with_hybrid_each_search_shows_its_steps(store):
    hybrid = HybridRetriever(DenseRetriever(HashEmbedder(64), store, dedupe_overlap=None),
                             BM25Retriever(store, dedupe_overlap=None))
    rag, _ = fusion(store, ["laptop does not turn on", "the answer"], retriever=hybrid)
    result = rag.run(QUESTION)

    one = ["vector_search", "keyword_search", "fuse"]
    assert names(result) == ["write_queries", *one, *one, "fuse_queries", "rerank", "generate"]
    assert follows_flow(rag, result)


@pytest.mark.parametrize("reply, expected", [
    ("1. laptop does not turn on\n2) computer power problem\n3: dead battery",
     ["laptop does not turn on", "computer power problem", "dead battery"]),
    ("- laptop does not turn on\n* computer power problem\n• dead battery",
     ["laptop does not turn on", "computer power problem", "dead battery"]),
    ('Here are 3 search queries:\n"laptop does not turn on"\nQuery 2: computer power problem',
     ["laptop does not turn on", "computer power problem"]),
    ("laptop does not turn on\nLaptop does NOT turn on!\nmy notebook will not start.",
     ["laptop does not turn on"]),                                   # duplicate, the question
    ("a\nb\nc\nd\ne", ["a", "b", "c"]),                               # at most count
    ("", []),
    (None, []),
])
def test_parse_queries_reads_what_small_models_write(reply, expected):
    assert parse_queries(reply, QUESTION, 3) == expected


@pytest.mark.parametrize("queries", [0, 11, "3", True, 2.5])
def test_bad_query_counts_raise(queries):
    with pytest.raises(ValueError, match="queries"):
        RAGFusionRAG(FakeLLM(), None, None, None, queries=queries)


def test_empty_question_and_empty_index(store):
    rag, _ = fusion(store, [QUERIES])
    assert rag.answer("  ") == "Please enter a question."
    empty = RAGFusionRAG(FakeLLM(), DenseRetriever(HashEmbedder(64), MemoryStore()),
                         NoReranker(), None)
    assert empty.answer("laptop?") == "Please index your documents first."


def test_setting_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FUSION_QUERIES", "5")
    assert rag_options("rag_fusion", get_settings()) == {"queries": 5, "question_weight": 1.0,
                                                          "keep_top": False}
    monkeypatch.delenv("FUSION_QUERIES")
    assert get_settings()["fusion_queries"] == 3

    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "it.md").write_text("Charge a laptop that does not turn on.", encoding="utf-8")
    assert RAGTitan(rag_type="rag_fusion", settings={"fusion_queries": 2}).rag.queries == 2
    cli.main(["ask", "--files", "it.md", "--question", "notebook dead?", "--rag-type",
              "rag_fusion", "--fusion-queries", "2", "--trace"])
    out = capsys.readouterr().out
    assert "write_queries" in out and "fuse_queries" in out
