"""
Tests that the RRF options are wired everywhere: hybrid search, HyDE and RAG-Fusion,
.env settings, the CLI flags, and the settings that never rebuild the index.
"""

import argparse

import pytest
from conftest import FakeLLM

from rag_titan import HyDERAG, RAGFusionRAG, get_settings
from rag_titan.chunking import WordChunker
from rag_titan.cli_options import add_tuning_options, tuning_settings
from rag_titan.embeddings import HashEmbedder
from rag_titan.evaluation.end_to_end import SEARCH_ONLY
from rag_titan.factory import build_retriever
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline
from rag_titan.rag_types.rag_registry import rag_options
from rag_titan.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from rag_titan.retrieval.rerankers import NoReranker
from rag_titan.vector_stores import MemoryStore

FUSION_SETTINGS = {"HYBRID_DENSE_WEIGHT": "hybrid_dense_weight",
                   "HYBRID_BM25_WEIGHT": "hybrid_bm25_weight",
                   "QUESTION_WEIGHT": "question_weight", "RRF_KEEP_TOP": "rrf_keep_top"}


@pytest.fixture
def store(tmp_path):
    (tmp_path / "hr.md").write_text("Every employee gets 20 paid leave days per year. Sick "
                                    "days need a doctor's note.", encoding="utf-8")
    (tmp_path / "it.md").write_text("Remote work needs a laptop and a headset.",
                                    encoding="utf-8")
    store = MemoryStore()
    IndexingPipeline(WordChunker(8, 0), HashEmbedder(64), store).run(
        [tmp_path / "hr.md", tmp_path / "it.md"])
    return store


@pytest.fixture
def clean_env(monkeypatch):
    for name in FUSION_SETTINGS:
        monkeypatch.delenv(name, raising=False)


def test_the_defaults_change_nothing(clean_env):
    settings = get_settings()

    assert {key: settings[key] for key in FUSION_SETTINGS.values()} == {
        "hybrid_dense_weight": 1.0, "hybrid_bm25_weight": 1.0, "question_weight": 1.0,
        "rrf_keep_top": False}
    for rag_type in ("hyde", "rag_fusion"):
        options = rag_options(rag_type, settings)
        assert (options["question_weight"], options["keep_top"]) == (1.0, False)


def test_env_settings_reach_hybrid_search(store, monkeypatch):
    monkeypatch.setenv("RETRIEVER", "hybrid")
    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "2")
    monkeypatch.setenv("RRF_KEEP_TOP", "true")
    monkeypatch.delenv("HYBRID_BM25_WEIGHT", raising=False)

    hybrid = build_retriever(get_settings(), HashEmbedder(64), store)

    assert hybrid.weights == {"dense": 2.0, "bm25": 1.0} and hybrid.keep_top is True


def test_env_settings_reach_hyde_and_rag_fusion(monkeypatch):
    monkeypatch.setenv("QUESTION_WEIGHT", "1.5")
    monkeypatch.setenv("RRF_KEEP_TOP", "yes")

    options = rag_options("rag_fusion", get_settings())

    assert (options["question_weight"], options["keep_top"]) == (1.5, True)


def test_a_bad_env_value_says_which_setting(monkeypatch):
    monkeypatch.setenv("HYBRID_BM25_WEIGHT", "lots")

    with pytest.raises(ValueError, match="Setting HYBRID_BM25_WEIGHT must be a number"):
        get_settings()


def test_the_playground_s_short_settings_still_build_hybrid(store):
    # playground/presets.py with_retriever() passes only these two keys
    hybrid = build_retriever({"retriever": "hybrid", "dedupe_overlap": 0.5},
                             HashEmbedder(64), store)

    assert hybrid.weights == {"dense": 1.0, "bm25": 1.0} and hybrid.keep_top is False


def hit(chunk_id, retriever):
    return {"id": chunk_id, "source": "a.md", "text": chunk_id, "score": 0.5,
            "retriever": retriever, "score_trace": {retriever: 0.5}}


def hybrid_on(store, **options):
    return HybridRetriever(DenseRetriever(HashEmbedder(64), store, dedupe_overlap=None),
                           BM25Retriever(store, dedupe_overlap=None), **options)


def test_hybrid_weights_and_keep_top_change_the_fused_results(store):
    dense = [hit("a", "dense"), hit("b", "dense"), hit("c", "dense"), hit("d", "dense")]
    keyword = [hit("x", "bm25"), hit("b", "bm25"), hit("y", "bm25"), hit("a", "bm25")]

    def top(top_k=2, **options):
        return [m["id"] for m in hybrid_on(store, **options).fuse_top(dense, keyword, top_k)]

    assert top() == ["b", "a"]                     # b: 2/62 beats a: 1/61 + 1/64
    assert top(dense_weight=3) == ["a", "b"]       # a: 3/61 + 1/64 beats b: 3/62 + 1/62
    assert top(top_k=3) == ["b", "a", "x"]
    assert top(top_k=2, keep_top=True) == ["a", "x"]   # both searches' #1s


@pytest.mark.parametrize("options, message", [
    ({"dense_weight": 0}, "dense_weight must be a number above 0"),
    ({"bm25_weight": -1}, "bm25_weight must be a number above 0"),
    ({"bm25_weight": "2"}, "bm25_weight must be a number above 0"),
    ({"keep_top": "yes"}, "keep_top must be True or False"),
])
def test_bad_hybrid_options_raise(store, options, message):
    with pytest.raises(ValueError, match=message):
        hybrid_on(store, **options)


def fused_input(rag_class, store, replies, question="How many leave days?", **options):
    llm = FakeLLM(replies)
    rag = rag_class(llm, DenseRetriever(HashEmbedder(64), store), NoReranker(),
                    GroundedGenerator(llm), **options)
    steps = rag.run(question)["steps"]
    return [step for step in steps if step["step"] == "fuse_queries"][0]["input"]


def test_hyde_weights_the_question_s_search(store):
    # HyDE searches the made-up passage first, then the question
    used = fused_input(HyDERAG, store, ["Staff get twenty days off.", "20 days."],
                       question_weight=2.0, keep_top=True)

    assert used["weights"] == {"query_1": 1.0, "query_2": 2.0} and used["keep_top"] is True


def test_rag_fusion_weights_the_question_s_search(store):
    # RAG-Fusion searches the question first, then each rewording
    used = fused_input(RAGFusionRAG, store, ["vacation days\nannual leave", "20 days."],
                       question_weight=0.5)

    assert used["weights"] == {"query_1": 0.5, "query_2": 1.0, "query_3": 1.0}
    assert "keep_top" not in used


def test_a_default_run_fuses_as_before(store):
    used = fused_input(RAGFusionRAG, store, ["vacation days\nannual leave", "20 days."])

    assert "weights" not in used and "keep_top" not in used


@pytest.mark.parametrize("rag_class", [HyDERAG, RAGFusionRAG])
@pytest.mark.parametrize("options, message", [
    ({"question_weight": 0}, "question_weight must be a number above 0"),
    ({"question_weight": True}, "question_weight must be a number above 0"),
    ({"keep_top": 1}, "keep_top must be True or False"),
])
def test_bad_question_options_raise(rag_class, options, message):
    with pytest.raises(ValueError, match=message):
        rag_class(FakeLLM(), None, None, None, **options)


def parse(*flags):
    parser = argparse.ArgumentParser()
    add_tuning_options(parser)
    return tuning_settings(parser.parse_args(list(flags)))


def test_cli_flags_change_the_settings():
    assert parse("--hybrid-dense-weight", "2", "--hybrid-bm25-weight", "0.5",
                 "--question-weight", "3", "--rrf-keep-top") == {
        "hybrid_dense_weight": 2.0, "hybrid_bm25_weight": 0.5, "question_weight": 3.0,
        "rrf_keep_top": True}
    assert parse("--no-rrf-keep-top") == {"rrf_keep_top": False}
    assert parse() == {}                           # unset flags keep the .env values


@pytest.mark.parametrize("value", ["0", "-1", "two", "nan", "inf"])
def test_cli_refuses_bad_weights(value):
    with pytest.raises(SystemExit):
        parse("--question-weight", value)


def test_search_only_settings_never_rebuild_the_index():
    assert set(FUSION_SETTINGS.values()) <= set(SEARCH_ONLY)
