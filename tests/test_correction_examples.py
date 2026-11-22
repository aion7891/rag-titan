"""
Tests for examples/corrective_rag.py and examples/iterative_rag.py, their copies on
the RAG types page, and the two-hop documents. The model is faked; the page's
keyword-search claims do not depend on a model, so they are checked exactly.
"""

import json
import runpy
from pathlib import Path

import pytest
from conftest import PromptLLM

import rag_titan.factory as factory
from rag_titan.chunking import ParagraphChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import BM25Retriever
from rag_titan.vector_stores import MemoryStore

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
PAGE = ROOT / "docs" / "rag-types.md"

pytestmark = pytest.mark.skipif(not (EXAMPLES / "multihop_docs").is_dir(),
                                reason="examples are only in the repo")


@pytest.fixture
def fakes(monkeypatch):
    """A fake LLM and the hash embedder, so the examples run without Ollama."""
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    for name in ("RETRIEVER", "RERANKER", "VECTOR_STORE", "TOP_K", "TOP_N"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("name", ["corrective_rag.py", "iterative_rag.py"])
def test_the_page_shows_the_script_exactly(name):
    script = (EXAMPLES / name).read_text(encoding="utf-8")
    assert f"```python\n{script}```" in PAGE.read_text(encoding="utf-8")


def test_corrective_example_runs(fakes, capsys):
    runpy.run_path(str(EXAMPLES / "corrective_rag.py"))
    out = capsys.readouterr().out
    assert "Grade: incorrect (nothing found)" in out and "Loop: {" in out
    assert out.strip().endswith("Answer: the answer")


def test_iterative_example_runs(fakes, capsys):
    runpy.run_path(str(EXAMPLES / "iterative_rag.py"))
    out = capsys.readouterr().out
    assert "Search 1: 'Who has to approve a new monitor for a Berlin employee?'" in out
    assert out.strip().endswith("Answer: the answer")


def keyword_search(folder):
    store = MemoryStore()
    IndexingPipeline(ParagraphChunker(300, 0), HashEmbedder(), store).run(
        sorted((EXAMPLES / folder).glob("*.md")))
    return BM25Retriever(store)


def test_one_keyword_search_never_reaches_the_second_hop():
    search = keyword_search("multihop_docs")
    questions = json.loads((EXAMPLES / "multihop_questions.json").read_text(encoding="utf-8"))

    for question in questions:
        found = {match["source"] for match in search.retrieve(question["question"], 5)}
        if len(question["required_evidence_ids"]) == 2:
            assert found == {"offices.md"}, question["id"]
        else:
            assert found >= set(question["required_evidence_ids"]), question["id"]


def test_the_page_claims_about_keyword_search_hold():
    # corrective_rag.py: the question finds nothing, the rewritten one finds new_starters.md
    search = keyword_search("retrieval_docs")
    assert search.retrieve("What gear does a newcomer receive?", 5) == []
    rewritten = ("Newcomer equipment distribution; initial starter kit; onboarding supplies; "
                 "new employee provisions")
    assert "new_starters.md" in {m["source"] for m in search.retrieve(rewritten, 5)}

    # iterative_rag.py: the model's own query finds teams.md
    hop = keyword_search("multihop_docs").retrieve("Facilities team approval process", 3)
    assert "teams.md" in {match["source"] for match in hop}
