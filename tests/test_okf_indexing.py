"""Tests for indexing OKF bundles: chunk metadata, filters, citations, settings and the CLI."""

import json
import zipfile
from datetime import datetime, timezone

import pytest
from conftest import FakeLLM, PromptLLM

import rag_titan.factory as factory
from rag_titan import RAGTitan, cli, get_settings
from rag_titan.chunking import ParagraphChunker, ParentChildChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.generation.context_builder import describe_source
from rag_titan.okf.documents import okf_documents
from rag_titan.pipelines import IndexingPipeline
from rag_titan.vector_stores import MemoryStore, SQLiteStore

REVIEWED = ('generated: { by: agent/x, at: "2026-06-20T14:00:00Z" }\n'
            'verified: { by: "human:ana", at: "2026-06-25T09:00:00Z" }\n')
BUNDLE = {
    "index.md": '---\nokf_version: "0.3"\ntitle: Acme travel\n---\n',
    "travel/hotel-limit.md": ("---\ntype: Policy\ntitle: Hotel limit\n"
                              "description: Hotel costs per night.\ntags: [travel]\n"
                              f"{REVIEWED}---\nHotels are refunded up to 180 euros a night.\n"),
    "travel/hotel-limit-2025.md": ("---\ntype: Policy\nstatus: deprecated\nsuperseded_by: "
                                   "hotel-limit.md\n---\nHotels are refunded up to 150 euros a "
                                   "night.\n"),
    "travel/train.md": ("---\ntype: Policy\nstatus: draft\nstale_after: \"2026-01-01\"\n---\n"
                        "Second class for trips under six hours.\n"),
    "travel/visa.md": "---\ntype: Policy\nstatus: retired\n---\nVisa rules.\n",
    "README.md": "# not a concept\n",
}
NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "acme"
    for path, text in BUNDLE.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return root


def by_source(documents):
    return {document["source"]: document for document in documents}


def test_current_knowledge_leaves_out_deprecated_and_unknown_statuses(bundle):
    documents = by_source(okf_documents(bundle, now=NOW))

    assert sorted(documents) == ["travel/hotel-limit.md", "travel/train.md"]   # draft stays
    hotel = documents["travel/hotel-limit.md"]
    assert hotel["text"].startswith("# Hotel limit\n\nHotel costs per night.\n\nHotels are")
    assert hotel["metadata"] == {"okf_concept": "travel/hotel-limit", "okf_type": "Policy",
                                 "tags": ["travel"], "status": "stable",
                                 "trust_tier": "human-reviewed", "stale": False,
                                 "okf_bundle": "Acme travel"}
    assert documents["travel/train.md"]["metadata"]["stale"] is True


def test_all_keeps_every_concept_and_says_what_replaced_an_old_one(bundle):
    documents = by_source(okf_documents(bundle, include="all", now=NOW))

    assert len(documents) == 4
    old = documents["travel/hotel-limit-2025.md"]["metadata"]
    assert (old["status"], old["replaced_by"]) == ("deprecated", "travel/hotel-limit")
    json.dumps(list(documents.values()))              # plain values: a saved index keeps them


def test_a_minimum_trust_tier(bundle):
    documents = okf_documents(bundle, min_trust="human-reviewed", now=NOW)

    assert [d["source"] for d in documents] == ["travel/hotel-limit.md"]


@pytest.mark.parametrize("options, message", [({"include": "everything"}, "OKF_INCLUDE"),
                                              ({"min_trust": "trusted"}, "OKF_MIN_TRUST")])
def test_bad_options_raise(bundle, options, message):
    with pytest.raises(ValueError, match=message):
        okf_documents(bundle, **options)
    with pytest.raises(ValueError, match=message):
        IndexingPipeline(None, None, None, **{f"okf_{key}": value
                                               for key, value in options.items()})


def test_every_chunk_gets_its_own_copy_of_the_metadata():
    documents = [{"source": "a.md", "text": "One.\n\nTwo.", "metadata": {"tags": ["x"]}},
                 {"source": "b.md", "text": "Three."}]

    chunks = ParagraphChunker(6).chunk(documents)

    assert [c.get("metadata") for c in chunks] == [{"tags": ["x"]}, {"tags": ["x"]}, None]
    assert chunks[0]["metadata"] is not chunks[1]["metadata"]
    children = ParentChildChunker(ParagraphChunker(10), 100).chunk(documents)
    assert children[0]["metadata"] == {"tags": ["x"]} and "metadata" not in children[-1]


def test_the_source_line_tells_the_model_how_far_to_trust_a_passage():
    plain = {"id": "chunk_1", "source": "a.md"}
    okf = {"id": "chunk_2", "source": "travel/old.md", "metadata": {
        "okf_concept": "travel/old", "trust_tier": "unverified", "status": "deprecated",
        "replaced_by": "travel/new", "stale": True}}

    assert describe_source(plain) == "[chunk_1] Source: a.md"                 # unchanged
    assert describe_source(okf) == (
        "[chunk_2] Source: travel/old.md, trust: unverified, "
        "status: deprecated, replaced by travel/new, stale: due for a check")


def index(bundle, store, **options):
    pipeline = IndexingPipeline(ParagraphChunker(300), HashEmbedder(64), store, **options)
    return pipeline.run([bundle])


def test_a_bundle_folder_or_zip_is_indexed_and_a_saved_index_keeps_the_metadata(bundle, tmp_path):
    assert index(bundle, MemoryStore()) == 2

    archive = tmp_path / "acme.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in bundle.rglob("*"):
            zipped.write(path, "acme/" + path.relative_to(bundle).as_posix())
    store = SQLiteStore(str(tmp_path / "index.db"))
    assert index(archive, store, okf_include="all") == 4

    reopened = SQLiteStore(str(tmp_path / "index.db"))
    saved = {c["source"]: c["metadata"] for c in reopened.all_chunks()}
    assert saved["travel/hotel-limit-2025.md"]["replaced_by"] == "travel/hotel-limit"


def test_settings_and_the_answer_prompt(bundle, monkeypatch, tmp_path):
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.delenv("OKF_INCLUDE", raising=False)
    monkeypatch.chdir(tmp_path)
    llm = FakeLLM(["Up to 180 euros a night [chunk_0]."])
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: llm)
    assert (get_settings()["okf_include"], get_settings()["okf_min_trust"]) == ("current",
                                                                                "unverified")

    rag = RAGTitan(settings={"chunker": "paragraph", "chunk_size": 300})
    rag.index([bundle])

    assert rag.ask("What is the hotel limit per night?") == "Up to 180 euros a night [chunk_0]."
    prompt = llm.calls[-1][-1]["content"]
    assert "travel/hotel-limit.md" in prompt and "trust: human-reviewed" in prompt
    assert "150 euros" not in prompt                  # the deprecated concept was not indexed


def test_a_bad_setting_fails_when_the_rag_is_built(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setenv("OKF_MIN_TRUST", "very-high")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())

    with pytest.raises(ValueError, match="OKF_MIN_TRUST must be one of"):
        RAGTitan()


def test_the_cli_indexes_a_bundle_with_the_flags(bundle, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())

    cli.main(["ask", "--files", str(bundle), "--question", "hotel limit?", "--okf-include", "all",
              "--okf-min-trust", "unverified", "--trace"])
    assert "retrieve" in capsys.readouterr().out
    args = cli.build_parser().parse_args(["index", "--files", str(bundle), "--okf-include", "all"])
    assert args.okf_include == "all"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["ask", "--question", "q", "--okf-min-trust", "high"])
