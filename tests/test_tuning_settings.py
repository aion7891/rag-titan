"""
Tests for the tuning settings (TOP_K, TOP_N, ADAPTIVE_MAX_ATTEMPTS, PARENT_BUDGET)
in code, in .env and on the command line (hash embedder, fake LLM, no Ollama).
"""

import pytest
from conftest import FakeLLM, PromptLLM

import rag_titan.factory as factory
from rag_titan import AdaptiveRAG, ParentChildRAG, RAGTitan, SimpleRAG, cli, get_settings
from rag_titan.evaluation import end_to_end
from rag_titan.rag_types.rag_registry import CLASSES, build_rag, rag_options
from rag_titan.vector_stores import SQLiteStore


@pytest.fixture(autouse=True)
def folder(tmp_path, monkeypatch):
    """Work in tmp_path with one policy file, the hash embedder and a fake LLM."""
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    for name in ("TOP_K", "TOP_N", "ADAPTIVE_MAX_ATTEMPTS", "PARENT_BUDGET", "VECTOR_STORE"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "leave.md").write_text(
        "# Leave\n\n## Sick leave\n\nTell your manager before 10 am. "
        "From the third sick day you need a note from the doctor.\n", encoding="utf-8")
    return tmp_path


def test_settings_from_the_environment(monkeypatch):
    settings = get_settings()
    assert (settings["top_k"], settings["top_n"]) == (None, None)    # each type's own
    assert (settings["adaptive_max_attempts"], settings["parent_budget"]) == (2, 3000)

    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("TOP_N", " ")                                  # empty means default
    monkeypatch.setenv("PARENT_BUDGET", "1200")
    settings = get_settings()
    assert (settings["top_k"], settings["top_n"], settings["parent_budget"]) == (9, None, 1200)

    monkeypatch.setenv("TOP_K", "many")
    with pytest.raises(ValueError, match="TOP_K must be a whole number"):
        get_settings()


def test_rag_options_per_type():
    settings = get_settings({"top_k": 10, "adaptive_max_attempts": 3, "parent_budget": 900})

    assert rag_options("simple", settings) == {"top_k": 10, "top_n": 2}
    assert rag_options("adaptive", settings) == {"top_k": 10, "top_n": 2, "max_attempts": 3}
    assert rag_options("parent_child", settings) == {"top_k": 10, "top_n": 4,
                                                     "parent_budget": 900}
    assert rag_options("simple", None) == {}


def test_only_top_k_keeps_top_n_within_it():
    assert rag_options("parent_child", get_settings({"top_k": 3}))["top_n"] == 3
    assert rag_options("simple", get_settings({"top_k": 1}))["top_n"] == 1
    assert rag_options("simple", get_settings({"top_k": 6, "top_n": 6}))["top_n"] == 6


def test_class_defaults_are_the_documented_ones():
    sizes = {name: (cls.DEFAULT_TOP_K, cls.DEFAULT_TOP_N) for name, cls in CLASSES.items()}
    assert sizes == {"simple": (5, 2), "adaptive": (4, 2), "parent_child": (8, 4), "hyde": (5, 2),
                     "rag_fusion": (5, 2), "corrective": (5, 2), "iterative": (3, 2)}


def test_build_rag_reads_the_tuning_from_the_parts(parts):
    parts = {**parts, "settings": get_settings({"top_k": 7, "top_n": 3,
                                                "adaptive_max_attempts": 4,
                                                "parent_budget": 500})}

    simple, adaptive, parent = (build_rag(name, parts)
                                for name in ("simple", "adaptive", "parent_child"))
    assert (simple.top_k, simple.top_n) == (7, 3)
    assert (adaptive.top_k, adaptive.max_attempts) == (7, 4)
    assert (parent.top_k, parent.top_n, parent.parent_budget) == (7, 3, 500)
    assert build_rag("simple", {k: v for k, v in parts.items() if k != "settings"}).top_k == 5


@pytest.mark.parametrize("make", [
    lambda p: SimpleRAG(p["retriever"], p["reranker"], p["generator"], top_k=2, top_n=3),
    lambda p: SimpleRAG(p["retriever"], p["reranker"], p["generator"], top_k=0),
    lambda p: AdaptiveRAG(p["llm"], p["retriever"], p["reranker"], p["generator"],
                          max_attempts=0),
    lambda p: ParentChildRAG(p["retriever"], p["reranker"], p["generator"], top_n=True),
])
def test_bad_sizes_raise_a_clear_error(parts, make):
    with pytest.raises(ValueError, match="top_n|top_k|max_attempts"):
        make(parts)


def test_adaptive_sizes_grow_from_top_k(parts):
    llm = FakeLLM(["complex", "no", "wider question", "no"])
    rag = AdaptiveRAG(llm, parts["retriever"], parts["reranker"], parts["generator"],
                      max_attempts=2, top_k=3, top_n=1)
    searches = [s["input"] for s in rag.run("leave and sick pay?")["steps"]
                if s["step"] in ("retrieve", "rerank")]

    # complex: 2 x (3, 1) first, then one retry adds (3, 1) more
    assert [s.get("top_k", s.get("top_n")) for s in searches] == [6, 2, 9, 3]
    assert rag.search_sizes("simple") == (3, 1)


def test_rag_titan_takes_the_tuning():
    rag = RAGTitan(rag_type="parent_child", settings={"top_k": 12, "parent_budget": 800})
    assert (rag.rag.top_k, rag.rag.top_n, rag.rag.parent_budget) == (12, 4, 800)


def test_cli_index_and_ask_with_parent_child(folder, capsys):
    cli.main(["index", "--files", "leave.md", "--vector-store", "sqlite",
              "--rag-type", "parent_child", "--parent-size", "400"])
    chunks = SQLiteStore("rag_titan.db").all_chunks()
    assert chunks and all(c["parent_text"].startswith("Leave > Sick leave") for c in chunks)

    cli.main(["ask", "--question", "doctor's note?", "--vector-store", "sqlite", "--trace",
              "--rag-type", "parent_child", "--top-k", "3", "--top-n", "2",
              "--parent-budget", "900", "--reranker", "none"])
    out = capsys.readouterr().out
    assert "top_k=3" in out and "expand" in out and "#parent_0" in out


@pytest.mark.parametrize("value", ["0", "-2", "two"])
def test_cli_refuses_bad_numbers(value, capsys):
    with pytest.raises(SystemExit):
        cli.main(["ask", "--question", "q", "--files", "leave.md", "--top-k", value])
    assert "--top-k" in capsys.readouterr().err


def test_evaluate_setups_carry_the_tuning():
    args = cli.build_parser().parse_args(
        ["evaluate", "--files", "a.md", "--questions", "q.json", "--rag-type", "adaptive",
         "--top-k", "8", "--max-attempts", "3"])
    assert cli.build_setups(args)[0]["settings"] == {"top_k": 8, "adaptive_max_attempts": 3}


def test_tuning_never_builds_a_second_index(folder):
    indexes = {}
    for top_k in (3, 9):
        parts = end_to_end._indexed_parts(["leave.md"], {"top_k": top_k}, indexes, "simple")
        assert build_rag("simple", parts).top_k == top_k
    assert len(indexes) == 1
