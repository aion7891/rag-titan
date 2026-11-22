"""
Tests that Corrective and Iterative RAG are wired everywhere: rag_type names,
.env settings, RAGTitan(settings=...), the CLI flags and the playground knobs.
"""

import pytest
from conftest import PromptLLM

import rag_titan.factory as factory
from rag_titan import RAG_TYPES, CorrectiveRAG, IterativeRAG, RAGTitan, cli, get_settings
from rag_titan.evaluation.end_to_end import SEARCH_ONLY
from rag_titan.playground.tuning import TUNING_KEYS, check_tuning, default_tuning
from rag_titan.rag_types.rag_registry import CLASSES, rag_options

CORRECTION_SETTINGS = ["crag_upper", "crag_lower", "crag_refine", "crag_max_retries",
                       "iterative_max_steps"]


def test_both_types_are_registered():
    assert RAG_TYPES[-2:] == ["corrective", "iterative"]
    assert CLASSES["corrective"] is CorrectiveRAG and CLASSES["iterative"] is IterativeRAG


def test_the_defaults(monkeypatch):
    for name in ("CRAG_UPPER", "CRAG_LOWER", "CRAG_REFINE", "CRAG_MAX_RETRIES",
                 "ITERATIVE_MAX_STEPS"):
        monkeypatch.delenv(name, raising=False)
    settings = get_settings()

    assert {key: settings[key] for key in CORRECTION_SETTINGS} == {
        "crag_upper": 7, "crag_lower": 3, "crag_refine": True, "crag_max_retries": 1,
        "iterative_max_steps": 3}
    assert rag_options("corrective", settings) == {"upper": 7, "lower": 3, "refine": True,
                                                   "max_retries": 1}
    assert rag_options("iterative", settings) == {"max_steps": 3}


def test_env_settings_reach_the_classes(monkeypatch):
    monkeypatch.setenv("CRAG_UPPER", "8")
    monkeypatch.setenv("CRAG_LOWER", "2")
    monkeypatch.setenv("CRAG_REFINE", "false")
    monkeypatch.setenv("CRAG_MAX_RETRIES", "0")
    monkeypatch.setenv("ITERATIVE_MAX_STEPS", "5")
    settings = get_settings()

    assert rag_options("corrective", settings) == {"upper": 8, "lower": 2, "refine": False,
                                                   "max_retries": 0}
    assert rag_options("iterative", settings) == {"max_steps": 5}


def test_a_bad_env_value_says_which_setting(monkeypatch):
    monkeypatch.setenv("CRAG_UPPER", "high")

    with pytest.raises(ValueError, match="Setting CRAG_UPPER must be a whole number"):
        get_settings()


def test_only_top_k_keeps_top_n_within_it():
    assert rag_options("iterative", {"top_k": 1}) == {"top_k": 1, "top_n": 1}


def test_ragtitan_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.chdir(tmp_path)

    corrective = RAGTitan(rag_type="corrective",
                          settings={"crag_upper": 9, "crag_lower": 1, "crag_refine": False,
                                    "crag_max_retries": 2}).rag
    iterative = RAGTitan(rag_type="iterative", settings={"iterative_max_steps": 0}).rag

    assert (corrective.grader.upper, corrective.grader.lower) == (9, 1)
    assert (corrective.refine, corrective.max_retries) == (False, 2)
    assert iterative.max_steps == 0


def test_bad_settings_fail_when_the_rag_is_built(tmp_path, monkeypatch):
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=r"lower \(8\) cannot be more than upper \(7\)"):
        RAGTitan(rag_type="corrective", settings={"crag_lower": 8})


@pytest.fixture
def ask(tmp_path, monkeypatch, capsys):
    """Run `rag-titan ask` on a small file and return what it printed."""
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hr.md").write_text("You get 20 paid leave days.", encoding="utf-8")

    def run(*options):
        cli.main(["ask", "--files", "hr.md", "--question", "leave days?", "--trace", *options])
        return capsys.readouterr().out
    return run


def test_cli_corrective_flags(ask):
    out = ask("--rag-type", "corrective", "--crag-max-retries", "0", "--no-crag-refine",
              "--crag-upper", "9", "--crag-lower", "0")

    assert "grade" in out and "rewrite" not in out and "refine" not in out


def test_cli_iterative_flags(ask):
    out = ask("--rag-type", "iterative", "--iterative-max-steps", "0")

    assert "retrieve" in out and "think" not in out


@pytest.mark.parametrize("flag, value", [
    ("--crag-upper", "11"), ("--crag-lower", "-1"), ("--crag-upper", "7.5"),
    ("--crag-max-retries", "-1"), ("--iterative-max-steps", "two"),
])
def test_cli_refuses_bad_values(ask, flag, value):
    with pytest.raises(SystemExit):
        ask("--rag-type", "corrective", flag, value)


def test_search_only_settings_never_rebuild_the_index():
    assert set(CORRECTION_SETTINGS) <= set(SEARCH_ONLY)


def test_playground_knobs():
    defaults = default_tuning()

    assert defaults["corrective"] == {"top_k": 5, "top_n": 2, "crag_upper": 7, "crag_lower": 3,
                                      "crag_refine": True, "crag_max_retries": 1}
    assert defaults["iterative"] == {"top_k": 3, "top_n": 2, "iterative_max_steps": 3}
    assert set(CORRECTION_SETTINGS) <= set(TUNING_KEYS)
    assert check_tuning({"crag_upper": 8, "iterative_max_steps": None}, [1500]) == {
        "crag_upper": 8}
