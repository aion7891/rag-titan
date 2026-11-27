"""
Tests for examples/okf_rag.py, the example bundle and docs/okf.md: the script on
the page, its model-free output, the `okf info` output and the claims about them.
"""

import json
import re
import runpy
from pathlib import Path

import pytest
from conftest import PromptLLM

import rag_titan.factory as factory
from rag_titan import cli
from rag_titan.okf import read_bundle, validate_bundle
from rag_titan.okf.documents import okf_documents

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
BUNDLE = EXAMPLES / "okf_bundle"
PAGE = (ROOT / "docs" / "okf.md").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(not BUNDLE.is_dir(), reason="examples are only in the repo")


def test_the_page_shows_the_script_exactly():
    script = (EXAMPLES / "okf_rag.py").read_text(encoding="utf-8")
    assert f"```python\n{script}```" in PAGE


def test_the_example_bundle_is_conformant_with_one_expected_warning():
    report = validate_bundle(BUNDLE)

    assert report["conformant"] and report["errors"] == []
    assert [(w["path"], w["message"]) for w in report["warnings"]] == [
        ("README.md", "not a concept: no frontmatter")]


def test_the_info_output_on_the_page(capsys):
    cli.main(["okf", "info", str(BUNDLE)])
    shown = capsys.readouterr().out
    # the stale count depends on today's date: the draft is due since 2026-06-30
    assert f"```text\n{shown}```" in PAGE


def test_the_script_runs_and_its_model_free_part_is_on_the_page(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(EXAMPLES / "okf_rag.py"))
    out = capsys.readouterr().out

    checked = out.split("\nOKF_INCLUDE=")[0]
    assert checked in PAGE
    assert "OKF_INCLUDE=current (5 chunks)" in out and "OKF_INCLUDE=all (8 chunks)" in out


def test_each_question_s_values_sit_in_the_right_concepts():
    questions = json.loads((EXAMPLES / "okf_questions.json").read_text(encoding="utf-8"))
    current = " ".join(d["text"] for d in okf_documents(BUNDLE))
    old = " ".join(c["body"] for c in read_bundle(BUNDLE)["concepts"]
                   if c["frontmatter"].get("status") == "deprecated")

    def says(text, number):        # a whole number: "20" is not inside "2026"
        return re.search(rf"(?<!\d){number}(?!\d)", text) is not None

    for question in questions:
        assert says(current, question["current"]), question["id"]
        assert says(old, question["old"]) and not says(current, question["old"]), question["id"]


def test_three_policies_point_to_their_replacements():
    replaced = {d["source"]: d["metadata"]["replaced_by"]
                for d in okf_documents(BUNDLE, include="all") if "replaced_by" in d["metadata"]}

    assert replaced == {
        "policies/travel/hotel-limit-2025.md": "policies/travel/hotel-limit",
        "policies/remote/home-office-2024.md": "policies/remote/home-office",
        "policies/leave/annual-leave-2024.md": "policies/leave/annual-leave"}
