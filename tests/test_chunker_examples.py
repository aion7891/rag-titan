"""
Tests for examples/compare_chunkers.py and the markdown and texttiling sections of
docs/chunking.md: the script on the page, its model-free output, the snippets and
the claims about them.
"""

import json
import runpy
from pathlib import Path

import pytest

from rag_titan.chunking import HeadingChunker, MarkdownChunker, ParagraphChunker, TextTilingChunker

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
PAGE = (ROOT / "docs" / "chunking.md").read_text(encoding="utf-8")
DOCS = EXAMPLES / "structured_docs"

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="examples are only in the repo")


def test_the_page_shows_the_script_exactly():
    script = (EXAMPLES / "compare_chunkers.py").read_text(encoding="utf-8")
    assert f"```python\n{script}```" in PAGE


def test_the_model_free_output_on_the_page_is_what_the_script_prints(monkeypatch, capsys):
    monkeypatch.setenv("EMBED_PROVIDER", "hash")    # part 3 runs, but its numbers differ
    monkeypatch.delenv("TEXTTILING_WINDOW", raising=False)
    runpy.run_path(str(EXAMPLES / "compare_chunkers.py"))
    out = capsys.readouterr().out

    parts_1_and_2 = out.split("\nSearch (")[0]
    assert f"```text\n{parts_1_and_2}" in PAGE


def test_the_gold_spans_and_answers_are_exact_quotes():
    handbook = (DOCS / "it_handbook.md").read_text(encoding="utf-8")
    notes = (DOCS / "meeting_notes.txt").read_text(encoding="utf-8")
    gold = json.loads((EXAMPLES / "chunker_gold.json").read_text(encoding="utf-8"))
    questions = json.loads((EXAMPLES / "chunker_questions.json").read_text(encoding="utf-8"))

    assert all(block in handbook for block in gold["handbook_blocks"])
    assert len(gold["handbook_blocks"]) == 13 and len(gold["meeting_topics"]) == 4
    assert "".join(gold["meeting_topics"]).replace(" ", "") == notes.strip().replace(" ", "")
    assert all(q["answer"] in handbook + notes for q in questions)


def test_three_chunkers_cut_the_backup_row_in_half():
    handbook = (DOCS / "it_handbook.md").read_text(encoding="utf-8")
    answer = "Leave the laptop on and plugged in overnight"

    for chunker in (ParagraphChunker(300), HeadingChunker(300), TextTilingChunker(300)):
        assert not any(answer in chunk for chunk in chunker.split_text(handbook))
    assert any(answer in chunk for chunk in MarkdownChunker(300).split_text(handbook))


MD = ("# Errors\n\nError codes:\n| Code | Meaning |\n|------|---------|\n"
      "| E-1 | VPN down |\n| E-2 | Password too old |\n| E-3 | Disk full |\n"
      "| E-4 | Printer offline |\n| E-5 | Badge blocked |\n\n"
      "## Setup\n\nRun once:\n\n```bash\n# not a heading\n\nsudo setup\n```\n")

NOTES = ("The new laptop arrives on your first day. The laptop comes with a charger and a dock. "
         "IT installs the laptop software before you start. A broken laptop goes back to IT. "
         "Travel is booked through the travel portal. Train travel is preferred for short trips. "
         "Hotel costs for travel are refunded up to 150 euros. Travel receipts go to finance. "
         "Leave days are requested in the leave tool. Every employee has twenty leave days. "
         "Unused leave days carry over to March. Sick leave needs a doctor's note.")


def test_the_markdown_snippet_output():
    chunks = MarkdownChunker(chunk_size=130).split_text(MD)

    assert len(chunks) == 3
    for chunk in chunks:
        assert repr(chunk) in PAGE      # the page prints the list of chunks as Python shows it
    assert HeadingChunker(chunk_size=130).split_text(MD)[1] == (
        "Errors\n\nE-4 | Printer offline |\n| E-5 | Badge blocked |")


def test_the_texttiling_snippet_and_its_table():
    chunker = TextTilingChunker(chunk_size=1000)

    assert [chunk[:30] for chunk in chunker.split_text(NOTES)] == [
        "The new laptop arrives on your", "Travel is booked through the t",
        "Leave days are requested in th"]
    gaps = chunker.tiling(NOTES)["gaps"]
    assert chunker.tiling(NOTES)["cuts"] == [4, 8]
    shown = {3: (0.16, 0.00), 4: (0.00, 0.49), 5: (0.33, 0.00), 6: (0.32, 0.01),
             7: (0.16, 0.16), 8: (0.00, 0.70), 9: (0.37, 0.00)}
    for gap, (score, depth) in shown.items():
        assert round(gaps[gap - 1]["score"], 2) == score
        assert round(gaps[gap - 1]["depth"], 2) == depth
    assert all(gaps[gap - 1]["score"] is None for gap in (1, 2, 10, 11))
