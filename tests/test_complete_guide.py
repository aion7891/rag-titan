"""
The complete guide (docs/cookbook/complete-guide.md) must show only real things.

The kit in docs/assets/cookbook/your-rag/ is run by tools/cookbook/build_examples.py,
which saves what every command printed in expected/complete-run.txt. These tests
compare the page with those files, so the page cannot drift from the package.
"""

import json
import re
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "cookbook" / "complete-guide.md"
KIT = ROOT / "docs" / "assets" / "cookbook" / "your-rag"

# the source package on PyPI has the tests but not docs/
pytestmark = pytest.mark.skipif(not PAGE.is_file(), reason="the docs are only in the repository")


def blocks(language):
    """Return the text of every ```language block of the page, without its indent."""
    text = PAGE.read_text(encoding="utf-8")
    found = re.findall(rf"^( *)```{language}\n(.*?)^\1```", text, re.M | re.S)
    return [textwrap.dedent(body).strip("\n") for _, body in found]


def kit(name):
    return (KIT / name).read_text(encoding="utf-8")


def test_every_output_block_is_real_output():
    run = kit("expected/complete-run.txt")
    outputs = blocks("text")

    assert len(outputs) >= 8
    for output in outputs:
        assert output in run or output == kit("questions.txt").strip(), output


def test_every_command_was_really_run():
    run = kit("expected/complete-run.txt")
    commands = [line for block in blocks("bash") for line in block.splitlines()
                if line.startswith(("rag-titan ", "python "))]

    assert len(commands) >= 12
    for command in commands:
        if "pip install" not in command:
            assert f"$ {command}\n" in run, command


def test_the_scripts_on_the_page_are_the_kit_scripts():
    shown = blocks("python")
    for name in ("prepare_docs.py", "my_rag.py", "make_data.py", "add_stages.py"):
        assert kit(name).strip("\n") in shown, name


def test_the_json_examples_are_real():
    shown = [json.loads(block) for block in blocks("json")]

    assert json.loads(kit("minimal.json")) in shown
    assert json.loads(kit("expected/golden-draft.json"))[1] in shown     # the draft of q2
    assert json.loads(kit("golden.json"))[1] in shown                    # q2 after review
    first_run = json.loads(kit("expected/my_rag_logs.json"))["runs"][0]
    first_run["query_vector"] = first_run["query_vector"][:3]            # shortened on the page
    assert first_run in shown


def test_the_scorecard_and_the_scores_are_real():
    page = PAGE.read_text(encoding="utf-8")
    rows = [line for line in kit("expected/scorecard.md").splitlines() if line.startswith("|")]
    scores = re.findall(r"^Overall score: ([0-9.]+)/100", kit("expected/complete-run.txt"), re.M)

    assert len(rows) == 9
    for row in rows:
        assert row in page, row
    assert f"**{scores[-2]} → {scores[-1]}**" in page    # baseline -> improved
