"""
The chunk playground's JS chunkers give exactly the chunks of the Python package.

Hundreds of generated texts (Markdown full of traps, planted topics, tricky
sentences) go through both; every chunk must be identical. Needs Node: skipped
without it.
"""

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from markdown_docs import HANDBOOK, TABLE, random_document
from tiling_docs import THREE_TOPICS, planted_text

from rag_titan.chunking import (
    CharacterChunker,
    MarkdownChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    TextTilingChunker,
    WordChunker,
)
from rag_titan.retrieval.bm25 import STOPWORDS

RUNNER = Path(__file__).parent / "js" / "run_chunkers.js"
TRICKY = [
    "Dr. Smith met J. K. Rowling at 3.5 p.m. on Monday. She said: \"Hi!\" Then they left... "
    "Steps: 1. Open the form. 2. Sign it. Fruit, e.g. apples, is fine. 東京は大きい。大阪も。",
    "Line one\r\nLine two\r\n\r\n# Title\r\n\r\nText under it.\r\n",
    "  \n\n# Only a heading\n\n",
    "Café déjà vu. Ünïcödé wörds hère. Naïve façade! Résumé? Ok.",
]

# tokens that trick sentence splitting: abbreviations, initials, decimals, list numbers
TRAPS = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "vs.", "etc.", "Fig.", "No.", "approx.",
         "e.g.", "i.e.", "U.S.", "J. K.", "3.5", "1.", "2)", "...", "!", "?", "\"Hi!\"", "(see.)",
         "。", "！", "\n", "\n\n", "\n\n# Title\n\n", "\n- item", "\n| a | b |", "\n```\n"]


def trap_text(seed):
    """A text full of sentence-splitting traps, like the playground's own check."""
    rng = random.Random(seed)
    words = ["leave", "Days", "laptop", "the", "Office", "a", "note", "Smith", "week", "x"]
    return " ".join(rng.choice(TRAPS if rng.random() < 0.3 else words)
                    for _ in range(rng.randint(5, 80)))


PYTHON = {
    "markdown": lambda c: MarkdownChunker(c["size"], c["overlap"]),
    "texttiling": lambda c: TextTilingChunker(c["size"], c["overlap"], window=c["window"]),
    "paragraph": lambda c: ParagraphChunker(c["size"], c["overlap"]),
    "sentence": lambda c: SentenceChunker(c["size"], c["overlap"]),
    "recursive": lambda c: RecursiveChunker(c["size"], c["overlap"]),
    "character": lambda c: CharacterChunker(c["size"], c["overlap"]),
    "word": lambda c: WordChunker(c["size"], c["overlap"]),
}

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


def cases():
    """Every (chunker, text, size, overlap, window) to compare."""
    texts = [HANDBOOK, TABLE, THREE_TOPICS, *TRICKY]
    texts += [random_document(seed)[0] for seed in range(120)]
    texts += [planted_text(seed)[0] for seed in range(80)]
    texts += [trap_text(seed) for seed in range(200)]
    result = []
    for number, text in enumerate(texts):
        size = 40 + (number * 37) % 400
        overlap = (number * 13) % (size // 2)
        window = 1 + number % 4
        for mode in PYTHON:
            if mode == "word":
                result.append({"mode": mode, "text": text, "size": 4 + number % 40,
                               "overlap": number % 4, "window": window})
            else:
                result.append({"mode": mode, "text": text, "size": size, "overlap": overlap,
                               "window": window})
    return result


@pytest.fixture(scope="module")
def compared(tmp_path_factory):
    """(cases, the Python chunks, the JS output) for every case."""
    all_cases = cases()
    path = tmp_path_factory.mktemp("parity") / "cases.json"
    path.write_text(json.dumps(all_cases), encoding="utf-8")
    run = subprocess.run(["node", str(RUNNER), str(path)], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    python = [PYTHON[case["mode"]](case).split_text(case["text"]) for case in all_cases]
    return all_cases, python, json.loads(run.stdout)


@pytest.mark.parametrize("mode", list(PYTHON))
def test_every_js_chunker_matches_python(compared, mode):
    all_cases, python, js = compared
    checked = 0
    for case, expected, got in zip(all_cases, python, js["chunks"]):
        if case["mode"] != mode:
            continue
        checked += 1
        assert got == expected, (mode, case["size"], case["overlap"], case["text"][:80])
    assert checked > 200


def test_the_js_stopwords_are_bm25_s(compared):
    assert compared[2]["stopwords"] == sorted(STOPWORDS)
