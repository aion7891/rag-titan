"""
What each diagram shows: the example text, the chunker's units and which
units go into each chunk. Everything comes from running the real chunkers.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))   # use the package in this repo, not an installed one

from rag_titan.chunking import (  # noqa: E402
    CharacterChunker,
    TokenChunker,
    WordChunker,
    count_windows,
    fixed_windows,
)

TEXT = ("Leave policy\n\nStaff get twenty days. Unused days carry over. "
        "Ask HR for the form. Sick leave is separate.")


def window_spec(name, chunker, units, joiner, show, unit_name):
    """Spec for a sliding-window chunker: chunk k = units k*H ... k*H + S - 1."""
    size, overlap = chunker.chunk_size, chunker.overlap
    groups = fixed_windows(list(range(len(units))), size, overlap)
    count = count_windows(len(units), size, overlap)
    return {
        "title": name,
        "subtitle": f"{name}(chunk_size={size}, overlap={overlap}): windows of {size} {unit_name}, "
                    f"each starting {size - overlap} {unit_name} after the last one.",
        "desc": f"{name} cuts the leave-policy text into {len(units)} {unit_name} and makes "
                f"{len(groups)} chunks of {size} {unit_name} that overlap by {overlap}.",
        "units_label": f"The text, cut into {len(units)} {unit_name}",
        "units": units, "groups": groups, "joiner": joiner, "strip": False, "show": show,
        "chunks": chunker.split_text(TEXT),
        "footer": f"N = {len(units)} {unit_name} · S = {size} · O = {overlap} · "
                  f"H = {size - overlap} · K = 1 + ⌈(N − S) / H⌉ = {count} chunks",
    }


def word_spec():
    return window_spec("WordChunker", WordChunker(6, 2), TEXT.split(), " ", str, "words")


def character_spec():
    show = {" ": "·", "\n": "↵"}
    spec = window_spec("CharacterChunker", CharacterChunker(30, 6), list(TEXT), "",
                       lambda char: show.get(char, char), "characters")
    spec["pad"] = 4
    spec["units_label"] += " (· = space, ↵ = line break)"
    return spec


def token_spec():
    chunker = TokenChunker(8, 2)   # needs tiktoken: pip install -e ".[tokens]"
    ids = chunker.tokenizer.encode(TEXT)
    tokens = [chunker.tokenizer.decode([token]) for token in ids]
    show = lambda token: token.replace(" ", "␣").replace("\n", "↵")  # noqa: E731
    spec = window_spec("TokenChunker", chunker, tokens, "", show, "tokens")
    spec["units_label"] += " (tiktoken cl100k_base; ␣ = space, ↵ = line break)"
    return spec


SPECS = {
    "word": word_spec,
    "character": character_spec,
    "token": token_spec,
}
