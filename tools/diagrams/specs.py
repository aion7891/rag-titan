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
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    TokenChunker,
    WordChunker,
    count_windows,
    fixed_windows,
)
from rag_titan.chunking.packing import pack  # noqa: E402

TEXT = ("Leave policy\n\nStaff get twenty days. Unused days carry over. "
        "Ask HR for the form. Sick leave is separate.")
# one tricky sentence more, to show what sentence rules change
TRICKY = TEXT + " Dr. Lee signs it within 3.5 days."
# the same sentences, written as paragraphs
PARAGRAPHS = ("Leave policy\n\nStaff get twenty days. Unused days carry over.\n\n"
              "Ask HR for the form. Sick leave is separate.\n\nDr. Lee signs it within 3.5 days.")


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


def piece_spec(name, chunker, text, pieces, unit_name, rule):
    """Spec for a chunker that packs whole pieces (recursive pieces, sentences, paragraphs)."""
    size, overlap = chunker.chunk_size, chunker.overlap
    sizes = [chunker.length(piece) for piece in pieces]
    # like the chunkers: a chunk that is only spaces is dropped
    groups = [g for g in pack(sizes, size, overlap) if "".join(pieces[i] for i in g).strip()]
    return {
        "title": name,
        "subtitle": f"{name}(chunk_size={size}, overlap={overlap}): {rule}",
        "desc": f"{name} cuts the text into {len(pieces)} {unit_name} and packs whole "
                f"{unit_name} into {len(groups)} chunks of at most {size} characters.",
        "units_label": f"The text, cut into {len(pieces)} {unit_name} "
                       "(number = size in characters)",
        "units": pieces, "groups": groups, "joiner": "", "strip": True,
        "show": lambda piece: piece.strip().replace("\n", "↵") + f"  {len(piece)}",
        "max_chars": 60, "chunks": chunker.split_text(text),
        "footer": f"S = {size} · O = {overlap} · K = {len(groups)} chunks · "
                  f"whole {unit_name}, sizes add up to ≤ S",
    }


def recursive_spec():
    chunker = RecursiveChunker(50, 25)
    return piece_spec("RecursiveChunker", chunker, TRICKY, chunker.split_pieces(TRICKY), "pieces",
                      'splits on "\\n\\n", "\\n", ". ", " " until pieces fit. Note "Dr." cut off.')


def sentence_spec():
    chunker = SentenceChunker(50, 25)
    return piece_spec("SentenceChunker", chunker, TRICKY, chunker.pieces(TRICKY), "sentences",
                      'whole sentences; "Dr." and "3.5" do not end a sentence.')


def paragraph_spec():
    chunker = ParagraphChunker(70, 0)
    return piece_spec("ParagraphChunker", chunker, PARAGRAPHS, chunker.pieces(PARAGRAPHS),
                      "paragraphs", "whole paragraphs (split at blank lines), packed up to 70.")


SPECS = {
    "word": word_spec,
    "character": character_spec,
    "token": token_spec,
    "recursive": recursive_spec,
    "sentence": sentence_spec,
    "paragraph": paragraph_spec,
}
