"""Chunker that follows the heading hierarchy (handbook section 3.3)."""

import re

from rag_titan.chunking.structure_aware.base import PieceChunker
from rag_titan.chunking.structure_aware.paragraph_chunker import ParagraphChunker

# Markdown headings "# Title" to "###### Title" (the DOCX loader writes these too)
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
# ``` or ~~~ starts and ends a code block; "#" lines inside it are code, not headings
FENCE = re.compile(r"^\s*(```|~~~)")


def only_headings(lines):
    """True when `lines` hold at least one heading and nothing else but blank lines."""
    headings = [line for line in lines if HEADING.match(line.rstrip("\r\n"))]
    others = [line for line in lines if line.strip() and line not in headings]
    return bool(headings) and not others


def split_sections(text, keep_heading_lines=False):
    """
    Cut Markdown text into sections at its headings.

    Args:
        text: Markdown text.
        keep_heading_lines: keep each "## Title" line at the start of its body.

    Returns:
        List of (path, body): path is the list of heading titles from the top
        level down, for example ["Leave policy", "Sick leave"].
    """
    sections, levels, body, in_code = [], [], [], False

    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            in_code = not in_code
        match = None if in_code else HEADING.match(line.rstrip("\r\n"))

        if match:
            # a kept heading line with no text under it moves into the next section
            carry = body if keep_heading_lines and only_headings(body) else []
            if not carry:
                sections.append(([title for _, title in levels], "".join(body)))
            level = len(match.group(1))
            # a heading closes every open heading at its level or deeper
            levels = [(lvl, title) for lvl, title in levels if lvl < level]
            levels.append((level, match.group(2).strip()))
            body = carry + [line] if keep_heading_lines else []
        else:
            body.append(line)

    sections.append(([title for _, title in levels], "".join(body)))
    return [(path, body) for path, body in sections if body.strip()]


class HeadingChunker(PieceChunker):
    """
    Splits Markdown text at its headings, so a chunk never mixes two sections.
    Inside a section, whole paragraphs are packed up to `chunk_size` (then
    sentences, then words, when one is too big).

    Args:
        chunk_size: largest chunk, in characters (or tokens with a tokenizer),
            including the heading path.
        overlap: whole paragraphs repeated inside a section, up to this size.
        include_path: start each chunk with its heading path, for example
            "Leave policy > Sick leave". The path is skipped when it would take
            more than half of chunk_size.
        tokenizer: optional, counts tokens instead of characters.
    """

    def __init__(self, chunk_size=1000, overlap=0, include_path=True, tokenizer=None):
        super().__init__(chunk_size, overlap, tokenizer)
        self.include_path = include_path

    def split_text(self, text):
        """Return chunks that each stay inside one section."""
        if not text.strip():
            return []

        chunks = []
        for path, body in split_sections(text, keep_heading_lines=not self.include_path):
            chunks.extend(self.section_chunks(path, body))
        return chunks

    def section_chunks(self, path, body):
        """Chunk one section's body and put the heading path in front."""
        prefix = " > ".join(path) + "\n\n" if self.include_path and path else ""
        room = self.chunk_size - self.length(prefix)

        if room < self.chunk_size / 2:
            prefix, room = "", self.chunk_size   # a very long path would crowd out the text

        inner = ParagraphChunker(room, min(self.overlap, room - 1), self.tokenizer)
        return [prefix + chunk for chunk in inner.split_text(body)]
