"""
Element-aware Markdown chunker (chunking handbook 3.5 and 3.7).

Like the heading chunker it never mixes two sections and starts each chunk with
its heading path. Inside a section it packs whole blocks, so a code block with
blank lines, a table, a list item or a "> Warning" callout is never cut, and a
lead-in line ("You get:") stays with the list after it. A block too big for a
chunk is cut at its own lines (see block_pieces.py); a big table gets chunks of
its own, and each one starts with the lead-in and the header row.
"""

from rag_titan.chunking.fixed_size.recursive_chunker import RecursiveChunker
from rag_titan.chunking.markdown.block_pieces import block_pieces, table_parts
from rag_titan.chunking.markdown.blocks import glue_lead_ins, split_blocks
from rag_titan.chunking.offsets import trim_span
from rag_titan.chunking.packing import pack
from rag_titan.chunking.structure_aware.base import PieceChunker
from rag_titan.chunking.structure_aware.heading_chunker import HeadingChunker


class MarkdownChunker(HeadingChunker):
    """
    Splits Markdown at its headings, then packs whole blocks (paragraphs, lists,
    tables, code blocks, quotes) up to `chunk_size`.

    Args:
        chunk_size: largest chunk, in characters, including the heading path.
        overlap: whole blocks (or rows, items, lines) repeated inside a section,
            up to this size.
        include_path: start each chunk with its heading path, for example
            "Leave policy > Sick leave".
    """

    def __init__(self, chunk_size=1000, overlap=0, include_path=True):
        super().__init__(chunk_size, overlap, include_path)

    def section_packer(self, room):
        """Pack the blocks of one section into `room` characters."""
        return BlockPacker(room, min(self.overlap, room - 1))


class BlockPacker(PieceChunker):
    """Packs the Markdown blocks of one section (used by MarkdownChunker)."""

    def split_spans(self, text):
        """(chunk, start, end) for the whole blocks of `text`, packed up to chunk_size."""
        if not text.strip():
            return []

        spans, run, position = [], [], 0   # run: (piece, start) packed together
        for kind, lead, body in glue_lead_ins(split_blocks(text)):
            unit = lead + body
            if kind == "table" and self.length(unit) > self.chunk_size:
                spans += self.pack_run(text, run) + self.table_spans(text, lead, body, position)
                run = []
            else:
                run += self.located(self.unit_pieces(kind, lead, body), position)
            position += len(unit)
        return spans + self.pack_run(text, run)

    def unit_pieces(self, kind, lead, body):
        """The pieces of one block (with its lead-in): whole when it fits, else cut."""
        if self.length(lead + body) <= self.chunk_size:
            return [lead + body]
        pieces = self.fit(block_pieces("paragraph", lead), self.word_splitter.split_pieces)
        return pieces + self.fit(block_pieces(kind, body), self.word_splitter.split_pieces)

    def located(self, pieces, start):
        """Pair each piece with where it starts, counting from `start`."""
        result = []
        for piece in pieces:
            result.append((piece, start))
            start += len(piece)
        return result

    def pack_run(self, text, run, size=None, context=""):
        """
        Pack located pieces into chunks of at most `size` (default chunk_size).
        `context` (a table's lead-in and header) goes in front of every chunk
        that does not already start with it.
        """
        if not run:
            return []
        groups = pack([self.length(piece) for piece, _ in run], size or self.chunk_size,
                      min(self.overlap, (size or self.chunk_size) - 1))
        spans = []
        for group in groups:
            (_, first), (last, last_start) = run[group[0]], run[group[-1]]
            start, end = trim_span(text, first, last_start + len(last))
            if start < end:
                spans.append((context + text[start:end], start, end))
        return spans

    def table_spans(self, text, lead, body, position):
        """
        Chunks of a table too big for one: groups of whole rows, each starting
        with the lead-in and the header row. The offsets point at the rows.
        """
        header, rows = table_parts(body)
        context = (lead + header).lstrip()
        room = self.chunk_size - self.length(context)
        if room < self.chunk_size / 2:
            # the header is too big to repeat: whole rows, without the header
            return self.pack_run(text, self.located(self.unit_pieces("table", lead, body),
                                                    position))

        cut = RecursiveChunker(room, 0, (" ", "")).split_pieces
        row_start = position + len(lead + header)
        spans = self.pack_run(text, self.located(self.fit_to(rows, room, cut), row_start),
                              room, context)
        # the first chunk is the table's own start: its exact text, header included
        if spans:
            start, end = trim_span(text, position, spans[0][2])
            spans[0] = (text[start:end], start, end)
        return spans

    def fit_to(self, pieces, size, cut):
        """Keep pieces of at most `size`; cut the others with `cut(piece)`."""
        result = []
        for piece in pieces:
            result.extend(cut(piece) if self.length(piece) > size else [piece])
        return result
