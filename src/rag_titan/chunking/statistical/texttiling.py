"""
TextTiling chunker (Hearst 1997; chunking handbook 5.1): cut where the topic
changes, found from the words alone. No model, and the same result every time.

For text without headings (notes, transcripts, emails): the sentences around
every gap are compared (see cohesion.py), and a new chunk starts where their
vocabulary changes most. A cut next to a paragraph break moves onto it. A topic
longer than chunk_size is packed into several chunks of whole sentences.

Simplified from the paper: whole sentences instead of 20-word pseudo-sentences,
no smoothing of the scores (see cohesion.py), and cuts only at valleys instead of
a minimum distance between cuts.
"""

import re

from rag_titan.chunking.offsets import group_spans, located_spans
from rag_titan.chunking.packing import join_groups, pack
from rag_titan.chunking.statistical.cohesion import boundaries, content_words
from rag_titan.chunking.structure_aware.base import PieceChunker
from rag_titan.chunking.structure_aware.sentences import split_sentences

# sentences compared on each side of a gap (TEXTTILING_WINDOW)
DEFAULT_WINDOW = 3

# a sentence that ends its paragraph carries the blank line after it
ENDS_PARAGRAPH = re.compile(r"\n[ \t]*\n\s*$")


def snap_to_paragraphs(cuts, pieces):
    """Move each cut onto a paragraph break one sentence before or after it."""
    breaks = {i for i in range(1, len(pieces)) if ENDS_PARAGRAPH.search(pieces[i - 1])}
    snapped = []
    for cut in cuts:
        if cut not in breaks:
            near = [gap for gap in (cut - 1, cut + 1) if gap in breaks]
            cut = near[0] if near else cut
        if cut not in snapped:
            snapped.append(cut)
    return sorted(snapped)


class TextTilingChunker(PieceChunker):
    """
    Cuts text into topics by word overlap, then packs each topic into chunks.

    Args:
        chunk_size: largest chunk, in characters (or tokens with a tokenizer).
        overlap: whole sentences repeated between chunks of the same topic,
            up to this size; never across a topic boundary.
        window: sentences compared on each side of a gap (default 3). Text with
            fewer than 2 * window sentences is never cut by topic.
        tokenizer: optional, counts tokens instead of characters.
    """

    def __init__(self, chunk_size=1000, overlap=0, window=DEFAULT_WINDOW, tokenizer=None):
        super().__init__(chunk_size, overlap, tokenizer)
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a whole number of at least 1, got {window!r}")
        self.window = window

    def pieces(self, text):
        """Sentences, with too-long sentences cut at words."""
        return self.fit(split_sentences(text), self.word_splitter.split_pieces)

    def tiling(self, text):
        """
        What the chunker sees, for showing the cohesion curve.

        Returns:
            {"sentences": [...], "gaps": [{"gap", "score", "depth"}],
            "cuts": [...]}: a cut at gap i starts a topic with sentence i.
        """
        pieces = self.pieces(text) if text.strip() else []
        cuts, gaps = boundaries([content_words(piece) for piece in pieces], self.window)
        return {"sentences": pieces, "gaps": gaps, "cuts": snap_to_paragraphs(cuts, pieces)}

    def topics(self, pieces):
        """The piece indexes of each topic, e.g. [[0, 1, 2], [3, 4, 5, 6]]."""
        cuts, _ = boundaries([content_words(piece) for piece in pieces], self.window)
        edges = [0, *snap_to_paragraphs(cuts, pieces), len(pieces)]
        return [list(range(start, end)) for start, end in zip(edges, edges[1:]) if end > start]

    def split_spans(self, text):
        """(chunk, start, end) for every chunk; a chunk never holds two topics."""
        if not text.strip():
            return []

        pieces = self.pieces(text)
        groups = []
        for topic in self.topics(pieces):
            sizes = [self.length(pieces[i]) for i in topic]
            groups += [[topic[i] for i in group]
                       for group in pack(sizes, self.chunk_size, self.overlap)]

        spans = group_spans(text, pieces, groups)
        if spans is None:   # a tokenizer changed the text, so search for the chunks instead
            chunks = [chunk.strip() for chunk in join_groups(pieces, groups) if chunk.strip()]
            return located_spans(text, chunks)
        return spans
