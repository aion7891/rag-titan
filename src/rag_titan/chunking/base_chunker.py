"""
Base class for every chunking strategy.

To add a new strategy (sentence, semantic, recursive, ...), inherit from
BaseChunker and implement split_text(). The id/source/offset handling is shared.
"""

from rag_titan.chunking.locations import Locations
from rag_titan.chunking.offsets import located_spans


def document_metadata(document):
    """
    The document's own metadata for each of its chunks: {"metadata": {...}}, or
    {} when it has none (so plain documents give exactly the same chunks).

    A loader can attach facts every chunk should carry, for example an OKF
    concept's id and trust tier. Each chunk gets its own copy.
    """
    metadata = document.get("metadata")
    return {"metadata": dict(metadata)} if metadata else {}


class BaseChunker:
    """Splits documents into smaller pieces called chunks."""

    def chunk(self, documents):
        """
        Turn a list of document dicts into a list of chunk dicts.

        Args:
            documents: list of {"source": ..., "text": ...} dicts.

        Returns:
            List of {"id": "chunk_0", "source": ..., "text": ...,
            "start_offset": 0, "end_offset": 120} dicts.
            `document["text"][start_offset:end_offset]` is where the chunk comes
            from (the offsets are None if it cannot be found).
            When known, chunks also get "page_numbers" (PDFs) and
            "section_path" (documents with headings), and the document's own
            "metadata" when it has one.
        """
        chunks = []

        for document in documents:
            locations = Locations(document["text"])
            for text, start, end in self.split_spans(document["text"]):
                chunks.append({
                    "id": f"chunk_{len(chunks)}",
                    "source": document["source"],
                    "text": text,
                    "start_offset": start,
                    "end_offset": end,
                    **locations.describe(start, end),
                    **document_metadata(document),
                })

        return chunks

    def split_spans(self, text):
        """
        Return (chunk_text, start, end) for every chunk of `text`.

        This default searches for each chunk from split_text() in the text.
        Chunkers that know their exact positions override it.
        """
        return located_spans(text, self.split_text(text))

    def split_text(self, text):
        """Return a list of text pieces."""
        raise NotImplementedError
