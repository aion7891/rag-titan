"""
Parent-child chunking (RAG handbook 2026, section 09: "store parent/child IDs and
both granularities").

Every document is cut twice:
    parents:  heading sections (paragraph groups when there are no headings),
              at most parent_size characters; what the model reads
    children: your normal chunker run inside each parent; what gets searched

Small children match a question precisely; their parent gives the model the
whole section around the match. Each child carries its parent:

    {"id": "chunk_4", "source": "leave.md", "text": "Sick leave needs a note ...",
     "parent_id": "leave.md#parent_1",
     "parent_text": "Leave policy > Sick leave\n\nTell your manager ... a note ...",
     "parent_start_offset": 120, "parent_end_offset": 410}

Stores save the whole chunk, so the parent is saved (and replaced by
`index --add`) together with its children, in every store.
"""

from rag_titan.chunking.base_chunker import BaseChunker, document_metadata
from rag_titan.chunking.locations import Locations
from rag_titan.chunking.offsets import shift
from rag_titan.chunking.structure_aware.heading_chunker import HeadingChunker

# about 250 words: a whole section of a typical policy or manual
DEFAULT_PARENT_SIZE = 1500


class ParentChildChunker(BaseChunker):
    """
    Small chunks for search, each labelled with the larger section it sits in.

    Args:
        child_chunker: the BaseChunker that makes the children (for example a
            WordChunker with 40 words). It runs inside each parent, so a child
            never mixes two sections.
        parent_size: largest parent in characters, heading path included.
    """

    def __init__(self, child_chunker, parent_size=DEFAULT_PARENT_SIZE):
        if not isinstance(child_chunker, BaseChunker):
            raise ValueError("child_chunker must be a chunker, for example WordChunker(40, 10)")
        if isinstance(child_chunker, ParentChildChunker):
            raise ValueError("child_chunker cannot be another ParentChildChunker")
        # bool is an int in Python, but parent_size=True is surely a mistake
        if isinstance(parent_size, bool) or not isinstance(parent_size, int) or parent_size <= 0:
            raise ValueError(f"parent_size must be a whole number of characters above 0, "
                             f"got {parent_size!r}")

        self.child_chunker = child_chunker
        self.parent_size = parent_size
        # parents keep their heading path ("Leave policy > Sick leave"), so the
        # model knows which section it is reading
        self.parent_chunker = HeadingChunker(parent_size, 0, include_path=True)

    def chunk(self, documents):
        """
        Turn documents into children that know their parent.

        Args:
            documents: list of {"source": ..., "text": ...} dicts.

        Returns:
            The child chunks, like BaseChunker.chunk() returns them, plus the
            parent fields (see describe_parent). Offsets point into the document.
        """
        chunks = []

        for document in documents:
            text, source = document["text"], document["source"]
            locations = Locations(text)
            parents = self.parent_chunker.split_spans(text)

            for number, (parent_text, start, end) in enumerate(parents):
                # the file name keeps ids unique when two files share a section title
                parent = self.describe_parent(source, number, parent_text, start, end, locations)
                for child, first, last in self.child_spans(text, start, end, parent_text):
                    chunks.append({
                        "id": f"chunk_{len(chunks)}",
                        "source": source,
                        "text": child,
                        "start_offset": first,
                        "end_offset": last,
                        **locations.describe(first, last),
                        **parent,
                        **document_metadata(document),
                    })

        return chunks

    def describe_parent(self, source, number, parent_text, start, end, locations):
        """
        The parent fields every child of this parent carries.

        Returns:
            {"parent_id": "leave.md#parent_1", "parent_text": ...,
            "parent_start_offset": 120, "parent_end_offset": 410}, plus
            "parent_page_numbers" for documents with pages. The RAG type uses
            them to hand the model the parent, with its own place in the file.
        """
        # the file name keeps ids unique when two files share a section title
        parent = {"parent_id": f"{source}#parent_{number}", "parent_text": parent_text,
                  "parent_start_offset": start, "parent_end_offset": end}
        pages = locations.describe(start, end).get("page_numbers")
        if pages:
            parent["parent_page_numbers"] = pages
        return parent

    def child_spans(self, text, start, end, parent_text):
        """
        Cut one parent into children: (child text, start, end) in the document.

        The child chunker works on the parent's own text (without the heading
        path), so its positions start at 0; shift() moves them by the parent's
        start to point into the whole document again.
        """
        if start is None or end is None:
            # the parent's place is unknown: children still work, without offsets
            return [(child, None, None) for child in self.child_chunker.split_text(parent_text)]
        return shift(self.child_chunker.split_spans(text[start:end]), start)

    def split_text(self, text):
        """Return only the children's texts (what gets embedded and searched)."""
        return [chunk["text"] for chunk in self.chunk([{"source": "text", "text": text}])]
