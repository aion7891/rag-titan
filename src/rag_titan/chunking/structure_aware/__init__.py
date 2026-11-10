"""
Structure-aware chunkers (handbook section 3): keep the document's own units
(sentences, paragraphs, ...) whole, and pack them up to a size limit.
"""

from rag_titan.chunking.structure_aware.base import PieceChunker
from rag_titan.chunking.structure_aware.paragraph_chunker import ParagraphChunker
from rag_titan.chunking.structure_aware.sentence_chunker import SentenceChunker
from rag_titan.chunking.structure_aware.sentences import split_paragraphs, split_sentences

__all__ = [
    "ParagraphChunker",
    "PieceChunker",
    "SentenceChunker",
    "split_paragraphs",
    "split_sentences",
]
