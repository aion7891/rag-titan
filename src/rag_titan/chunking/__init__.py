"""
Chunkers that split documents into small pieces.

Each family of strategies lives in its own sub-folder:
    fixed_size/   characters, words, tokens, recursive separators (handbook section 2)

Everything is also exported here, so `from rag_titan.chunking import WordChunker` works.
"""

from rag_titan.chunking.base_chunker import BaseChunker
from rag_titan.chunking.fixed_size import (
    CharacterChunker,
    RecursiveChunker,
    TokenChunker,
    WordChunker,
    fixed_windows,
)

__all__ = [
    "BaseChunker",
    "CharacterChunker",
    "RecursiveChunker",
    "TokenChunker",
    "WordChunker",
    "fixed_windows",
]
