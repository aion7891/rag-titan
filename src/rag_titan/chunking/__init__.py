"""
Chunkers that split documents into small pieces.

Each family of strategies lives in its own sub-folder:
    fixed_size/        characters, words, tokens, recursive separators (handbook section 2)
    structure_aware/   sentences, paragraphs, headings, pages (handbook section 3)
    markdown/          whole Markdown blocks: lists, tables, code (handbook 3.5, 3.7)
    statistical/       topic boundaries from word overlap: TextTiling (handbook 5.1)
    parent_child.py    small children for search inside larger parent sections

Everything is also exported here, so `from rag_titan.chunking import WordChunker` works.
"""

from rag_titan.chunking.base_chunker import BaseChunker
from rag_titan.chunking.fixed_size import (
    CharacterChunker,
    RecursiveChunker,
    TokenChunker,
    WordChunker,
    count_windows,
    fixed_windows,
)
from rag_titan.chunking.markdown import MarkdownChunker
from rag_titan.chunking.parent_child import ParentChildChunker
from rag_titan.chunking.statistical import TextTilingChunker
from rag_titan.chunking.structure_aware import (
    HeadingChunker,
    PageChunker,
    ParagraphChunker,
    SentenceChunker,
)

__all__ = [
    "BaseChunker",
    "CharacterChunker",
    "HeadingChunker",
    "MarkdownChunker",
    "PageChunker",
    "ParagraphChunker",
    "ParentChildChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "TextTilingChunker",
    "TokenChunker",
    "WordChunker",
    "count_windows",
    "fixed_windows",
]
