"""
Element-aware Markdown chunking (handbook sections 3.5 and 3.7): keep lists,
tables, code blocks and callouts whole, and pack them inside heading sections.
"""

from rag_titan.chunking.markdown.blocks import glue_lead_ins, split_blocks
from rag_titan.chunking.markdown.markdown_chunker import MarkdownChunker

__all__ = ["MarkdownChunker", "glue_lead_ins", "split_blocks"]
