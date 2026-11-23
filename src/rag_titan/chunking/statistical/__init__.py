"""
Statistical chunking (handbook section 5): cut where the words of the text
change, without a model. TextTiling (Hearst 1997) for now.
"""

from rag_titan.chunking.statistical.cohesion import boundaries, content_words
from rag_titan.chunking.statistical.texttiling import TextTilingChunker

__all__ = ["TextTilingChunker", "boundaries", "content_words"]
