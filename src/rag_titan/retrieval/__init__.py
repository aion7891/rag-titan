"""Retrieval: find similar chunks, then rerank them."""

from rag_titan.retrieval.dedupe import overlap_fraction, remove_duplicates
from rag_titan.retrieval.retriever import Retriever

__all__ = ["Retriever", "overlap_fraction", "remove_duplicates"]
