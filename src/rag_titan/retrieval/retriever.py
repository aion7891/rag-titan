"""
First step of retrieval: turn the question into a vector
and find the most similar chunks in the vector store.
"""

from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Finds chunks that are similar to a question.

    Args:
        embedder: a BaseEmbedder (must be the same one used for indexing).
        store: a BaseStore that holds the indexed chunks.
    """

    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def has_documents(self):
        """Return True when the store holds at least one chunk."""
        return self.store.count() > 0

    def retrieve(self, query, top_k=5):
        """
        Return the top_k most similar chunks, best first.

        Each result is a chunk dict with an extra "score" key.
        """
        if not self.has_documents():
            logger.warning("Vector store is empty, nothing to retrieve")
            return []

        query_vector = self.embedder.embed(query)
        matches = self.store.search(query_vector, top_k=top_k)

        for match in matches:
            logger.debug("Retrieved %s (score %.4f)", match["id"], match["score"])

        return matches
