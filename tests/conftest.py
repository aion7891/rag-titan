"""
Shared test helpers.

FakeLLM and FakeEmbedder replace Ollama, so the tests run anywhere,
fast, and without downloading any model.
"""

import numpy as np
import pytest

from rag_titan.chunking import WordChunker
from rag_titan.embeddings import BaseEmbedder
from rag_titan.generation import GroundedGenerator
from rag_titan.llm import BaseLLM
from rag_titan.prompts import EVIDENCE_CHECK_PROMPT, ROUTER_PROMPT
from rag_titan.retrieval import Retriever
from rag_titan.retrieval.rerankers import KeywordReranker
from rag_titan.vector_stores import MemoryStore

# a few words used to build small, easy to understand vectors
VOCABULARY = ["leave", "days", "salary", "remote", "work", "office", "holiday"]


class FakeLLM(BaseLLM):
    """
    LLM that replies from a list of prepared answers, in order.
    Every message it receives is saved in `calls` so tests can check it.
    """

    def __init__(self, replies=None):
        super().__init__("fake-model")
        self.replies = list(replies or ["fake answer"])
        self.calls = []

    def chat(self, messages, tools=None):
        """Return the next prepared reply (the last one repeats)."""
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return {"content": reply, "tool_calls": [], "message": {"role": "assistant"}}

    def tool_result_message(self, tool_call, result):
        """Build a simple tool message."""
        return {"role": "tool", "content": str(result)}


class PromptLLM(BaseLLM):
    """Replies by prompt: route "simple", evidence check `enough`, else an answer."""

    def __init__(self, enough="yes"):
        super().__init__("prompt-fake")
        self.enough = enough

    def chat(self, messages, tools=None):
        system = messages[0]["content"]
        if system == ROUTER_PROMPT:
            reply = "simple"
        elif system == EVIDENCE_CHECK_PROMPT:
            reply = self.enough
        else:
            reply = "the answer"
        return {"content": reply, "tool_calls": [], "message": {"role": "assistant"}}


class FakeEmbedder(BaseEmbedder):
    """Counts how often each VOCABULARY word appears in the text."""

    def __init__(self):
        super().__init__("fake-embedder")

    def embed_many(self, texts):
        """Return one word-count vector per text."""
        vectors = []
        for text in texts:
            words = text.lower().split()
            vectors.append(np.array([words.count(word) for word in VOCABULARY], dtype=float))
        return vectors


def make_chunks():
    """Return three small chunks about different topics."""
    return [
        {"id": "chunk_0", "source": "hr.pdf", "text": "you get 20 leave days every year"},
        {"id": "chunk_1", "source": "hr.pdf", "text": "salary is paid every month"},
        {"id": "chunk_2", "source": "it.pdf", "text": "remote work needs a laptop"},
    ]


@pytest.fixture
def parts():
    """Build RAG parts with fakes and three indexed chunks."""
    llm = FakeLLM()
    embedder = FakeEmbedder()
    store = MemoryStore()

    chunks = make_chunks()
    store.add(chunks, embedder.embed_many([chunk["text"] for chunk in chunks]))

    return {
        "llm": llm,
        "embedder": embedder,
        "store": store,
        "chunker": WordChunker(chunk_size=5, overlap=1),
        "retriever": Retriever(embedder, store),
        "reranker": KeywordReranker(),
        "generator": GroundedGenerator(llm),
    }
