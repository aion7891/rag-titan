"""
Playground presets: the parts every playground run is built from.

The recorder (recording.py), the tests and the browser all build their RAG
types with build_parts(), so a replay is the same code as the recording.
"""

from rag_titan.chunking.parent_child import ParentChildChunker
from rag_titan.config import get_settings
from rag_titan.embeddings.replay_embedder import ReplayEmbedder
from rag_titan.factory import build_chunker, build_generator, build_reranker, build_retriever
from rag_titan.llm.replay_llm import ReplayLLM
from rag_titan.playground.tuning import RETRIEVER_DEFAULTS
from rag_titan.vector_stores.memory_store import MemoryStore

# the ingestion choices offered in the playground
PRESET_CHUNKING = {
    "word": {"chunker": "word", "chunk_size": 40, "chunk_overlap": 20},
    "sentence": {"chunker": "sentence", "chunk_size": 300, "chunk_overlap": 0},
    "paragraph": {"chunker": "paragraph", "chunk_size": 300, "chunk_overlap": 0},
}

# the only settings a preset needs; saved with it so every machine uses the same ones
PRESET_KEYS = ["chunker", "chunk_size", "chunk_overlap", "chunk_encoding",
               "dedupe_overlap", "reranker", "generator"]


def preset_settings(overrides):
    """The settings of one preset: the defaults plus `overrides`, only PRESET_KEYS."""
    settings = get_settings(overrides)
    return {key: settings[key] for key in PRESET_KEYS}


def build_parts(settings, embedder, llm, retriever="dense", parent_size=None):
    """
    Build every part of a RAG from preset settings, a model and an embedder.

    Args:
        parent_size: for Parent-Child RAG, the largest parent in characters;
            the chunker then makes children inside parents. None: plain chunks.
    """
    store = MemoryStore()
    chunker = build_chunker(settings)
    if parent_size:
        chunker = ParentChildChunker(chunker, parent_size)
    parts = {
        "llm": llm,
        "chunker": chunker,
        "embedder": embedder,
        "store": store,
        "dedupe_overlap": settings["dedupe_overlap"],
        "reranker": build_reranker(settings, llm, embedder),
        "generator": build_generator(settings, llm),
    }
    return with_retriever(parts, retriever)


def with_retriever(parts, retriever, tuning=None):
    """
    The same parts (and index) searched with another retriever: dense, bm25 or hybrid.

    Args:
        tuning: optional knobs of the hybrid search (hybrid_dense_weight,
            hybrid_bm25_weight, rrf_keep_top); the others are ignored here.
    """
    settings = {"retriever": retriever, "dedupe_overlap": parts["dedupe_overlap"]}
    settings.update({key: value for key, value in (tuning or {}).items()
                     if key in RETRIEVER_DEFAULTS})
    return {**parts, "retriever": build_retriever(settings, parts["embedder"], parts["store"])}


def index_documents(parts, documents):
    """Chunk, embed and store documents given as {"source", "text"} dicts (no files needed)."""
    chunks = parts["chunker"].chunk(documents)
    vectors = parts["embedder"].embed_many([chunk["text"] for chunk in chunks])

    parts["store"].clear()
    parts["store"].add(chunks, vectors)
    return len(chunks)


def replay_parts(presets, name, retriever="dense", parent_size=None):
    """
    Build and index the parts of one preset chunking, answering from the recording.

    Args:
        parent_size: index with parents of this size (Parent-Child RAG), or None.
    """
    llm = ReplayLLM(presets["replies"], model=presets["model"])
    embedder = ReplayEmbedder(presets["vectors"], model=presets["embed_model"])

    parts = build_parts(presets["settings"][name], embedder, llm, retriever, parent_size)
    index_documents(parts, presets["documents"])
    return parts
