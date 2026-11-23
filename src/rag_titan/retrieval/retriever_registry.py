"""
Pick the retriever and the reranker named in the settings.

Adding one: create its class, add its name to RETRIEVERS or RERANKERS and one
`if name == ...` line below.
"""

from rag_titan.retrieval.bm25_retriever import BM25Retriever
from rag_titan.retrieval.dense_retriever import DenseRetriever
from rag_titan.retrieval.hybrid_retriever import HybridRetriever
from rag_titan.retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
from rag_titan.retrieval.rerankers.keyword_reranker import KeywordReranker
from rag_titan.retrieval.rerankers.llm_reranker import LLMReranker
from rag_titan.retrieval.rerankers.mmr_reranker import MMRReranker
from rag_titan.retrieval.rerankers.no_reranker import NoReranker

# every name build_retriever() / build_reranker() knows, used by the CLI and errors
RETRIEVERS = ["dense", "bm25", "hybrid"]
RERANKERS = ["keyword", "none", "mmr", "llm", "cross_encoder"]


def build_retriever(settings, embedder, store):
    """
    Create the retriever named in settings["retriever"].

    "dense" finds chunks by meaning, "bm25" by exact words, "hybrid" uses both
    and fuses the rankings (RRF; HYBRID_DENSE_WEIGHT, HYBRID_BM25_WEIGHT and
    RRF_KEEP_TOP tune it). All of them search the same store.

    Args:
        settings: the dict from get_settings().
        embedder: the BaseEmbedder that indexed the store.
        store: the BaseStore holding the chunks.
    """
    name = _name(settings, "retriever")
    dedupe = settings["dedupe_overlap"]

    dense = DenseRetriever(embedder, store, dedupe_overlap=dedupe)
    if name == "dense":
        return dense
    if name == "bm25":
        return BM25Retriever(store, dedupe_overlap=dedupe)
    if name == "hybrid":
        # each search keeps near-copies; hybrid removes them once, after fusing
        return HybridRetriever(DenseRetriever(embedder, store, dedupe_overlap=None),
                               BM25Retriever(store, dedupe_overlap=None), dedupe_overlap=dedupe,
                               dense_weight=settings.get("hybrid_dense_weight", 1.0),
                               bm25_weight=settings.get("hybrid_bm25_weight", 1.0),
                               keep_top=settings.get("rrf_keep_top", False))

    raise ValueError(f"Unknown retriever: {settings['retriever']!r}. "
                     f"Use one of: {', '.join(RETRIEVERS)}")


def build_reranker(settings, llm=None, embedder=None):
    """
    Create the reranker named in settings["reranker"].

    "keyword" adds a bonus for shared words, "none" keeps the search order,
    "mmr" trades some relevance for variety, "llm" asks the chat model to rate
    the chunks, "cross_encoder" scores (question, chunk) pairs with a model
    (pip install "rag-titan[rerank]"; RERANK_MODEL picks it).

    Args:
        settings: the dict from get_settings().
        llm: the chat model, needed by "llm".
        embedder: needed by "mmr", which compares chunks with each other.
    """
    name = _name(settings, "reranker")

    if name == "keyword":
        return KeywordReranker()
    if name == "none":
        return NoReranker()
    if name == "mmr":
        if embedder is None:
            raise ValueError("The mmr reranker needs the embedder: build_reranker(settings, "
                             "embedder=...)")
        return MMRReranker(embedder)
    if name == "llm":
        if llm is None:
            raise ValueError("The llm reranker needs the chat model: build_reranker(settings, "
                             "llm=...)")
        return LLMReranker(llm)
    if name == "cross_encoder":
        return CrossEncoderReranker(settings["rerank_model"])

    raise ValueError(f"Unknown reranker: {settings['reranker']!r}. "
                     f"Use one of: {', '.join(RERANKERS)}")


def _name(settings, key):
    """The setting's value as a clean lowercase name ("Hybrid " -> "hybrid")."""
    return str(settings[key]).strip().lower()
