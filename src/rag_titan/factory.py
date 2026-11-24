"""
The only place that knows which class to use for each setting.

Adding a new provider or strategy:
    1. create its class in the right folder
    2. add one `if name == ...` line in the matching build_* function
    3. change the setting in .env (or pass it in `get_settings()`)
"""

from rag_titan.chunking.fixed_size import (
    CharacterChunker,
    RecursiveChunker,
    TokenChunker,
    WordChunker,
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
from rag_titan.embeddings.hash_embedder import HashEmbedder
from rag_titan.embeddings.ollama_embedder import OllamaEmbedder
from rag_titan.embeddings.openai_compatible_embedder import OpenAICompatibleEmbedder
from rag_titan.generation.grounded_generator import GroundedGenerator
from rag_titan.llm.anthropic_llm import AnthropicLLM
from rag_titan.llm.ollama_llm import OllamaLLM
from rag_titan.llm.openai_compatible_llm import OpenAICompatibleLLM
from rag_titan.pipelines.indexing_pipeline import IndexingPipeline
from rag_titan.providers import OPENAI_COMPATIBLE, PROVIDERS, connection, provider_name
from rag_titan.rag_types.rag_registry import (  # noqa: F401  (build_rag: old import path)
    PARENT_RAG_TYPES,
    RAG_TYPES,
    build_rag,
)
from rag_titan.retrieval.retriever_registry import (  # noqa: F401  (build_reranker: old import path)
    RERANKERS,
    RETRIEVERS,
    build_reranker,
    build_retriever,
)
from rag_titan.vector_stores.store_registry import VECTOR_STORES, build_store  # noqa: F401

# every LLM provider build_llm() knows ("claude" also works for "anthropic")
LLM_PROVIDERS = list(PROVIDERS)
EMBED_PROVIDERS = ["ollama", "hash"] + [name for name in OPENAI_COMPATIBLE
                                        if not PROVIDERS[name].get("no_embeddings")]

# every chunker build_chunker() knows, used by the CLI
CHUNKERS = ["word", "character", "token", "recursive", "sentence", "paragraph", "heading", "page",
            "markdown", "texttiling"]


def build_llm(settings, model=None, think=False):
    """
    Create the chat model named in settings["llm_provider"].

    "ollama" runs models locally; "anthropic" (or "claude") uses Claude and
    reads the key from the ANTHROPIC_API_KEY environment variable. Every other
    provider (openai, gemini, openrouter, huggingface, lmstudio, vllm,
    openai_compatible) uses the OpenAI-compatible client; see providers.py.
    """
    name = provider_name(settings["llm_provider"])
    model = model or settings["llm_model"]

    if name == "ollama":
        return OllamaLLM(model, host=settings["ollama_host"], think=think)

    if name == "anthropic":
        return AnthropicLLM(model)

    if name in OPENAI_COMPATIBLE:
        return OpenAICompatibleLLM(model, **connection(name, settings, "llm"))

    raise ValueError(f"Unknown LLM provider: {settings['llm_provider']!r}. "
                     f"Use one of: {', '.join(LLM_PROVIDERS)}")


def build_agent_llm(settings):
    """Create the chat model used by agents (it must support tools)."""
    return build_llm(settings, model=settings["agent_model"], think=settings["agent_think"])


def build_embedder(settings):
    """
    Create the embedding model named in settings["embed_provider"]: "ollama",
    "hash" (no model, for tests), or an OpenAI-compatible provider.
    """
    name = provider_name(settings["embed_provider"])

    if name == "ollama":
        return OllamaEmbedder(settings["embed_model"], host=settings["ollama_host"])

    if name == "hash":
        return HashEmbedder()

    if PROVIDERS.get(name, {}).get("no_embeddings") or name == "anthropic":
        raise ValueError(f"{PROVIDERS[name]['service']} has no OpenAI-style embeddings "
                         f"endpoint: use another EMBED_PROVIDER ({', '.join(EMBED_PROVIDERS)})")
    if name in OPENAI_COMPATIBLE:
        return OpenAICompatibleEmbedder(settings["embed_model"],
                                        **connection(name, settings, "embed"))

    raise ValueError(f"Unknown embedding provider: {settings['embed_provider']!r}. "
                     f"Use one of: {', '.join(EMBED_PROVIDERS)}")


def build_chunker(settings):
    """
    Create the chunker named in settings["chunker"].
    chunk_size and chunk_overlap count words, characters or tokens,
    depending on the chunker.
    """
    name = settings["chunker"]
    size, overlap = settings["chunk_size"], settings["chunk_overlap"]

    if name == "word":
        return WordChunker(size, overlap)

    if name == "character":
        return CharacterChunker(size, overlap)

    if name == "token":
        return TokenChunker(size, overlap, encoding=settings["chunk_encoding"])

    if name == "recursive":
        return RecursiveChunker(size, overlap)

    # structure-aware chunkers: whole sentences, paragraphs, sections or pages
    structure_aware = {
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "heading": HeadingChunker,
        "page": PageChunker,
        # whole Markdown blocks: lists, tables, code, quotes (chunking/markdown/)
        "markdown": MarkdownChunker,
    }
    if name in structure_aware:
        return structure_aware[name](size, overlap)

    if name == "texttiling":
        # topic boundaries from word overlap (chunking/statistical/)
        return TextTilingChunker(size, overlap, window=settings.get("texttiling_window", 3))

    raise ValueError(f"Unknown chunker: {name}")


def build_generator(settings, llm):
    """Create the answer generator named in settings["generator"]."""
    name = settings["generator"]

    if name == "grounded":
        return GroundedGenerator(llm)

    raise ValueError(f"Unknown generator: {name}")


def build_rag_parts(settings, rag_type=None):
    """
    Build every RAG building block, sharing one embedder and one store.

    Args:
        settings: the dict from get_settings().
        rag_type: the RAG type the parts are for. "parent_child" indexes
            small children inside parent sections (PARENT_SIZE); others, or
            None, index with the CHUNKER alone.

    Returns:
        A dict with "llm", "pipeline", "retriever", "reranker", "generator"
        and "settings" (build_rag() reads the tuning options from it).
    """
    llm = build_llm(settings)
    embedder = build_embedder(settings)
    store = build_store(settings)

    chunker = build_chunker(settings)
    if rag_type in PARENT_RAG_TYPES:
        chunker = ParentChildChunker(chunker, settings["parent_size"])

    return {
        "llm": llm,
        "pipeline": IndexingPipeline(chunker, embedder, store),
        "retriever": build_retriever(settings, embedder, store),
        "reranker": build_reranker(settings, llm, embedder),
        "generator": build_generator(settings, llm),
        "settings": settings,
    }
