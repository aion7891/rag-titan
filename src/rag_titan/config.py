"""
All settings in one place.

Values are read from environment variables (or a `.env` file in the
folder where you run your code). Every setting has a safe default, so
the package works without any configuration.

Example:
    from rag_titan.config import get_settings

    settings = get_settings({"llm_model": "llama3.2", "chunk_size": 200})
"""

import os

try:
    from dotenv import load_dotenv
except ImportError:
    # python-dotenv is missing in the browser (Pyodide): settings then come only
    # from environment variables and overrides
    load_dotenv = None

# chat and agent models used when LLM_MODEL / AGENT_MODEL are not set,
# so switching LLM_PROVIDER alone is enough
DEFAULT_MODELS = {
    "ollama": {"llm_model": "gemma3:4b", "agent_model": "qwen3:4b"},
    "anthropic": {"llm_model": "claude-haiku-5-5", "agent_model": "claude-haiku-5-5"},
}
PROVIDER_ALIASES = {"claude": "anthropic"}

# where a saved index lives when VECTOR_STORE_PATH is not set: a file or a folder
DEFAULT_STORE_PATHS = {"sqlite": "rag_titan.db", "chroma": "rag_titan_chroma",
                       "qdrant": "rag_titan_qdrant"}


def provider_name(name):
    """Turn "Claude", " anthropic " and the like into the provider's real name."""
    name = str(name).strip().lower()
    return PROVIDER_ALIASES.get(name, name)


def _get_int(name, default):
    """Read a whole number from the environment, or use the default."""
    value = os.getenv(name, str(default))

    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Setting {name} must be a whole number, got '{value}'")


def _get_optional_int(name):
    """Read a whole number from the environment; missing or empty gives None."""
    value = os.getenv(name, "").strip()
    return _get_int(name, None) if value else None


def _get_bool(name, default):
    """Read a yes/no value from the environment, or use the default."""
    value = os.getenv(name, str(default)).strip().lower()
    return value in ("1", "true", "yes", "on")


def _get_overlap(name, default):
    """Read a share between 0 and 1 (like 0.5), or "off" for None."""
    value = os.getenv(name, str(default)).strip().lower()
    if value in ("off", "none", "false", "no"):
        return None

    try:
        share = float(value)
    except ValueError:
        raise ValueError(f"Setting {name} must be a number like 0.5, or off; got '{value}'")
    if not 0 < share <= 1:
        raise ValueError(f"Setting {name} must be more than 0 and at most 1, or off")
    return share


def get_settings(overrides=None):
    """
    Build the settings dictionary.

    Args:
        overrides: optional dict of settings that win over the environment,
            for example {"chunk_size": 200}.

    Returns:
        A dict with every setting the package uses.
    """
    if load_dotenv is not None:
        load_dotenv()           # read .env only when settings are requested, not on import

    settings = {
        # LLM used by RAG
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "llm_model": os.getenv("LLM_MODEL"),          # None: the provider's default
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),

        # LLM used by agents (needs tool-calling support)
        "agent_model": os.getenv("AGENT_MODEL"),      # None: the provider's default
        "agent_think": _get_bool("AGENT_THINK", True),
        "agent_max_steps": _get_int("AGENT_MAX_STEPS", 5),

        # Embeddings
        "embed_provider": os.getenv("EMBED_PROVIDER", "ollama"),
        "embed_model": os.getenv("EMBED_MODEL", "nomic-embed-text"),

        # Chunking
        **_chunk_settings(),

        # Storage, retrieval and generation
        **_store_settings(),
        # drop a search result that overlaps a better one by this share ("off" keeps all)
        "dedupe_overlap": _get_overlap("DEDUPE_OVERLAP", 0.5),
        "reranker": os.getenv("RERANKER", "keyword"),
        "rerank_model": os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        **_tuning_settings(),
        "generator": os.getenv("GENERATOR", "grounded"),
    }

    if overrides:
        settings.update(overrides)

    # fill these last, so an overridden provider or store gets its own defaults
    _fill_defaults(settings)
    return settings


def _chunk_settings():
    """Read the chunking settings from the environment."""
    return {
        "chunker": os.getenv("CHUNKER", "word"),
        "chunk_size": _get_int("CHUNK_SIZE", 150),
        "chunk_overlap": _get_int("CHUNK_OVERLAP", 30),
        "chunk_encoding": os.getenv("CHUNK_ENCODING", "cl100k_base"),  # token chunker only
        # parent_child RAG only: largest parent section, in characters
        "parent_size": _get_int("PARENT_SIZE", 1500),
    }


def _tuning_settings():
    """Read how much each RAG type searches and keeps (see rag_types/rag_registry.py)."""
    return {
        # empty: each RAG type's own default (simple 5/2, adaptive 4/2, parent_child 8/4)
        "top_k": _get_optional_int("TOP_K"),      # chunks searched
        "top_n": _get_optional_int("TOP_N"),      # chunks kept after reranking
        # adaptive: searches before giving up
        "adaptive_max_attempts": _get_int("ADAPTIVE_MAX_ATTEMPTS", 2),
        # parent_child: most characters of parent sections given to the model
        "parent_budget": _get_int("PARENT_BUDGET", 3000),
        # hyde: also search with the question and fuse it with the made-up answer's search
        "hyde_with_question": _get_bool("HYDE_WITH_QUESTION", True),
        # rag_fusion: rewordings of the question to search with (1 to 10)
        "fusion_queries": _get_int("FUSION_QUERIES", 3),
        # corrective: grades (0-10) that make the evidence good / bad, keep only useful
        # sentences, and how many times to search again
        "crag_upper": _get_int("CRAG_UPPER", 7),
        "crag_lower": _get_int("CRAG_LOWER", 3),
        "crag_refine": _get_bool("CRAG_REFINE", True),
        "crag_max_retries": _get_int("CRAG_MAX_RETRIES", 1),
        # iterative: think-then-search rounds after the first search
        "iterative_max_steps": _get_int("ITERATIVE_MAX_STEPS", 3),
    }


def _store_settings():
    """Read the vector store and retrieval settings from the environment."""
    return {
        "retriever": os.getenv("RETRIEVER", "dense"),         # dense, bm25 or hybrid
        "vector_store": os.getenv("VECTOR_STORE", "memory"),
        "vector_store_path": os.getenv("VECTOR_STORE_PATH"),  # None: the store's default
        "vector_store_url": os.getenv("VECTOR_STORE_URL"),    # chroma/qdrant server
        "collection": os.getenv("COLLECTION", "rag_titan"),   # chroma/qdrant
    }


def _fill_defaults(settings):
    """Fill empty models and store path with the provider's and the store's defaults."""
    defaults = DEFAULT_MODELS.get(provider_name(settings["llm_provider"]), DEFAULT_MODELS["ollama"])
    for key, model in defaults.items():
        if not settings.get(key):
            settings[key] = model

    store = str(settings["vector_store"]).strip().lower()
    if not settings.get("vector_store_path"):
        settings["vector_store_path"] = DEFAULT_STORE_PATHS.get(store)
