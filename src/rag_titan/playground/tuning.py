"""
The tuning knobs the playground pages offer, checked before a run.

    check_tuning({"top_k": 8, "reranker": "mmr"}, parent_sizes=[1500])
    default_tuning()["parent_child"]   # {"top_k": 8, "top_n": 4, "parent_budget": 3000, ...}

The names are the setting names (see rag_types/rag_registry.py), so a page
setting and a .env setting mean the same thing.
"""

from rag_titan.chunking.parent_child import DEFAULT_PARENT_SIZE
from rag_titan.rag_types.rag_registry import CLASSES

# rerankers that run in the browser: cross_encoder needs torch, which Pyodide cannot load
BROWSER_RERANKERS = ["keyword", "none", "mmr", "llm"]

# every knob a page may send
TUNING_KEYS = ["reranker", "top_k", "top_n", "adaptive_max_attempts", "parent_budget",
               "parent_size", "hyde_with_question", "fusion_queries", "crag_upper", "crag_lower",
               "crag_refine", "crag_max_retries", "iterative_max_steps"]


def default_tuning():
    """Each RAG type's knobs with their default values, for the page's controls."""
    defaults = {name: {"top_k": cls.DEFAULT_TOP_K, "top_n": cls.DEFAULT_TOP_N}
                for name, cls in CLASSES.items()}
    defaults["adaptive"]["adaptive_max_attempts"] = CLASSES["adaptive"].DEFAULT_MAX_ATTEMPTS
    defaults["parent_child"]["parent_budget"] = CLASSES["parent_child"].DEFAULT_PARENT_BUDGET
    defaults["parent_child"]["parent_size"] = DEFAULT_PARENT_SIZE
    defaults["hyde"]["hyde_with_question"] = CLASSES["hyde"].DEFAULT_WITH_QUESTION
    defaults["rag_fusion"]["fusion_queries"] = CLASSES["rag_fusion"].DEFAULT_QUERIES
    corrective = CLASSES["corrective"]
    defaults["corrective"].update(crag_upper=corrective.DEFAULT_UPPER,
                                  crag_lower=corrective.DEFAULT_LOWER,
                                  crag_refine=corrective.DEFAULT_REFINE,
                                  crag_max_retries=corrective.DEFAULT_MAX_RETRIES)
    defaults["iterative"]["iterative_max_steps"] = CLASSES["iterative"].DEFAULT_MAX_STEPS
    return defaults


def check_tuning(tuning, parent_sizes):
    """
    Return the tuning as a dict, or raise a clear ValueError.

    Args:
        tuning: None, or a dict with some of TUNING_KEYS. Numbers are checked
            by the RAG type itself, with the same messages as in Python.
        parent_sizes: the recorded parent sizes; only those have vectors.
    """
    if tuning is None:
        return {}
    if not isinstance(tuning, dict):
        raise ValueError("tuning must be a dict like {'top_k': 8}")

    unknown = sorted(set(tuning) - set(TUNING_KEYS))
    if unknown:
        raise ValueError(f"Unknown tuning {unknown}: use {', '.join(TUNING_KEYS)}")

    reranker = tuning.get("reranker")
    if reranker == "cross_encoder":
        raise ValueError("The cross_encoder reranker needs torch, which cannot run in the "
                         "browser: try it in Python with pip install \"rag-titan[rerank]\"")
    if reranker is not None and reranker not in BROWSER_RERANKERS:
        raise ValueError(f"Unknown reranker {reranker!r}: use {', '.join(BROWSER_RERANKERS)}")

    size = tuning.get("parent_size")
    if size is not None and size not in parent_sizes:
        raise ValueError(f"parent_size {size} was not recorded: use one of {parent_sizes}")
    return {key: value for key, value in tuning.items() if value is not None}
