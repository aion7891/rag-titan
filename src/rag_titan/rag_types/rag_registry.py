"""
Pick the RAG type by name and give it its tuning settings.

Adding one: create its class, add its name to RAG_TYPES, one `if` in
build_rag() and its own options in rag_options().
"""

from rag_titan.rag_types.adaptive.adaptive_rag import AdaptiveRAG
from rag_titan.rag_types.corrective.corrective_rag import CorrectiveRAG
from rag_titan.rag_types.hyde_rag import HyDERAG
from rag_titan.rag_types.iterative.iterative_rag import IterativeRAG
from rag_titan.rag_types.parent_child_rag import ParentChildRAG
from rag_titan.rag_types.rag_fusion_rag import RAGFusionRAG
from rag_titan.rag_types.simple_rag import SimpleRAG

# every RAG type the package can build, used by build_rag() and the CLI
RAG_TYPES = ["simple", "adaptive", "parent_child", "hyde", "rag_fusion", "corrective",
             "iterative"]

# RAG types that need an index built with parent sections (see chunking/parent_child.py)
PARENT_RAG_TYPES = ["parent_child"]

# the class of each RAG type (its DEFAULT_TOP_K / DEFAULT_TOP_N are the defaults)
CLASSES = {"simple": SimpleRAG, "adaptive": AdaptiveRAG, "parent_child": ParentChildRAG,
           "hyde": HyDERAG, "rag_fusion": RAGFusionRAG, "corrective": CorrectiveRAG,
           "iterative": IterativeRAG}

# each type's own options, and the setting each one reads
TYPE_OPTIONS = {
    "adaptive": {"max_attempts": "adaptive_max_attempts"},
    "parent_child": {"parent_budget": "parent_budget"},
    "hyde": {"with_question": "hyde_with_question"},
    "rag_fusion": {"queries": "fusion_queries"},
    "corrective": {"upper": "crag_upper", "lower": "crag_lower", "refine": "crag_refine",
                   "max_retries": "crag_max_retries"},
    "iterative": {"max_steps": "iterative_max_steps"},
}


def rag_options(rag_type, settings):
    """
    The tuning options of one RAG type, read from the settings.

    TOP_K and TOP_N work for every type; left empty (None) each type keeps its
    own default (simple 5/2, adaptive 4/2, parent_child 8/4, hyde, rag_fusion and
    corrective 5/2, iterative 3/2 per search).

    Args:
        rag_type: one of RAG_TYPES.
        settings: the dict from get_settings(), or None for every default.

    Returns:
        Keyword arguments for the type's class, e.g. {"top_k": 8, "max_attempts": 3}.
    """
    if not settings:
        return {}

    options = {key: settings[key] for key in ("top_k", "top_n")
               if settings.get(key) is not None}
    if "top_k" in options and "top_n" not in options and rag_type in CLASSES:
        # only TOP_K given: keep the type's top_n, but never more than top_k
        options["top_n"] = min(CLASSES[rag_type].DEFAULT_TOP_N, options["top_k"])
    for option, key in TYPE_OPTIONS.get(rag_type, {}).items():
        if settings.get(key) is not None:
            options[option] = settings[key]
    return options


def build_rag(rag_type, parts):
    """
    Create one RAG type from the parts made by build_rag_parts().

    Args:
        rag_type: one of RAG_TYPES, for example "simple" or "adaptive".
        parts: the dict returned by build_rag_parts(); for "parent_child" it
            must be built with build_rag_parts(settings, "parent_child").
            Its "settings" (when there) give the tuning options.
    """
    options = rag_options(rag_type, parts.get("settings"))

    if rag_type == "simple":
        return SimpleRAG(parts["retriever"], parts["reranker"], parts["generator"], **options)

    if rag_type == "adaptive":
        return AdaptiveRAG(parts["llm"], parts["retriever"], parts["reranker"],
                           parts["generator"], **options)

    if rag_type == "parent_child":
        return ParentChildRAG(parts["retriever"], parts["reranker"], parts["generator"],
                              **options)

    if rag_type == "hyde":
        return HyDERAG(parts["llm"], parts["retriever"], parts["reranker"], parts["generator"],
                       **options)

    if rag_type == "rag_fusion":
        return RAGFusionRAG(parts["llm"], parts["retriever"], parts["reranker"],
                            parts["generator"], **options)

    if rag_type == "corrective":
        return CorrectiveRAG(parts["llm"], parts["retriever"], parts["reranker"],
                             parts["generator"], **options)

    if rag_type == "iterative":
        return IterativeRAG(parts["llm"], parts["retriever"], parts["reranker"],
                            parts["generator"], **options)

    raise ValueError(f"Unknown RAG type: {rag_type}. Choose from {RAG_TYPES}")
