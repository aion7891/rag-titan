"""
rag-titan: a Swiss Army knife for Retrieval-Augmented Generation (RAG).

Quick start:
    from rag_titan import RAGTitan

    rag = RAGTitan(rag_type="simple")
    rag.index(["handbook.pdf"])
    print(rag.ask("How many leave days do I get?"))
"""

from rag_titan.app import RAGTitan
from rag_titan.config import get_settings
from rag_titan.factory import RAG_TYPES, build_rag, build_rag_parts
from rag_titan.rag_types.adaptive.adaptive_rag import AdaptiveRAG
from rag_titan.rag_types.corrective.corrective_rag import CorrectiveRAG
from rag_titan.rag_types.hyde_rag import HyDERAG
from rag_titan.rag_types.iterative.iterative_rag import IterativeRAG
from rag_titan.rag_types.parent_child_rag import ParentChildRAG
from rag_titan.rag_types.rag_fusion_rag import RAGFusionRAG
from rag_titan.rag_types.simple_rag import SimpleRAG
from rag_titan.utils.logger import setup_logging

__version__ = "0.15.2"

__all__ = [
    "RAGTitan",
    "SimpleRAG",
    "AdaptiveRAG",
    "ParentChildRAG",
    "HyDERAG",
    "RAGFusionRAG",
    "CorrectiveRAG",
    "IterativeRAG",
    "RAG_TYPES",
    "build_rag",
    "build_rag_parts",
    "get_settings",
    "setup_logging",
    "__version__",
]
