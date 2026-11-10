"""Evaluation: measure how well retrieval finds the evidence for test questions."""

from rag_titan.evaluation.benchmark import compare_retrievers, evaluate_retriever, format_report
from rag_titan.evaluation.end_to_end import compare_setups, evaluate_rag, format_rag_report
from rag_titan.evaluation.metrics import (
    matches_evidence,
    ndcg_at_k,
    parse_evidence_id,
    recall_at_k,
    reciprocal_rank,
)
from rag_titan.evaluation.questions import as_questions, check_questions, load_questions

__all__ = [
    "as_questions",
    "compare_setups",
    "evaluate_rag",
    "format_rag_report",
    "check_questions",
    "compare_retrievers",
    "evaluate_retriever",
    "format_report",
    "load_questions",
    "matches_evidence",
    "ndcg_at_k",
    "parse_evidence_id",
    "recall_at_k",
    "reciprocal_rank",
]
