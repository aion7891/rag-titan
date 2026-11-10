"""Evaluation: measure how well retrieval finds the evidence for test questions."""

from rag_titan.evaluation.metrics import (
    matches_evidence,
    ndcg_at_k,
    parse_evidence_id,
    recall_at_k,
    reciprocal_rank,
)
from rag_titan.evaluation.questions import check_questions, load_questions

__all__ = [
    "check_questions",
    "load_questions",
    "matches_evidence",
    "ndcg_at_k",
    "parse_evidence_id",
    "recall_at_k",
    "reciprocal_rank",
]
