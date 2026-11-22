"""
Run a question in the playground with any tuning, also past what was recorded.

The recorded model replies fit the prompts of the recorded settings. Move a
slider (top_k, top_n, ...) and the search, reranking and parent steps still
run for real; only a model reply for a prompt never recorded is missing. Then
the run stops at that step, and the page shows how far it got and exactly
what the model would have read, instead of an error.
"""

import time

from rag_titan.evaluation.run_scoring import NOT_ANSWERED, score_sources
from rag_titan.rag_types.trace import Trace
from rag_titan.utils.not_recorded import NotRecorded


def run_question(rag, question):
    """
    Run one preset question and score it.

    Args:
        rag: a RAG type built on replayed parts.
        question: a preset question dict (with "required_evidence_ids").

    Returns:
        {"answer", "steps", "sources", "recall", "ndcg", "answered", "ms",
        "recorded", "stopped_at", "loop"}. "loop" is how a correcting RAG type's
        loop went (rounds, cap, stop reason), else None. When a model reply was not recorded:
        "recorded" is False, "answer" and "answered" are None, "stopped_at"
        names the step, and the scores are those of the chunks the model
        would have been given (when the run got as far as the answer).
    """
    trace = Trace()
    start = time.perf_counter()
    try:
        result = rag.run(question["question"], trace=trace)
    except NotRecorded:
        return _stopped(question, trace.steps, (time.perf_counter() - start) * 1000)

    scores = score_sources(question, result["sources"])
    return {"answer": result["answer"], "steps": result["steps"],
            "sources": scores["sources"], "recall": scores["recall"],
            "ndcg": scores["ndcg"],
            "answered": result["answer"].strip() not in NOT_ANSWERED,
            "ms": (time.perf_counter() - start) * 1000, "recorded": True, "stopped_at": None,
            "loop": result.get("loop")}


def _stopped(question, steps, ms):
    """The result of a run that stopped at a step whose model reply was not recorded."""
    last = steps[-1] if steps else None
    # Trace saves a failed step with the output {"error": ...}
    failed = None
    if last and isinstance(last["output"], dict) and "error" in last["output"]:
        failed = last
    # at the answer step, the context the model would get is known: score it
    context = failed["input"].get("chunks") if failed else None
    scores = score_sources(question, context) if context else None
    return {"answer": None, "steps": steps,
            "sources": scores["sources"] if scores else [],
            "recall": scores["recall"] if scores else None,
            "ndcg": scores["ndcg"] if scores else None,
            "answered": None, "ms": ms, "recorded": False,
            "stopped_at": failed["step"] if failed else None, "loop": None}


def summarize_questions(rows):
    """
    Sum up the runs of every question with one setup.

    Returns:
        {"questions", "recorded", "recall", "answered"}: recall is the mean
        over the runs with scored evidence, answered the share of the recorded
        runs that gave an answer (None when no run could be scored).
    """
    scored = [row["recall"] for row in rows if row["recall"] is not None]
    recorded = [row for row in rows if row["recorded"]]
    return {
        "questions": len(rows),
        "recorded": len(recorded),
        "recall": sum(scored) / len(scored) if scored else None,
        "answered": (sum(1 for row in recorded if row["answered"]) / len(recorded)
                     if recorded else None),
    }
