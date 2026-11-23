"""
End-to-end evaluation: run whole RAG types on test questions, with any ingestion settings.

    results = compare_setups(
        files=["leave_policy.md", "expenses.md"],
        questions="questions.json",
        setups=[
            {"name": "simple + word", "rag_type": "simple", "settings": {"chunker": "word"}},
            {"name": "adaptive + word", "rag_type": "adaptive", "settings": {"chunker": "word"}},
        ],
    )
    print(format_rag_report(results))

The evidence is scored on the chunks each RAG type really gave to the LLM
(the "sources" of run()), not on the first search only.
"""

import json

from rag_titan.config import get_settings
from rag_titan.evaluation.benchmark import format_table, summarize_rows
from rag_titan.evaluation.questions import as_questions
from rag_titan.evaluation.run_scoring import (  # noqa: F401  (NOT_ANSWERED: old import path)
    NOT_ANSWERED,
    run_question,
    score_sources,
)
from rag_titan.factory import (
    PARENT_RAG_TYPES,
    build_rag,
    build_rag_parts,
    build_reranker,
    build_retriever,
)

# settings that change how the chunks are searched and used, not the chunks themselves
SEARCH_ONLY = ("retriever", "reranker", "rerank_model", "top_k", "top_n",
               "adaptive_max_attempts", "parent_budget", "hyde_with_question",
               "fusion_queries", "crag_upper", "crag_lower", "crag_refine", "crag_max_retries",
               "iterative_max_steps", "hybrid_dense_weight", "hybrid_bm25_weight",
               "question_weight", "rrf_keep_top")

RAG_COLUMNS = [
    ("Recall", "recall", "{:.3f}"),
    ("MRR", "mrr", "{:.3f}"),
    ("nDCG", "ndcg", "{:.3f}"),
    ("Answered", "answered", "{:.0%}"),
    ("Steps", "steps", "{:.1f}"),
    ("p50 s", "p50_s", "{:.2f}"),
    ("p95 s", "p95_s", "{:.2f}"),
]


def evaluate_rag(rag, questions, name=None):
    """
    Answer every question with rag.run() and score the run.

    Args:
        rag: any BaseRAG (SimpleRAG, AdaptiveRAG, ...) with indexed documents.
        questions: path to a questions file, or a list of question dicts.
        name: name shown in the report, defaults to the class name.

    Returns:
        A dict with the means ("recall", "mrr", "ndcg" on the sources,
        "answered" share, average "steps"), the total time per question
        ("p50_s", "p95_s") and one row per question in "rows"
        (each row keeps the answer and every step).
    """
    questions = as_questions(questions)
    rows = [run_question(rag, question) for question in questions]

    summary = summarize_rows(rows)
    result = {"name": name or type(rag).__name__}
    result.update({key: summary[key] for key in ("recall", "mrr", "ndcg")})
    result["answered"] = sum(row["answered"] for row in rows) / len(rows)
    result["steps"] = sum(len(row["steps"]) for row in rows) / len(rows)
    result["p50_s"] = summary["p50_ms"] / 1000
    result["p95_s"] = summary["p95_ms"] / 1000
    result["rows"] = rows
    return result


def compare_setups(files, questions, setups):
    """
    Index the files and evaluate every setup on the same questions.

    Args:
        files: list of file paths to index.
        questions: path to a questions file, or a list of question dicts.
        setups: list of {"name": ..., "rag_type": "simple", "settings": {...}}.
            "settings" changes any setting (chunker, chunk_size, models, ...).

    Returns:
        A list with one evaluate_rag() result per setup, in order; each also
        has the setup's "rag_type" and "settings".

    Setups with the same ingestion settings share one index, so comparing RAG
    types, retrievers or rerankers on the same chunks embeds the files only once.
    """
    if not isinstance(files, list) or not files:
        raise ValueError("files must be a non-empty list of file paths")
    questions = as_questions(questions)
    _check_setups(setups)

    indexes = {}
    results = []
    for setup in setups:
        parts = _indexed_parts(files, setup.get("settings") or {}, indexes, setup["rag_type"])
        rag = build_rag(setup["rag_type"], parts)

        result = evaluate_rag(rag, questions, name=setup["name"])
        result["rag_type"] = setup["rag_type"]
        result["settings"] = setup.get("settings") or {}
        results.append(result)

    return results


def format_rag_report(results):
    """Return the end-to-end results as a text table, one line per setup."""
    if not results:
        raise ValueError("There are no results to report")

    title = (f"End-to-end report: {len(results[0]['rows'])} questions "
             "(evidence scored on the chunks given to the LLM)")
    return format_table(title, "setup", results, RAG_COLUMNS)


def _indexed_parts(files, overrides, indexes, rag_type=None):
    """
    Return built and indexed parts, reusing them when the settings are the same.

    A RAG type that needs parents (parent_child) gets an index of its own, so
    the chunks of the other setups stay exactly as they were.
    """
    # always a temporary in-memory index: comparing setups must never replace
    # (or mix several setups into) an index saved with VECTOR_STORE=sqlite
    settings = get_settings({**overrides, "vector_store": "memory"})
    ingestion = {key: value for key, value in settings.items() if key not in SEARCH_ONLY}
    ingestion["parents"] = rag_type in PARENT_RAG_TYPES
    key = json.dumps(ingestion, sort_keys=True, default=str)

    if key not in indexes:
        parts = build_rag_parts(settings, rag_type)
        parts["pipeline"].run(files)
        indexes[key] = parts

    # same chunks, but this setup's own way of searching and reranking them
    pipeline = indexes[key]["pipeline"]
    return {**indexes[key],
            "retriever": build_retriever(settings, pipeline.embedder, pipeline.store),
            "reranker": build_reranker(settings, indexes[key]["llm"], pipeline.embedder),
            "settings": settings}


def _check_setups(setups):
    """Raise a clear error for a setup list that cannot be compared."""
    if not isinstance(setups, list) or not setups:
        raise ValueError("setups must be a non-empty list of dicts")

    names = set()
    for number, setup in enumerate(setups, start=1):
        if not isinstance(setup, dict) or not setup.get("name") or not setup.get("rag_type"):
            raise ValueError(f"Setup {number} needs a 'name' and a 'rag_type'")
        if setup["name"] in names:
            raise ValueError(f"Setup name {setup['name']!r} is used more than once")
        if not isinstance(setup.get("settings") or {}, dict):
            raise ValueError(f"Setup {setup['name']!r}: 'settings' must be a dict")
        names.add(setup["name"])
