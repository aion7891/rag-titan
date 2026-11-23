"""
Record the playground's runs with a real model once, and check that they replay.

    presets = record_presets(documents, questions, PRESET_CHUNKING, llm, embedder,
                             grid=PRESET_GRID)
    problems = check_presets(presets)       # [] when every replay gives the same run

Every RAG type runs with every chunking, retriever and question: first with its
defaults, then with each value of PRESET_GRID, one knob at a time (so "top_k=8"
means top_k 8 and every other knob at its default). The pages mark exactly those
settings as "answers recorded".

Recording and replaying both go through Playground, the code the browser runs:
only the model and the embedder differ (recording vs replaying).
"""

from rag_titan.embeddings.replay_embedder import RecordingEmbedder
from rag_titan.factory import RAG_TYPES, RETRIEVERS
from rag_titan.llm.replay_llm import RecordingLLM
from rag_titan.playground.presets import build_parts, index_documents, preset_settings
from rag_titan.playground.session import Playground
from rag_titan.playground.tuning import RETRIEVER_DEFAULTS, default_tuning

RERANKER_VALUES = ["keyword", "none", "mmr", "llm"]

# the values recorded for each knob, one knob at a time (the default is included)
PRESET_GRID = {
    # the hybrid knobs are recorded for Simple RAG only: they change nothing but the search
    "simple": {"top_k": [3, 5, 8], "top_n": [1, 2, 3], "reranker": RERANKER_VALUES,
               "hybrid_dense_weight": [1.0, 2.0], "hybrid_bm25_weight": [1.0, 2.0],
               "rrf_keep_top": [False, True]},
    "adaptive": {"top_k": [2, 4, 6], "top_n": [1, 2, 3], "adaptive_max_attempts": [1, 2, 3],
                 "reranker": RERANKER_VALUES},
    "parent_child": {"top_k": [4, 8, 12], "top_n": [2, 4, 6], "parent_budget": [300, 600, 3000],
                     "parent_size": [1500, 200], "reranker": RERANKER_VALUES},
    "hyde": {"top_k": [3, 5, 8], "top_n": [1, 2, 3], "hyde_with_question": [True, False],
             "question_weight": [0.5, 1.0, 2.0], "rrf_keep_top": [False, True],
             "reranker": RERANKER_VALUES},
    "rag_fusion": {"top_k": [3, 5, 8], "top_n": [1, 2, 3], "fusion_queries": [1, 3, 5],
                   "question_weight": [0.5, 1.0, 2.0], "rrf_keep_top": [False, True],
                   "reranker": RERANKER_VALUES},
    "corrective": {"top_k": [3, 5, 8], "top_n": [1, 2, 3], "crag_upper": [5, 7, 9],
                   "crag_lower": [1, 3, 5], "crag_refine": [True, False],
                   "crag_max_retries": [0, 1, 2], "reranker": RERANKER_VALUES},
    "iterative": {"top_k": [2, 3, 5], "top_n": [1, 2, 3], "iterative_max_steps": [0, 1, 3],
                  "reranker": RERANKER_VALUES},
}


def grid_tunings(rag_type, grid, reranker):
    """
    The tunings to record for one RAG type: ("default", None) first, then one
    ("top_k=8", {"top_k": 8}) per knob value that is not the default.

    Args:
        reranker: the presets' default reranker (the other knobs' defaults come
            from the RAG type itself).
    """
    defaults = {**RETRIEVER_DEFAULTS, **default_tuning()[rag_type], "reranker": reranker}
    tunings = [("default", None)]
    for knob, values in (grid or {}).get(rag_type, {}).items():
        for value in values:
            if value != defaults.get(knob):
                tunings.append((f"{knob}={value}", {knob: value}))
    return tunings


def record_presets(documents, questions, chunking, llm, embedder, grid=None, earlier=None):
    """
    Run and record every RAG type x chunking x retriever x tuning x question.

    Args:
        documents: list of {"source", "text"} dicts.
        questions: list of question dicts (see load_questions).
        chunking: dict of {name: setting overrides}, e.g. PRESET_CHUNKING.
        llm: the real model, e.g. OllamaLLM("gemma3:4b").
        embedder: the real embedder, e.g. OllamaEmbedder("nomic-embed-text").
        grid: the knob values to record, e.g. PRESET_GRID; None records the
            defaults only.
        earlier: optional, an earlier recording (presets dict). Prompts and texts
            it holds reuse its replies and vectors, so the runs recorded before
            stay exactly the same and only new ones ask the model.

    Returns:
        A JSON-ready dict: documents, questions, settings, the grid, the recorded
        replies and vectors, and the expected summary of every run:
        expected[chunking][retriever][rag_type][tuning label][question id].
    """
    earlier = earlier or {}
    llm = RecordingLLM(llm, known=earlier.get("replies"))
    embedder = RecordingEmbedder(embedder, known=earlier.get("vectors"))
    parent_sizes = ((grid or {}).get("parent_child", {}).get("parent_size")
                    or [default_tuning()["parent_child"]["parent_size"]])
    presets = {"model": llm.model, "embed_model": embedder.model, "documents": documents,
               "questions": questions, "rag_types": list(RAG_TYPES),
               "retrievers": list(RETRIEVERS), "parent_sizes": list(parent_sizes),
               "settings": {name: preset_settings(o) for name, o in chunking.items()},
               "grid": grid or {}}

    def make_parts(name, parent_size):
        parts = build_parts(presets["settings"][name], embedder, llm, parent_size=parent_size)
        index_documents(parts, documents)
        return parts

    presets["expected"] = run_everything(Playground(presets, make_parts))
    presets["replies"], presets["vectors"] = llm.replies, embedder.vectors
    return presets


def check_presets(presets):
    """Replay every recorded run; return a list of differences ([] means all the same)."""
    replayed = run_everything(Playground(presets))
    problems = []
    for path, run in _flatten(presets["expected"]):
        again = _lookup(replayed, path)
        if again != run:
            problems.append(f"{' / '.join(path)}: recorded {run}, replayed {again}")
    return problems


def run_everything(playground):
    """expected[chunking][retriever][rag_type][tuning label][question id] = summarize_run(...)."""
    presets = playground.presets
    reranker = next(iter(presets["settings"].values()))["reranker"]
    expected = {}
    for name in presets["settings"]:
        for retriever in presets["retrievers"]:
            for rag_type in presets["rag_types"]:
                runs = expected.setdefault(name, {}).setdefault(retriever, {})
                for label, tuning in grid_tunings(rag_type, presets["grid"], reranker):
                    runs.setdefault(rag_type, {})[label] = {
                        q["id"]: summarize_run(playground.run(name, rag_type, q["id"],
                                                              retriever, tuning))
                        for q in presets["questions"]}
    return expected


def summarize_run(run):
    """The part of a playground run a replay must repeat exactly (timings change)."""
    return {"answer": run["answer"], "steps": [step["step"] for step in run["steps"]],
            "sources": run["sources"]}


def _flatten(tree, path=()):
    """Every (path, run) in the nested expected dict; a run is the dict with "steps"."""
    if "steps" in tree:
        yield path, tree
        return
    for key, value in tree.items():
        yield from _flatten(value, path + (key,))


def _lookup(tree, path):
    """tree[path[0]][path[1]]..., or None when a key is missing."""
    for key in path:
        tree = tree.get(key) if isinstance(tree, dict) else None
    return tree
