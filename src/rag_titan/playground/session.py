"""
What the browser playground calls: one object, plain JSON-ready dicts in and out.

    playground = Playground(json.loads(presets_text))
    playground.info()                                    # RAG types, FLOWs, questions, ...
    playground.run("paragraph", "adaptive", "q_002")     # answer, steps, scores
    playground.run("paragraph", "simple", "q_002", retriever="hybrid",
                   tuning={"top_k": 8, "top_n": 3, "reranker": "mmr"})
    playground.evaluate_tuned("word", "parent_child", "dense", {"parent_budget": 600})
    playground.evaluate(["word", "paragraph"], ["simple", "adaptive"], ["dense", "hybrid"])
"""

from rag_titan.chunking.parent_child import DEFAULT_PARENT_SIZE
from rag_titan.evaluation.end_to_end import evaluate_rag, format_rag_report
from rag_titan.factory import PARENT_RAG_TYPES, RAG_TYPES, build_rag, build_reranker
from rag_titan.playground.presets import replay_parts, with_retriever
from rag_titan.playground.runs import run_question, summarize_questions
from rag_titan.playground.tuning import (
    BROWSER_RERANKERS,
    RETRIEVER_DEFAULTS,
    check_tuning,
    default_tuning,
)


class Playground:
    """
    Runs the recorded presets: real code, recorded model replies and vectors.

    Args:
        presets: the dict saved by tools/playground/record_traces.py.
        make_parts: optional, make_parts(chunking, parent_size) -> indexed parts.
            By default the recording is replayed; the recorder passes parts with
            the real (recording) model, so recording and replay run the same code.
    """

    def __init__(self, presets, make_parts=None):
        self.presets = presets
        self.make_parts = make_parts or (lambda chunking, parent_size: replay_parts(
            presets, chunking, parent_size=parent_size))
        self.parts = {}       # one index per (chunking, parent size), built when first used
        self.searchers = {}   # those parts with each retriever

    def info(self):
        """Everything the pages need to draw their controls and diagrams."""
        first = next(iter(self.presets["settings"]))
        retrievers = self.presets.get("retrievers", ["dense"])
        return {
            "model": self.presets["model"],
            "embed_model": self.presets["embed_model"],
            "rag_types": list(self.presets["rag_types"]),      # the recorded ones
            "retrievers": list(retrievers),
            "rerankers": list(BROWSER_RERANKERS),
            "reranker": self.presets["settings"][first]["reranker"],
            "parent_sizes": self.parent_sizes(),
            "defaults": default_tuning(),
            "retriever_defaults": dict(RETRIEVER_DEFAULTS),  # hybrid's knobs, every page
            "grid": self.presets.get("grid", {}),             # tuning values recorded
            # flows[rag_type][retriever] for every RAG type: the diagram needs no recording
            "flows": {name: {r: build_rag(name, self._searcher(first, r)).flow()
                             for r in retrievers} for name in RAG_TYPES},
            "chunkings": self.presets["settings"],
            "questions": self.presets["questions"],
            "documents": self.presets["documents"],
        }

    def run(self, chunking, rag_type, question_id, retriever="dense", tuning=None):
        """
        Answer one preset question with one setup and optional tuning.

        Returns:
            A dict with "answer", "steps", "sources", "recall", "ndcg",
            "answered", "ms", "recorded" and "stopped_at" (see runs.run_question).
        """
        rag = self._rag(chunking, rag_type, retriever, tuning)
        return run_question(rag, self._question(question_id))

    def evaluate_tuned(self, chunking, rag_type, retriever="dense", tuning=None):
        """Every preset question with one tuned setup: the rows and their summary."""
        rag = self._rag(chunking, rag_type, retriever, tuning)
        rows = []
        for question in self.presets["questions"]:
            run = run_question(rag, question)
            rows.append({"id": question["id"], **{key: value for key, value in run.items()
                                                 if key != "steps"}})
        return {"rows": rows, **summarize_questions(rows)}

    def evaluate(self, chunkings, rag_types, retrievers=None):
        """Every RAG type x chunking x retriever on every preset question, like compare_setups()."""
        results = []
        setups = [(c, t, r) for c in chunkings for r in retrievers or ["dense"] for t in rag_types]
        for chunking, rag_type, retriever in setups:
            name = f"{rag_type} + {chunking}" + (f" + {retriever}" if retrievers else "")
            result = evaluate_rag(self._rag(chunking, rag_type, retriever),
                                  self.presets["questions"], name=name)
            # the steps of every question are too much for a report: keep the scores
            result["rows"] = [{key: value for key, value in row.items() if key != "steps"}
                              for row in result["rows"]]
            results.append(result)
        return {"results": results, "report": format_rag_report(results)}

    def parent_sizes(self):
        """The parent sizes recorded for Parent-Child RAG (the first is the default)."""
        return list(self.presets.get("parent_sizes", [DEFAULT_PARENT_SIZE]))

    def _rag(self, chunking, rag_type, retriever="dense", tuning=None):
        """Build a RAG type on the (cached) index of one chunking, retriever and tuning."""
        tuning = check_tuning(tuning, self.parent_sizes())
        parent_size = None
        if rag_type in PARENT_RAG_TYPES:
            parent_size = tuning.get("parent_size") or self.parent_sizes()[0]
        parts = self._searcher(chunking, retriever, parent_size)
        if retriever == "hybrid" and set(tuning) & set(RETRIEVER_DEFAULTS):
            # a tuned hybrid search: a new retriever on the same index (no re-indexing)
            parts = with_retriever(parts, retriever, tuning)

        reranker = tuning.get("reranker")
        if reranker and reranker != self.presets["settings"][chunking]["reranker"]:
            settings = {**self.presets["settings"][chunking], "reranker": reranker}
            parts = {**parts, "reranker": build_reranker(settings, parts["llm"], parts["embedder"])}
        return build_rag(rag_type, {**parts, "settings": tuning})

    def _searcher(self, chunking, retriever, parent_size=None):
        """The indexed parts of one chunking (with parents of this size) and one retriever."""
        if chunking not in self.presets["settings"]:
            raise ValueError(f"Unknown chunking {chunking!r}: use one of "
                             f"{list(self.presets['settings'])}")
        index = (chunking, parent_size)
        if index not in self.parts:
            self.parts[index] = self.make_parts(chunking, parent_size)
        if (index, retriever) not in self.searchers:
            self.searchers[(index, retriever)] = with_retriever(self.parts[index], retriever)
        return self.searchers[(index, retriever)]

    def _question(self, question_id):
        """Return the preset question with this id."""
        for question in self.presets["questions"]:
            if question["id"] == question_id:
                return question
        raise ValueError(f"Unknown question id {question_id!r}")
