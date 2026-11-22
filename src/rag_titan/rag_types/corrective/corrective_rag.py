"""
Corrective RAG (inspired by CRAG, Yan et al. 2024; RAG handbook #27):

    Question -> Retrieve -> Grade each chunk -> correct:   keep them
                                             -> ambiguous: keep the good ones, search again
                                             -> incorrect: throw them away, search again
             -> (search again = rewrite the question, retrieve, grade the new chunks)
             -> Rerank the kept chunks -> Keep useful sentences -> Generate
             -> nothing kept after the retries -> "I don't know"

Plain RAG answers from whatever the search found, even when it is off-topic.
This RAG grades the evidence first and repairs it: a better search when it is
weak, and only the useful sentences when it is good.

Simplified from the paper: the chat model grades (no trained T5 evaluator), and
"search again" uses your own index with a rewritten question; the paper uses
web search there, which you can plug in with `fallback_retriever`.
"""

from rag_titan.prompts import NO_ANSWER
from rag_titan.rag_types.adaptive.query_rewriter import QueryRewriter
from rag_titan.rag_types.base_rag import BaseRAG, check_search_sizes
from rag_titan.rag_types.corrective.grader import CORRECT, INCORRECT, ChunkGrader
from rag_titan.rag_types.corrective.refiner import SentenceRefiner
from rag_titan.rag_types.loops import EVIDENCE_GOOD, LoopLimit
from rag_titan.rag_types.trace import Trace


class CorrectiveRAG(BaseRAG):
    """
    Grades the chunks a search found, repairs weak evidence, then answers.

    Args:
        llm: any BaseLLM; grades, rewrites and picks the useful sentences.
        retriever: any BaseRetriever (dense, bm25 or hybrid).
        reranker: a BaseReranker; orders the kept chunks against the question.
        generator: a BaseGenerator.
        top_k: chunks per search.
        top_n: chunks the model reads.
        upper: a chunk graded this or more (0-10) makes the evidence "correct".
        lower: every chunk graded below this makes it "incorrect"; chunks
            graded below it are never kept.
        refine: keep only the useful sentences of the chunks the model reads.
        max_retries: how many times to search again; 0 grades only.
        fallback_retriever: where to search again (e.g. your own web search);
            by default the same retriever.
    """

    FLOW = {
        "nodes": [
            {"id": "retrieve", "label": "Retrieve", "kind": "search"},
            {"id": "grade", "label": "Grade chunks", "kind": "llm"},
            {"id": "rewrite", "label": "Rewrite query", "kind": "llm"},
            {"id": "rerank", "label": "Rerank", "kind": "search"},
            {"id": "refine", "label": "Keep key sentences", "kind": "llm"},
            {"id": "generate", "label": "Generate answer", "kind": "output"},
            {"id": "give_up", "label": "No answer", "kind": "output"},
        ],
        "edges": [
            ["retrieve", "grade"], ["grade", "rewrite"], ["rewrite", "retrieve"],
            ["grade", "rerank"], ["grade", "give_up"],
            ["rerank", "refine"], ["refine", "generate"], ["rerank", "generate"],
        ],
    }

    DEFAULT_TOP_K, DEFAULT_TOP_N = 5, 2
    DEFAULT_UPPER, DEFAULT_LOWER = 7, 3
    DEFAULT_REFINE = True
    DEFAULT_MAX_RETRIES = 1

    def __init__(self, llm, retriever, reranker, generator, top_k=DEFAULT_TOP_K,
                 top_n=DEFAULT_TOP_N, upper=DEFAULT_UPPER, lower=DEFAULT_LOWER,
                 refine=DEFAULT_REFINE, max_retries=DEFAULT_MAX_RETRIES,
                 fallback_retriever=None):
        check_search_sizes(top_k, top_n)
        if not isinstance(refine, bool):
            raise ValueError(f"refine must be True or False, got {refine!r}")
        LoopLimit(max_retries, name="max_retries")  # checks the value early

        self.grader = ChunkGrader(llm, upper=upper, lower=lower)
        self.refiner = SentenceRefiner(llm)
        self.rewriter = QueryRewriter(llm)
        self.retriever = retriever
        self.fallback_retriever = fallback_retriever or retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.top_n = top_n
        self.refine = refine
        self.max_retries = max_retries

    def run(self, query, trace=None):
        """
        Answer the question from graded, repaired evidence, recording each step.

        The result also has "loop": the searches again, the cap, and why it stopped.
        """
        query = query.strip()
        trace = trace if trace is not None else Trace()

        if not query:
            return trace.result("Please enter a question.")

        if not self.retriever.has_documents():
            return trace.result("Please index your documents first.")

        limit = LoopLimit(self.max_retries, name="max_retries")
        kept = self.gather(query, limit, trace)

        if kept:
            result = self.answer_from(query, kept, trace)
        else:
            trace.add("give_up", {"loop": limit.summary()}, NO_ANSWER)
            result = trace.result(NO_ANSWER)

        result["loop"] = limit.summary()
        return result

    def gather(self, query, limit, trace):
        """
        Search, grade, and search again until the evidence is good or the
        retries run out. Returns the kept chunks, best grade first.
        """
        found = trace.search(self.retriever, query=query, top_k=self.top_k)
        limit.remember(found)
        kept = []

        while True:
            graded = trace.record("grade", self.grader.grade, query=query, chunks=found)
            kept.extend(self.keep(graded))

            if graded["verdict"] == CORRECT:
                limit.stop(EVIDENCE_GOOD)
            if not limit.next_step():
                break

            # the question is rewritten, never the answer: grading still uses `query`
            search_query = trace.record("rewrite", self.rewriter.rewrite, query=query)
            results = trace.search(self.fallback_retriever, query=search_query, top_k=self.top_k)
            # only new chunks are graded; none new stops the loop after this grade
            found = limit.new_chunks(results)

        # the grade is the model's own view of each chunk: the best go first
        return sorted(kept, key=lambda chunk: chunk.get("grade", 0.0), reverse=True)

    def keep(self, graded):
        """
        The chunks worth keeping from one grading: none when the verdict is
        "incorrect", else those graded at least `lower` (all of them when the
        grades could not be read).
        """
        if graded["verdict"] == INCORRECT:
            return []
        lower = self.grader.lower
        return [chunk for chunk in graded["chunks"] if chunk.get("grade", lower) >= lower]

    def answer_from(self, query, kept, trace):
        """Rerank the kept chunks, keep their useful sentences, and answer."""
        best = trace.record("rerank", self.reranker.rerank,
                            query=query, matches=kept, top_n=self.top_n)
        if self.refine:
            best = trace.record("refine", self.refiner.refine, query=query, chunks=best)

        answer = trace.record("generate", self.generator.generate, query=query, chunks=best)
        return trace.result(answer, best)
