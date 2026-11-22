"""
Iterative RAG (inspired by IRCoT, Trivedi et al. 2022; RAG handbook #28):

    Question -> Retrieve -> Rerank -> Think -> SEARCH: next query -> Retrieve -> Rerank -> Think ...
                                           -> DONE -> Generate (from everything collected)

For multi-hop questions, where the first search shows what to look up next:
"Who approves gear for a new starter in Berlin?" first finds "Berlin IT requests
go to the Facilities team", and only then can a search find who leads Facilities.

Each search keeps its own best top_n chunks (reranked against that search's
query), so the second hop is not thrown away for using other words than the
question. The model answers from all of them.

The loop is capped (max_steps), stops when a search finds nothing new, and
stops when the model says DONE; the result's "loop" says which one happened.

Simplified from the paper: one LLM call per round writes a thought and the next
query (IRCoT uses each new chain-of-thought sentence as the query).
"""

from rag_titan.prompts import NO_ANSWER
from rag_titan.rag_types.base_rag import BaseRAG, check_search_sizes
from rag_titan.rag_types.iterative.next_step import NextStepThinker
from rag_titan.rag_types.loops import MODEL_DONE, LoopLimit
from rag_titan.rag_types.trace import Trace


class IterativeRAG(BaseRAG):
    """
    Searches, thinks about what is missing, and searches again, then answers once.

    Args:
        llm: any BaseLLM; decides what to search for next.
        retriever: any BaseRetriever (dense, bm25 or hybrid).
        reranker: a BaseReranker; ranks each search's chunks against its own query.
        generator: a BaseGenerator.
        top_k: chunks per search.
        top_n: chunks kept from each search; the model reads all the kept ones.
        max_steps: the most think-then-search rounds after the first search;
            0 searches once, like Simple RAG.
    """

    FLOW = {
        "nodes": [
            {"id": "retrieve", "label": "Retrieve", "kind": "search"},
            {"id": "rerank", "label": "Rerank", "kind": "search"},
            {"id": "think", "label": "Think: what next?", "kind": "llm"},
            {"id": "generate", "label": "Generate answer", "kind": "output"},
            {"id": "give_up", "label": "No answer", "kind": "output"},
        ],
        "edges": [
            ["retrieve", "rerank"], ["rerank", "think"], ["think", "retrieve"],
            ["think", "generate"], ["rerank", "generate"],
            ["think", "give_up"], ["rerank", "give_up"],
        ],
    }

    DEFAULT_TOP_K, DEFAULT_TOP_N = 3, 2
    DEFAULT_MAX_STEPS = 3

    def __init__(self, llm, retriever, reranker, generator, top_k=DEFAULT_TOP_K,
                 top_n=DEFAULT_TOP_N, max_steps=DEFAULT_MAX_STEPS):
        check_search_sizes(top_k, top_n)
        LoopLimit(max_steps)  # checks the value early
        self.thinker = NextStepThinker(llm)
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.top_n = top_n
        self.max_steps = max_steps

    def run(self, query, trace=None):
        """
        Answer the question from several searches, recording each step.

        The result also has "loop": the think rounds run, the cap, and why
        the loop stopped.
        """
        query = query.strip()
        trace = trace if trace is not None else Trace()

        if not query:
            return trace.result("Please enter a question.")

        if not self.retriever.has_documents():
            return trace.result("Please index your documents first.")

        limit = LoopLimit(self.max_steps)
        collected = self.collect(query, limit, trace)

        if collected:
            answer = trace.record("generate", self.generator.generate,
                                  query=query, chunks=collected)
            result = trace.result(answer, collected)
        else:
            trace.add("give_up", {"loop": limit.summary()}, NO_ANSWER)
            result = trace.result(NO_ANSWER)

        result["loop"] = limit.summary()
        return result

    def collect(self, query, limit, trace):
        """
        Search, keep the best new chunks, ask the model what is missing, repeat.
        Returns every kept chunk, in the order they were found.
        """
        collected, searches, search_query = [], [query], query

        while True:
            found = trace.search(self.retriever, query=search_query, top_k=self.top_k)
            # only chunks not seen before count; none new stops the loop
            fresh = limit.new_chunks(found)
            collected += trace.record("rerank", self.reranker.rerank,
                                      query=search_query, matches=fresh, top_n=self.top_n)
            if not limit.next_step():
                return collected

            # copies: the trace must keep what the model saw, not the lists' later contents
            step = trace.record("think", self.thinker.think,
                                query=query, chunks=list(collected), searches=list(searches))
            if step["search"] is None:
                limit.stop(MODEL_DONE)
                return collected

            search_query = step["search"]
            searches.append(search_query)
