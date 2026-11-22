"""
Fakes for the RAG types that correct themselves (Corrective, Iterative):
an LLM that replies by prompt, and a retriever that returns prepared lists.
"""

from rag_titan.llm import BaseLLM
from rag_titan.prompts import (
    LLM_RERANK_PROMPT,
    NEXT_STEP_PROMPT,
    QUERY_REWRITE_PROMPT,
    REFINE_PROMPT,
)


class ScriptLLM(BaseLLM):
    """Replies by prompt: the next grade reply, the rewrite, the refine reply, else the answer."""

    def __init__(self, grades, rewrite="annual leave allowance", refine="[1]", answer="20 days"):
        super().__init__("script-fake")
        self.grades = list(grades)
        self.replies = {QUERY_REWRITE_PROMPT: rewrite, REFINE_PROMPT: refine}
        self.answer = answer
        self.calls = []

    def chat(self, messages, tools=None):
        system, user = messages[0]["content"], messages[1]["content"]
        self.calls.append((system, user))
        if system == LLM_RERANK_PROMPT:
            reply = self.grades.pop(0)
        else:
            reply = self.replies.get(system, self.answer)
        return {"content": reply, "tool_calls": [], "message": {"role": "assistant"}}

    def prompts(self):
        return [system for system, _ in self.calls]


class ListRetriever:
    """Returns the prepared result lists in order (the last one repeats)."""

    def __init__(self, *results):
        self.results = list(results)
        self.queries = []

    def retrieve(self, query, top_k):
        self.queries.append(query)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]

    def has_documents(self):
        return True


def names(result):
    """The step names of a run, in order."""
    return [step["step"] for step in result["steps"]]


def assert_follows_flow(rag, result):
    """Every step is a FLOW node and two steps in a row are joined by an edge."""
    flow = rag.flow()
    edges = {tuple(edge) for edge in flow["edges"]}
    steps = names(result)
    assert set(steps) <= {node["id"] for node in flow["nodes"]}
    for pair in zip(steps, steps[1:]):
        assert pair in edges, f"no FLOW edge {pair}"


class ThinkLLM(BaseLLM):
    """Replies to the "think" prompt from a list (the last one repeats), else the answer."""

    def __init__(self, steps, answer="the answer"):
        super().__init__("think-fake")
        self.steps = list(steps)
        self.answer = answer
        self.calls = []

    def chat(self, messages, tools=None):
        system, user = messages[0]["content"], messages[1]["content"]
        self.calls.append((system, user))
        if system == NEXT_STEP_PROMPT:
            reply = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        else:
            reply = self.answer
        return {"content": reply, "tool_calls": [], "message": {"role": "assistant"}}
