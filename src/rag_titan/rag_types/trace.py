"""
Records every step a RAG type takes, so you can see (and draw) how an answer was made.

    trace = Trace()
    matches = trace.record("retrieve", retriever.retrieve, query="leave?", top_k=4)
    print(describe_steps(trace.steps))
"""

import time

# how many characters of a long text describe_steps() shows
SHORT_TEXT = 60


class Trace:
    """
    A list of steps. Each step is a dict:
        {"step": "retrieve", "input": {...}, "output": ..., "ms": 1.25}
    """

    def __init__(self):
        self.steps = []

    def record(self, step, function, **inputs):
        """
        Call function(**inputs), save it as a step, and return its output.

        Args:
            step: the step name, one of the RAG type's FLOW node ids.
            function: the function to call, e.g. retriever.retrieve.
            inputs: the keyword arguments for the function.
        """
        start = time.perf_counter()
        output = function(**inputs)
        ms = (time.perf_counter() - start) * 1000

        self.add(step, inputs, output, ms)
        return output

    def add(self, step, inputs, output, ms=0.0):
        """Save a step that was not made by calling a function (e.g. giving up)."""
        self.steps.append({"step": step, "input": inputs, "output": output, "ms": round(ms, 3)})

    def result(self, answer, sources=None):
        """Return what run() gives back: the answer, its sources and the steps."""
        return {"answer": answer, "sources": list(sources or []), "steps": self.steps}


def describe_steps(steps):
    """
    Return a short, readable log of the steps, one line per step.

    Example line:
        2. retrieve   0.21 ms  query='how many leave days?', top_k=4
                               -> 3 chunks: chunk_0 (0.894), chunk_2 (0.000), ...
    """
    if not steps:
        return "(no steps: the question was not searched)"

    lines = []
    width = max(len(step["step"]) for step in steps)

    for number, step in enumerate(steps, start=1):
        # chunk lists in the input are the previous step's output, so only count them
        inputs = ", ".join(
            f"{key}={_short(value, brief=True)}" for key, value in step["input"].items()
        )
        head = f"{number}. {step['step']:<{width}}  {step['ms']:7.2f} ms  "
        lines.append(head + inputs)
        lines.append(" " * len(head) + "-> " + _short(step["output"]))

    return "\n".join(lines)


def _short(value, brief=False):
    """Describe a value in a few words: chunk lists by id and score, long text cut."""
    if isinstance(value, list) and all(isinstance(item, dict) and "id" in item for item in value):
        return f"{len(value)} chunks" if brief else _describe_chunks(value)

    text = repr(value)
    if len(text) > SHORT_TEXT:
        text = text[:SHORT_TEXT - 3] + "..."
    return text


def _describe_chunks(chunks):
    """Return e.g. "2 chunks: chunk_0 (0.894), chunk_1 (rerank 0.500)"."""
    if not chunks:
        return "0 chunks"

    parts = []
    for chunk in chunks[:4]:
        if "rerank_score" in chunk:
            parts.append(f"{chunk['id']} (rerank {chunk['rerank_score']:.3f})")
        elif "score" in chunk:
            parts.append(f"{chunk['id']} ({chunk['score']:.3f})")
        else:
            parts.append(chunk["id"])

    more = ", ..." if len(chunks) > 4 else ""
    return f"{len(chunks)} chunks: " + ", ".join(parts) + more
