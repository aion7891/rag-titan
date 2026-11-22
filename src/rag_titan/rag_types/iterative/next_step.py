"""
The "think" step of Iterative RAG: the model reads what was found so far and
says what to look up next (IRCoT's "reason, then generate the next query").

It must reply with a short thought and, on the last line, SEARCH: <query> or DONE:

    The passages say IT requests in Berlin go to the Facilities team.
    SEARCH: Facilities team lead

Small models do not always follow the format, so the reply is read loosely
(labels in bold or backticks, any case). A reply with neither line stops the
loop, with a note in the trace: the loop never guesses a search.
"""

import re

from rag_titan.prompts import NEXT_STEP_PROMPT
from rag_titan.retrieval.rerankers.llm_reranker import FENCE_TAG

# each passage is cut to this many characters, so one long chunk cannot crowd out the rest
MAX_PASSAGE_CHARS = 800

# "SEARCH: query", "**Search:** query", "`SEARCH`: query"
SEARCH_LINE = re.compile(r"^search\s*:\s*(.*)$", re.IGNORECASE)
DONE_LINE = re.compile(r"^done\b", re.IGNORECASE)

NO_DECISION_NOTE = "the reply had no SEARCH: or DONE line, so the loop stopped"
EMPTY_SEARCH_NOTE = "the SEARCH: line was empty, so the loop stopped"


class NextStepThinker:
    """
    Asks the model for the next search, or whether it has enough.

    Args:
        llm: any BaseLLM.
    """

    def __init__(self, llm):
        if llm is None or not hasattr(llm, "ask"):
            raise ValueError("Iterative RAG needs a chat model: NextStepThinker(llm)")
        self.llm = llm

    def think(self, query, chunks, searches):
        """
        Returns:
            {"thought": the model's sentence,
             "search": the next search query, or None when the model is done,
             "note": why the loop stopped, when the reply was not usable (optional)}
        """
        reply = self.llm.ask(NEXT_STEP_PROMPT, build_input(query, chunks, searches))
        return parse_next_step(reply)


def build_input(query, chunks, searches):
    """The user message: the question, the searches so far, then the fenced passages."""
    asked = "\n".join(f"- {search}" for search in searches)
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        text = FENCE_TAG.sub("passage", chunk["text"][:MAX_PASSAGE_CHARS])
        blocks.append(f"<passage {number}>\n{text}\n</passage {number}>")
    passages = "\n\n".join(blocks) or "(nothing found yet)"
    return f"QUESTION:\n{query}\n\nSEARCHES SO FAR:\n{asked}\n\nPASSAGES:\n{passages}"


def parse_next_step(reply):
    """
    Read the model's reply: the last SEARCH: or DONE line decides, the lines
    before it are the thought. See NextStepThinker.think() for what it returns.
    """
    lines = [_plain(line) for line in (reply or "").splitlines()]
    lines = [line for line in lines if line]

    for index in range(len(lines) - 1, -1, -1):
        search = SEARCH_LINE.match(lines[index])
        done = DONE_LINE.match(lines[index])
        if not (search or done):
            continue

        thought = " ".join(lines[:index])
        if done:
            return {"thought": thought, "search": None}
        query = search.group(1).strip().strip("\"'<>").strip()
        if not query:
            return {"thought": thought, "search": None, "note": EMPTY_SEARCH_NOTE}
        return {"thought": thought, "search": query}

    return {"thought": " ".join(lines), "search": None, "note": NO_DECISION_NOTE}


def _plain(line):
    """
    A line without markdown marks: "**SEARCH:** x" -> "SEARCH: x", "- DONE" -> "DONE".
    Only bold/code marks and leading list or quote marks go: "error #42" stays.
    """
    return re.sub(r"[*`]+", "", line).strip().lstrip("#>- ").strip()
