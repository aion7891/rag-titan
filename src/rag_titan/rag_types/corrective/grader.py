"""
Grade every retrieved chunk, then decide what to do with the evidence
(CRAG's "retrieval evaluator", Yan et al. 2024, simplified).

One LLM call rates all the chunks from 0 to 10, with the same prompt, fencing
and reply reading as the LLM reranker. The grades give one of three verdicts:

    correct     at least one chunk scores >= upper   -> keep the evidence
    incorrect   every chunk scores < lower            -> throw it away, search again
    ambiguous   anything in between                   -> keep it and search again

The paper trains a small T5 model as the evaluator; here the chat model does it.
"""

from rag_titan.prompts import LLM_RERANK_PROMPT
from rag_titan.retrieval.rerankers.llm_reranker import LLMReranker, parse_scores
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

CORRECT, AMBIGUOUS, INCORRECT = "correct", "ambiguous", "incorrect"

NOTHING_FOUND_NOTE = "there were no chunks to grade"
UNGRADED_NOTE = "the model's grades could not be read, so the evidence counts as ambiguous"
NOT_GRADED_NOTE = "the model gave no grade for this chunk, so it counts as 0"


class ChunkGrader:
    """
    Rates each chunk from 0 to 10 and gives the evidence a verdict.

    Args:
        llm: any BaseLLM.
        upper: a chunk graded this or more makes the evidence "correct".
        lower: when every chunk is graded below this, the evidence is "incorrect".
    """

    def __init__(self, llm, upper=7, lower=3):
        check_thresholds(upper, lower)
        # the reranker checks the llm and builds the fenced, numbered prompt
        self.reranker = LLMReranker(llm)
        self.llm = llm
        self.upper = upper
        self.lower = lower

    def grade(self, query, chunks):
        """
        Grade the chunks against the question.

        Returns:
            A dict with "verdict" ("correct", "ambiguous" or "incorrect"),
            "chunks" (copies of the chunks in search order, each with its
            "grade" from 0 to 10) and, when something went wrong, "note" (why).
        """
        if not chunks:
            return {"verdict": INCORRECT, "chunks": [], "note": NOTHING_FOUND_NOTE}

        reply = self.llm.ask(LLM_RERANK_PROMPT, self.reranker.build_input(query, chunks))
        scores = parse_scores(reply, len(chunks))

        if not scores:
            # unsure is safer than wrong: keep the chunks and also search again
            logger.warning("Chunk grader: %s. Reply: %r", UNGRADED_NOTE, (reply or "")[:200])
            return {"verdict": AMBIGUOUS, "chunks": [dict(chunk) for chunk in chunks],
                    "note": UNGRADED_NOTE}

        graded = []
        for number, chunk in enumerate(chunks, start=1):
            item = {**chunk, "grade": scores.get(number, 0.0)}
            if number not in scores:
                item["grade_note"] = NOT_GRADED_NOTE
            graded.append(item)

        grades = [item["grade"] for item in graded]
        return {"verdict": decide(grades, self.upper, self.lower), "chunks": graded}


def decide(grades, upper, lower):
    """The verdict for a list of 0-10 grades (see the module docstring)."""
    if any(grade >= upper for grade in grades):
        return CORRECT
    if all(grade < lower for grade in grades):
        return INCORRECT
    return AMBIGUOUS


def check_thresholds(upper, lower):
    """Raise a clear ValueError unless 0 <= lower <= upper <= 10."""
    for name, value in (("upper", upper), ("lower", lower)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 10:
            raise ValueError(f"{name} must be a number from 0 to 10, got {value!r}")
    if lower > upper:
        raise ValueError(f"lower ({lower}) cannot be more than upper ({upper}): a chunk "
                         "graded between them makes the evidence ambiguous")
