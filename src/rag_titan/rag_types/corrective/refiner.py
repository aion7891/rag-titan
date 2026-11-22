"""
Keep only the sentences that help (CRAG's "decompose, then recompose",
Yan et al. 2024, simplified).

Each kept chunk is cut into sentences, every sentence gets a number, and one
LLM call returns the numbers worth keeping:

    [1] Every employee gets 20 leave days.      <- kept
    [2] The office has a coffee machine.        <- dropped
    [3] Unused days carry over to next year.    <- kept

The kept sentences are copied word for word, never rewritten, and the chunk
keeps its id, source, pages and offsets, so citations still point to real text.
"""

import json
import re

from rag_titan.chunking.structure_aware.sentences import split_sentences
from rag_titan.prompts import REFINE_PROMPT
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

KEPT_WHOLE_NOTE = "the model's reply could not be read, so the chunks were kept whole"
NONE_KEPT_NOTE = "the model kept no sentence, so the chunks were kept whole"


class SentenceRefiner:
    """
    Removes the sentences of each chunk that do not help answer the question.

    Args:
        llm: any BaseLLM.
    """

    def __init__(self, llm):
        if llm is None or not hasattr(llm, "ask"):
            raise ValueError("The sentence refiner needs a chat model: SentenceRefiner(llm)")
        self.llm = llm

    def refine(self, query, chunks):
        """
        Return the chunks with only their useful sentences.

        A chunk with no useful sentence is dropped. Every refined chunk says
        what happened in "refine_note" ("kept 2 of 5 sentences"). When the reply
        is unusable or keeps nothing at all, the chunks come back whole, with a note.
        """
        if not chunks:
            return []

        pieces = [split_sentences(chunk["text"]) for chunk in chunks]
        numbered = [(index, piece) for index, chunk_pieces in enumerate(pieces)
                    for piece in chunk_pieces if piece.strip()]

        reply = self.llm.ask(REFINE_PROMPT, build_input(query, numbered))
        keep = parse_numbers(reply, len(numbered))

        if not keep:
            note = NONE_KEPT_NOTE if keep == [] else KEPT_WHOLE_NOTE
            logger.warning("Sentence refiner: %s. Reply: %r", note, (reply or "")[:200])
            return [{**chunk, "refine_note": note} for chunk in chunks]

        kept_by_chunk = {}
        for number in keep:
            index, piece = numbered[number - 1]
            kept_by_chunk.setdefault(index, []).append(piece)

        refined = []
        for index, chunk in enumerate(chunks):
            if index in kept_by_chunk:
                total = sum(1 for piece in pieces[index] if piece.strip())
                kept = kept_by_chunk[index]
                refined.append({**chunk, "text": "".join(kept).strip(),
                                "refine_note": f"kept {len(kept)} of {total} sentences"})
        return refined


def build_input(query, numbered):
    """The user message: the question, then every sentence on its own numbered line."""
    lines = []
    for number, (_, piece) in enumerate(numbered, start=1):
        # one line per sentence, so a newline inside it cannot fake a new number
        lines.append(f"[{number}] {' '.join(piece.split())}")
    return f"QUESTION:\n{query}\n\nSENTENCES:\n" + "\n".join(lines)


def parse_numbers(reply, count):
    """
    Read the sentence numbers to keep, in order, without repeats.

    Accepts a JSON list ([1, 3]) or loose text ("1, 3" or "Sentences 1 and 3").
    Numbers outside 1..count are ignored.

    Returns:
        The numbers, sorted; [] when the model clearly kept nothing ("[]");
        None when the reply cannot be used (no number, or none in 1..count).
    """
    text = reply or ""
    # exactly one [...] is the JSON answer; "keep [1] and [3]" is loose text
    lists = re.findall(r"\[[^\[\]]*\]", text)
    try:
        data = json.loads(lists[0]) if len(lists) == 1 else None
    except ValueError:
        data = None

    if isinstance(data, list) and all(isinstance(item, int) and not isinstance(item, bool)
                                      for item in data):
        if not data:
            return []
        found = data
    else:
        found = [int(number) for number in re.findall(r"\d+", text)]

    numbers = sorted({number for number in found if 1 <= number <= count})
    return numbers or None
