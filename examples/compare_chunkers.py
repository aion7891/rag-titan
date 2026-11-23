"""
Compare chunkers on two kinds of text: an IT handbook in Markdown (a table, a list,
code and a warning) and meeting notes with no headings that move through four
topics. What does each chunker keep whole, where does it cut, and does a search
find the whole answer in one chunk?

Parts 1 and 2 need no model. Part 3 needs the embedder (Ollama running):
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/compare_chunkers.py
"""

import json
from pathlib import Path

from rag_titan import RAGTitan
from rag_titan.chunking import (
    HeadingChunker,
    MarkdownChunker,
    ParagraphChunker,
    RecursiveChunker,
    TextTilingChunker,
)
from rag_titan.evaluation import ChunkEvaluator

HERE = Path(__file__).parent
DOCS = HERE / "structured_docs"
SIZE = 300   # characters per chunk, the same for every chunker

handbook = (DOCS / "it_handbook.md").read_text(encoding="utf-8")
notes = (DOCS / "meeting_notes.txt").read_text(encoding="utf-8")
# what must stay whole (table rows, list, code, warning) and the 4 real topics
gold = json.loads((HERE / "chunker_gold.json").read_text(encoding="utf-8"))


def score(chunker, text, **labels):
    """The chunks of `text` and ChunkEvaluator's scores for them."""
    spans = chunker.split_spans(text)
    chunks = [{"text": text[start:end], "start": start, "end": end} for _, start, end in spans]
    return [chunk for chunk, _, _ in spans], ChunkEvaluator().evaluate(
        chunks, document_text=text, **labels)


# 1. The handbook: are the table rows, the list, the code and the warning kept whole?
print(f"Handbook, at most {SIZE} characters per chunk:")
chunkers = [("recursive", RecursiveChunker(SIZE, 0)), ("paragraph", ParagraphChunker(SIZE)),
            ("heading", HeadingChunker(SIZE)), ("markdown", MarkdownChunker(SIZE))]
for name, chunker in chunkers:
    chunks, scores = score(chunker, handbook, structural_blocks=gold["handbook_blocks"])
    table = [chunk for chunk in chunks if "| E-" in chunk]
    with_header = sum("| Code | Meaning |" in chunk for chunk in table)
    print(f"  {name:10} {len(chunks):2} chunks, "
          f"blocks kept whole {scores['structural_integrity']:.0%}, "
          f"table chunks with the header row: {with_header} of {len(table)}")

# 2. The meeting notes: do the cuts fall where the topic changes? (1.00 = all of them)
print("\nMeeting notes, no headings: boundary F1 against the 4 real topics")
setups = [("paragraph", ParagraphChunker(SIZE))] + [
    (f"texttiling, window {w}", TextTilingChunker(SIZE, window=w)) for w in (2, 3, 4)]
for name, chunker in setups:
    chunks, scores = score(chunker, notes, expected_chunks=gold["meeting_topics"])
    print(f"  {name:21} {len(chunks)} chunks, boundary F1 {scores['boundary_f1']:.2f}")

# 3. Search both files: is the whole answer inside one of the 2 chunks found?
questions = json.loads((HERE / "chunker_questions.json").read_text(encoding="utf-8"))
print(f"\nSearch ({len(questions)} questions): the whole answer in one of the top 2 chunks")
for name in ["recursive", "paragraph", "heading", "markdown", "texttiling"]:
    rag = RAGTitan(settings={"chunker": name, "chunk_size": SIZE, "chunk_overlap": 0,
                             "vector_store": "memory"})
    rag.index(sorted(DOCS.glob("*")))
    missed = [q["id"] for q in questions
              if not any(q["answer"] in found["text"]
                         for found in rag.parts["retriever"].retrieve(q["question"], 2))]
    print(f"  {name:10} {len(questions) - len(missed)} of {len(questions)}"
          + (f", missed {', '.join(missed)}" if missed else ""))
