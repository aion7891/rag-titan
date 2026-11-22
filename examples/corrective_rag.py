"""
Corrective RAG with every knob set and explained: the model grades every chunk
the search found (0-10), and when the evidence is weak the question is rewritten
and searched again. Only the useful sentences of the good chunks reach the answer.

The question uses other words than the documents ("gear", "newcomer" for
"equipment", "new employees"), and keyword search (bm25) only finds shared words,
so the first search finds nothing and the evidence is graded "incorrect".

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/corrective_rag.py
"""

from pathlib import Path

from rag_titan import RAGTitan

files = sorted((Path(__file__).parent / "retrieval_docs").glob("*.md"))
question = "What gear does a newcomer receive?"

settings = {
    # how the files are cut (CHUNKER, CHUNK_SIZE, CHUNK_OVERLAP)
    "chunker": "paragraph",
    "chunk_size": 300,
    "chunk_overlap": 0,
    # how chunks are found and chosen (RETRIEVER, RERANKER, TOP_K, TOP_N)
    "retriever": "bm25",       # exact words: the rewritten question brings other words
    "reranker": "keyword",     # orders the kept chunks against the original question
    "top_k": 5,                # chunks each search finds
    "top_n": 2,                # chunks the model reads
    # Corrective RAG's own knobs
    "crag_upper": 7,           # CRAG_UPPER: a chunk graded 7+ makes the evidence "correct"
    "crag_lower": 3,           # CRAG_LOWER: all chunks below 3 make it "incorrect"
    "crag_refine": True,       # CRAG_REFINE: give the model only the useful sentences
    "crag_max_retries": 1,     # CRAG_MAX_RETRIES: times to search again (0: grade only)
}

rag = RAGTitan(rag_type="corrective", settings=settings)
rag.index(files)
result = rag.run(question)

print(f"Question: {question}")
for step in result["steps"]:
    if step["step"] == "grade":
        grades = ", ".join(f"{chunk['source']} {chunk['grade']:g}"
                           for chunk in step["output"]["chunks"] if "grade" in chunk)
        print(f"Grade: {step['output']['verdict']} ({grades or 'nothing found'})")
    if step["step"] == "rewrite":
        print(f"Rewritten question (only used to search): {step['output']}")
for chunk in result["sources"]:
    print(f"The model reads from {chunk['source']} ({chunk.get('refine_note', 'whole chunk')}): "
          f"{' '.join(chunk['text'].split())}")
print(f"Loop: {result['loop']}")
print(f"Answer: {result['answer']}")
