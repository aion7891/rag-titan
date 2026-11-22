"""
Iterative RAG with every knob set and explained: search, let the model say what
is still missing, search again with the model's own query, then answer from
everything found.

A two-hop question: offices.md says Berlin hardware requests go to the
Facilities team, and only teams.md says who leads that team. The question shares
no word with teams.md, so one search cannot find it.

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/iterative_rag.py
"""

from pathlib import Path

from rag_titan import RAGTitan

files = sorted((Path(__file__).parent / "multihop_docs").glob("*.md"))
question = "Who has to approve a new monitor for a Berlin employee?"

settings = {
    # how the files are cut (CHUNKER, CHUNK_SIZE, CHUNK_OVERLAP)
    "chunker": "paragraph",
    "chunk_size": 300,
    "chunk_overlap": 0,
    # how chunks are found and chosen (RETRIEVER, RERANKER, TOP_K, TOP_N)
    "retriever": "bm25",       # exact words: the second hop needs the model's own query
    "reranker": "keyword",     # ranks each search's chunks against that search's query
    "top_k": 3,                # chunks each search finds
    "top_n": 2,                # chunks kept from each search; the model reads all of them
    # Iterative RAG's own knob (ITERATIVE_MAX_STEPS): think-then-search rounds
    # after the first search; 0 searches once, like Simple RAG
    "iterative_max_steps": 3,
}

rag = RAGTitan(rag_type="iterative", settings=settings)
rag.index(files)
result = rag.run(question)

print(f"Question: {question}")
searches = [step for step in result["steps"] if step["step"] == "retrieve"]
for number, search in enumerate(searches, start=1):
    found = sorted({chunk["source"] for chunk in search["output"]})
    print(f"Search {number}: {search['input']['query']!r} finds {', '.join(found) or 'nothing'}")
for step in result["steps"]:
    if step["step"] == "think":
        decision = step["output"]["search"] or "DONE"
        print(f"Think: {step['output']['thought']} -> {decision}")
print(f"The model reads: {', '.join(chunk['source'] for chunk in result['sources'])}")
print(f"Loop: {result['loop']}")
print(f"Answer: {result['answer']}")
