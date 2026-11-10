# RAG types

## Supported RAG types

rag-titan **0.1** supports **2 RAG types**. Every one has the same method,
`answer(question)`, and with `RAGTitan` you pick one using `rag_type`.

| Status | RAG type | How it works | Best for | LLM calls |
|---|---|---|---|---|
| ✅ Available | [**Simple RAG**](#simple-rag-simple)<br>`rag_type="simple"` | Retrieve → Rerank → Generate | Fast answers to clear questions | 1 |
| ✅ Available | [**Adaptive RAG**](#adaptive-rag-adaptive)<br>`rag_type="adaptive"` | Route → Retrieve → Check evidence → Rewrite & retry | Mixed questions, fewer wrong answers | 2–5 |

```python
from rag_titan import RAG_TYPES, RAGTitan

print(RAG_TYPES)                      # ['simple', 'adaptive']
rag = RAGTitan(rag_type="adaptive")   # any value from RAG_TYPES
```

Asking for a type that does not exist yet raises a clear error:

```text
ValueError: Unknown RAG type: graph. Choose from ['simple', 'adaptive']
```

See [Planned RAG types](#planned-rag-types) for what is coming next.

## Simple RAG (`"simple"`)

```
Question -> Retrieve -> Rerank -> Generate
```

The classic RAG. It searches once and answers once.

![Simple RAG flow](assets/images/simple-rag.svg)

- **Good for:** clear factual questions, fast answers, learning how RAG works.
- **LLM calls per question:** 1
- **Options:** `top_k` (chunks to retrieve, default 5), `top_n` (chunks to keep, default 2)

```python
from rag_titan import SimpleRAG

rag = SimpleRAG(retriever, reranker, generator, top_k=5, top_n=2)
```

## Adaptive RAG (`"adaptive"`)

```
Question -> Choose route -> Retrieve -> Rerank -> Check evidence
                                                 |
                                 Enough evidence -> Answer
                                                 |
                             Not enough -> Rewrite and retry
```

It decides how much work each question needs:

![Adaptive RAG flow](assets/images/adaptive-rag.svg)

| Route     | When                           | What it does                     |
|-----------|--------------------------------|----------------------------------|
| `direct`  | greetings, thanks, goodbyes    | replies without searching        |
| `simple`  | one-topic factual question     | small search (top_k 4, top_n 2)  |
| `complex` | comparisons, multi-part        | bigger search (top_k 8, top_n 4) |

After searching, the LLM checks if the chunks really answer the question. If
not, the question is rewritten into a better search query and the search is
made wider. After `max_attempts` (default 2) it says it does not know.

- **Good for:** mixed questions, avoiding wrong answers.
- **LLM calls per question:** 2 (greeting) to 5 (one retry), with the default `max_attempts=2`
- **Parts:** `Router`, `EvidenceChecker`, `QueryRewriter`

## Comparing RAG types

```python
from rag_titan import RAG_TYPES, build_rag, build_rag_parts, get_settings

parts = build_rag_parts(get_settings())
parts["pipeline"].run(["handbook.pdf"])        # index once

for rag_type in RAG_TYPES:
    print(rag_type, build_rag(rag_type, parts).answer("How many leave days?"))
```

## Planned RAG types

These are **not available yet**. They will be added as new `rag_type` values,
so code that uses `RAGTitan` keeps working.

| Status | RAG type | Idea |
|---|---|---|
| 🔜 Planned | Corrective RAG (CRAG) | Grade each chunk and fall back to web search when they are weak |
| 🔜 Planned | Multi-query RAG | Ask several versions of the question and merge the results |
| 🔜 Planned | HyDE | Write a hypothetical answer first, then search with it |
| 🔜 Planned | Hybrid RAG | Combine keyword (BM25) and vector search |
| 🔜 Planned | Self-RAG | The model checks and improves its own answer |
| 🔜 Planned | Parent-document RAG | Search small chunks, answer with the bigger section around them |
| 🔜 Planned | Graph RAG | Search a knowledge graph built from your documents |
| 🔜 Planned | Agentic RAG | An agent decides when and what to search |

Want one sooner? Open an issue or read [Extending rag-titan](extending.md#adding-a-new-rag-type).
See the full [roadmap](roadmap.md).
