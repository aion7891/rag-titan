---
hide:
  - toc
---

# Iterative RAG playground

**Iterative RAG** searches, asks the model what is still missing, and searches
again with the model's own query (inspired by IRCoT, Trivedi et al. 2022). It is
meant for questions where the first search shows what to look up next:

```
Question -> Retrieve -> Rerank -> Think: what is missing?
                                    SEARCH: <next query> -> Retrieve -> Rerank -> Think ...
                                    DONE                 -> Generate (from everything found)
```

Every search keeps its own best `top_n` chunks, and the model answers from all of
them. The loop stops when the model says DONE, when a search finds nothing new,
or after `max steps` rounds. Move any knob below and the question runs again;
**click "Think: what next?"** to read the model's thought and its next search.
The line under the answer says how the loop ended.

<div id="rag-type-app" data-rag-type="iterative"><p><em>The playground needs JavaScript.</em></p></div>

## Things to try

Every result below is a real recorded run you can repeat with the knobs.

1. **A second search for the missing half.** Pick *What do I get for working from
   home: equipment and money for furniture?*, chunker `paragraph`, retriever
   `dense`, and click **Think: what next?**: after the first search the model wrote
   *"... do not specify whether they offer money for furniture"* and searched
   **money for furniture**, which found `expenses.md`. Over all 54 recorded setups
   (3 chunkers × 3 retrievers × 6 questions) the model got all the evidence in
   **51** with Iterative RAG and 45 with Simple RAG; the six won are all this
   question.
2. **Evidence found is not an answer given.** In that same run the answer still
   ends with *"I don't know"*: the furniture sentence is inside a long expenses
   paragraph and the small model misses it. With retriever `bm25` it even says
   *"You do not get money for furniture"*, which is wrong. Finding the evidence
   and reading it well are two different problems.
3. **The loop limits do the stopping.** The model wrote DONE in only 3 of the 54
   runs; in the other 51 the loop stopped because a search found nothing new. In 9
   runs the model even asked for a search it had already made, although it sees its
   earlier searches (the run above searched *money for furniture* twice). Without
   the no-progress rule those runs would have spent every round of the cap on
   searches that find nothing new. Set **max steps** to 0: it is Simple RAG again
   (45 of 54).

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Chunker | `CHUNKER`, `CHUNK_SIZE` | how the documents are cut |
| Retriever | `RETRIEVER` | `dense`, `bm25` or `hybrid`; every search uses it |
| Reranker | `RERANKER` | orders each search's chunks against that search's own query |
| top_k | `TOP_K` | chunks each search finds (default 3) |
| top_n | `TOP_N` | chunks kept from each search (default 2); the model reads all the kept ones |
| max steps | `ITERATIVE_MAX_STEPS` | think-then-search rounds after the first search; 0 searches once, like Simple RAG (default 3) |
| vector / keyword search weight, keep each search's #1 | `HYBRID_DENSE_WEIGHT`, `HYBRID_BM25_WEIGHT`, `RRF_KEEP_TOP` | shown with retriever `hybrid`: how much each search counts when the two are fused (default 1), and keep each one's #1 (see [Retrieval](../retrieval.md#trusting-one-search-more-and-keeping-each-searchs-best)) |

## How the answers are made here

The searches, the reranking and the scores run **live** in your browser with the
real rag-titan code; the model's replies (`gemma3:4b`: each thought and next
search, and the answer) were recorded once and are replayed. Replies were
recorded with the defaults and with one knob at a time on the values listed under
it; the line under the knobs says **● Answers recorded** when they replay.
Elsewhere the run stops at the first model step whose reply was not recorded,
drawn dashed, and shows exactly what the model would read. The code below the
result runs the same setup in Python. The [Compare page](../rag-playground.md)
runs several RAG types side by side.
