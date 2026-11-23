---
hide:
  - toc
---

# Corrective RAG playground

**Corrective RAG** grades every chunk the search found before answering, and
repairs the evidence when it is weak (inspired by CRAG, Yan et al. 2024):

```
Question -> Retrieve -> Grade each chunk (0-10)
   correct   (a chunk >= upper)        -> keep the good chunks
   ambiguous (in between)              -> keep the good chunks, rewrite, search again
   incorrect (every chunk < lower)     -> throw them away, rewrite, search again
-> Rerank -> Keep key sentences -> Generate the answer
   (nothing good after the retries   -> "I don't know")
```

Move any knob below and the question runs again. **Click "Grade chunks"** to see
the model's grade for every chunk and the verdict, and **"Keep key sentences"**
to see which sentences of each chunk the model was given. The line under the
answer says how the loop ended: *evidence is good*, *no new chunks* or *max steps*.

<div id="rag-type-app" data-rag-type="corrective"><p><em>The playground needs JavaScript.</em></p></div>

## Things to try

Every result below is a real recorded run you can repeat with the knobs.

1. **Grades pick the right files.** Pick *What do I get for working from home:
   equipment and money for furniture?*, chunker `sentence`, retriever `dense`, and
   click **Grade chunks**: the model graded the `expenses.md` chunk about furniture
   **10**, so it reaches the answer: *"You get a laptop, a monitor, and a headset
   ... You can also claim up to 300 euros once for a desk and chair"*. Simple RAG's
   keyword reranker keeps two `remote_work.md` chunks instead, and its answer ends
   with *"I don't know"*. Over all 54 recorded setups (3 chunkers × 3 retrievers ×
   6 questions) the model got all the evidence in **51** with Corrective RAG and
   45 with Simple RAG; the six won are this question with `sentence` and `word`
   chunks.
2. **A small model's grades are rough.** Pick *What is the maximum hotel cost per
   night?*, chunker `paragraph`, retriever `dense`: the chunk that answers it is
   graded only **5**, so the evidence is *ambiguous*. The rewritten question finds
   only a chunk graded 0, the loop stops at *max steps*, and the answer still comes
   from the kept chunk (5 is above the lower grade, 3): *"Hotel costs are refunded
   up to 150 euros per night."* Set **upper grade** to 5: the evidence is *correct*
   at once, the same answer comes in 5 steps instead of 8, with two model calls
   fewer.
3. **Searching again rarely paid off here.** The first grades said *correct* in 38
   setups and *ambiguous* in 16, never *incorrect*; 15 of the 16 searches again
   found nothing better. With **max retries** 0 Corrective RAG still gets 51 of 54,
   so on these short policies, which use the same words as the questions, the win
   comes from the grades, not from searching again. **Keep key sentences** changes
   the wording of 47 of the 54 answers but not which of them hold the evidence.
   Searching again matters when the first search misses: measure it on your own
   documents with `rag-titan evaluate --rag-type simple corrective`.

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Chunker | `CHUNKER`, `CHUNK_SIZE` | how the documents are cut |
| Retriever | `RETRIEVER` | `dense`, `bm25` or `hybrid`; the first search and every search again use it |
| Reranker | `RERANKER` | orders the kept chunks against your question |
| top_k | `TOP_K` | chunks each search finds (default 5) |
| top_n | `TOP_N` | chunks the model reads (default 2) |
| upper grade | `CRAG_UPPER` | a chunk graded this or more makes the evidence *correct* (default 7) |
| lower grade | `CRAG_LOWER` | every chunk graded below this makes it *incorrect*; chunks below it are never kept (default 3) |
| keep useful sentences | `CRAG_REFINE` | on (default): the model reads only the sentences that help, word for word |
| max retries | `CRAG_MAX_RETRIES` | times to search again with a rewritten question; 0 grades only (default 1) |
| vector / keyword search weight, keep each search's #1 | `HYBRID_DENSE_WEIGHT`, `HYBRID_BM25_WEIGHT`, `RRF_KEEP_TOP` | shown with retriever `hybrid`: how much each search counts when the two are fused (default 1), and keep each one's #1 (see [Retrieval](../retrieval.md#trusting-one-search-more-and-keeping-each-searchs-best)) |

## How the answers are made here

The searches, the reranking and the scores run **live** in your browser with the
real rag-titan code; the model's replies (`gemma3:4b`: the grades, the rewritten
question, the kept sentences and the answer) were recorded once and are replayed.
Replies were recorded with the defaults and with one knob at a time on the values
listed under it; the line under the knobs says **● Answers recorded** when they
replay. Elsewhere the run stops at the first model step whose reply was not
recorded, drawn dashed, and shows exactly what the model would read. The code
below the result runs the same setup in Python. The
[Compare page](../rag-playground.md) runs several RAG types side by side.
