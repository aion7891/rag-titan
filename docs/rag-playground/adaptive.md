---
hide:
  - toc
---

# Adaptive RAG playground

**Adaptive RAG** decides how much work each question needs, and checks its own
evidence before answering:

```
Question -> Choose route -> Retrieve -> Rerank -> Check evidence
                                                   |-- enough      -> Answer
                                                   |-- not enough  -> Rewrite the question, search wider, check again
                                                   |-- out of tries -> "I don't know"
```

Move any knob below and the question runs again. The diagram lights up the steps
in the order they ran, with the loop in orange; **click a step** to see what went
in and what came out.

<div id="rag-type-app" data-rag-type="adaptive"><p><em>The playground needs JavaScript.</em></p></div>

## Things to try

Every result below is a real recorded run you can repeat with the knobs.

1. **A retry that rescues the answer.** Pick *What is the maximum hotel cost per
   night?* (chunker `paragraph`, retriever `dense`). The chunk with the answer
   (150 euros) is already first, but the model's evidence check says *no*: it is a
   small model's judgement. Adaptive RAG rewrites the question as *"Hotel price
   limits per night; maximum hotel rates; nightly hotel cost ceilings"*, searches
   wider (`top_k` 4 → 8), checks again, and answers in 9 steps: the line under the
   answer says *"loop: 2 of 2 rounds, stopped: evidence is good"*. Now set **max
   attempts** to **1**: no second chance, so it says it does not know after 5 steps
   (*"1 of 1 rounds, stopped: max steps"*).
2. **More chunks can hurt.** Same question, max attempts back to 2, `top_n`
   **3**: with the extra chunk the check says *no* twice and it gives up
   (*"2 of 2 rounds, stopped: max steps"*). More context is not always better for a
   small model.
3. **Words or meaning.** Pick *When do I need a note from the doctor?* with chunker
   `word`. Retriever `dense` answers in 5 steps. Retriever **`bm25`** finds only one
   chunk with the words *doctor's note*, and it starts mid-sentence, without *from
   the third sick day*, so the check says *no*. The rewritten question finds that
   chunk again plus an unrelated `remote_work.md` one, and Adaptive RAG gives up
   after 9 steps rather than guess.

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Chunker | `CHUNKER`, `CHUNK_SIZE` | how the documents are cut |
| Retriever | `RETRIEVER` | `dense`, `bm25` or `hybrid` |
| Reranker | `RERANKER` | `keyword`, `none`, `mmr` or `llm` |
| top_k | `TOP_K` | chunks searched for a *simple* question (default 4). A *complex* question searches 2 × top_k, and every retry adds top_k more |
| top_n | `TOP_N` | chunks kept the same way (default 2; 2 × for complex, + top_n per retry) |
| max attempts | `ADAPTIVE_MAX_ATTEMPTS` | searches before it gives up and says it does not know (default 2) |
| vector / keyword search weight, keep each search's #1 | `HYBRID_DENSE_WEIGHT`, `HYBRID_BM25_WEIGHT`, `RRF_KEEP_TOP` | shown with retriever `hybrid`: how much each search counts when the two are fused (default 1), and keep each one's #1 (see [Retrieval](../retrieval.md#trusting-one-search-more-and-keeping-each-searchs-best)) |

Adaptive RAG asks the model up to five times per question (route, check, rewrite,
check again, answer), so more of its steps depend on recorded replies than in
Simple RAG: a run past the recording can stop at *Check evidence* too. The search
steps before it are always live.

## How the answers are made here

The search, the reranking and the scores run **live** in your browser with the
real rag-titan code; the model's replies (`gemma3:4b`) were recorded once and are
replayed. Replies were recorded with the defaults and with one knob at a time on
the values listed under it; the line under the knobs says **● Answers recorded**
when they replay. Elsewhere the run stops at the first model step whose reply was
not recorded, drawn dashed. The code below the result runs the same setup in Python with your own
model. The [Compare page](../rag-playground.md) runs several RAG types side by side.
