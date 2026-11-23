---
hide:
  - toc
---

# Simple RAG playground

**Simple RAG** searches once and answers once:

```
Question -> Retrieve top_k chunks -> Rerank, keep top_n -> Generate the answer
```

Move any knob below and the question runs again. The diagram lights up the steps
in the order they ran; **click a step** to see what went in and what came out.

<div id="rag-type-app" data-rag-type="simple"><p><em>The playground needs JavaScript.</em></p></div>

## Things to try

Every result below is a real recorded run you can repeat with the knobs.

1. **Two files, two chunks.** Pick *What do I get for working from home: equipment
   and money for furniture?* (chunker `paragraph`, retriever `dense`). With
   `top_n` 2 only `remote_work.md` reaches the model: it lists the laptop, monitor
   and headset, and even says *"They do not provide money for furniture"*, because
   the expenses file never reached it (half the evidence). Set `top_n` to **3**:
   `expenses.md` comes in and the answer adds the 300 euros for a desk and chair.
2. **Let the model choose the chunks.** Same question, `top_n` back to 2, reranker
   **`llm`**. Click *Rerank*: the model rates a `remote_work.md` chunk and an
   `expenses.md` chunk 10/10 each, so both files reach the model with only two
   chunks, and all the evidence is found.
3. **Variety is not relevance.** Chunker `word`, retriever `hybrid`, same question.
   With reranker **`mmr`** the second chunk is a different one, a
   `leave_policy.md` chunk about the doctor's note, and the model answers *"I
   don't know"*. With **`llm`** both right files reach it and it answers.
4. **Too little context.** Chunker `word`, retriever `dense`, same question,
   `top_n` **1**: one chunk is not enough and the model says it does not know.
5. **Trust one search more.** Pick *When do I need a note from the doctor?*,
   chunker `word`, retriever **`hybrid`**: the hybrid knobs appear. The answer is
   *"From the third sick day you need a doctor's note"*. Set *keyword search weight*
   to **2**: in *Fuse ranks* the keyword search's votes count twice
   (`keyword ×2`), a neighbouring chunk replaces the one that says *third sick day*,
   and the answer becomes *"You need a doctor's note for sick leave"*. The evidence
   score stays 100%, because it counts files, not sentences. Over the 18 hybrid
   setups a weight of 2 reordered the fused chunks in 10 (vector) and 6 (keyword)
   but never changed which files the model got.

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Chunker | `CHUNKER`, `CHUNK_SIZE` | how the documents are cut: words, sentences or paragraphs |
| Retriever | `RETRIEVER` | `dense` finds meaning, `bm25` exact words, `hybrid` both (fused) |
| Reranker | `RERANKER` | `keyword` rewards shared words, `none` keeps the search order, `mmr` avoids near-copies, `llm` lets the model rate each chunk |
| top_k | `TOP_K` | how many chunks the search finds (default 5) |
| top_n | `TOP_N` | how many of them reach the model (default 2, at most top_k) |
| vector / keyword search weight, keep each search's #1 | `HYBRID_DENSE_WEIGHT`, `HYBRID_BM25_WEIGHT`, `RRF_KEEP_TOP` | shown with retriever `hybrid`: how much each search counts when the two are fused (default 1), and keep each one's #1 (see [Retrieval](../retrieval.md#trusting-one-search-more-and-keeping-each-searchs-best)) |

The code under the result builds the same setup in Python, ready to copy.

## How the answers are made here

The search, the reranking and the scores run **live** in your browser, for any
value, with the real rag-titan code. The model's answers are **real replies**
(`gemma3:4b`), recorded once and replayed. A reply exists only for the exact
prompt that was recorded, so:

- answers were recorded with the **defaults**, and with **one knob at a time** on
  the values listed under it (for example `top_k` 3 · 5 · 8). The line under the
  knobs says **● Answers recorded** when the real reply replays;
- change two knobs, or pick an unlisted value, and the run stops at *Generate
  answer*, drawn dashed, and shows **exactly what the model would read**, with its
  evidence score. Run that setup in Python (the code below the result) for the
  real answer.

The `cross_encoder` reranker needs PyTorch, which cannot run in a browser: try it in
Python with `pip install "rag-titan[rerank]"`. See also the
[Compare page](../rag-playground.md) to run several RAG types side by side.
