---
hide:
  - toc
---

# HyDE RAG playground

**HyDE RAG** asks the model for a short **made-up answer** first, then searches
with it, because an answer reads more like the documents than a question does:

```
Question -> Write a hypothetical answer -> Search with it -> Search with the question
         -> Fuse the two lists (RRF) -> Rerank -> Generate the answer
```

The made-up text is only used to search: the reranker and the answer use your
question, and only real chunks reach the model. Move any knob below and the
question runs again; **click "Write hypothetical answer"** to read what the model
made up, and **"Fuse lists"** to see each chunk's rank in both searches.

<div id="rag-type-app" data-rag-type="hyde"><p><em>The playground needs JavaScript.</em></p></div>

## Things to try

Every result below is a real recorded run you can repeat with the knobs.

1. **Read what the model made up.** Pick *What is the maximum hotel cost per
   night?*, chunker `word`, retriever `hybrid`, and click **Write hypothetical
   answer**: the model wrote that the maximum is **$350** at "designated hotels".
   The policy says 150 euros. The made-up text is only used to search; it is never
   given to the model as a source.
2. **When fusing hurts.** Same setup, with *search the question too* **on** (the
   default). Both searches put a 150-euro chunk of `expenses.md` first, but not the
   same one: word chunks overlap, and chunk_1 (the made-up answer's #1) and chunk_2
   (the question's #1) share that sentence. After fusing, chunk_1 is dropped as a
   near-copy of chunk_2, and chunk_2, found near the top by the question's search
   only, comes 5th of 5 in *Fuse lists*: chunks found by **both** searches rank
   above it. The reranker keeps two others and the model says it does not know.
   Turn the knob **off**: the made-up answer's search alone puts chunk_1 first and
   the answer is *"Hotel costs are refunded up to 150 euros per night."* Reciprocal
   Rank Fusion rewards agreement between the lists, even over a clear first place
   in one of them.
3. **Keep each search's #1.** Same setup, knob back **on**, and set *keep each
   search's #1* **on**: nothing changes, because chunk_2 is already in the fused five
   and it is the reranker that drops it. Now set `top_k` to **2**: without the knob
   the two fused chunks are a `remote_work.md` and a `leave_policy.md` one and the
   model does not know; with it, chunk_2 is *"kept as #1 of query 2"* and the answer
   is *"hotel costs are refunded up to 150 euros per night"*. Over all 54 setups with
   `top_k` 2, keeping each search's #1 raised the setups with all the evidence from
   47 to 49 and lowered none.
4. **Weigh the question.** Pick *What do I get for working from home: equipment and
   money for furniture?*, chunker `sentence`, retriever `dense`, `top_k` **2**: the
   model gets the laptop chunk and the `expenses.md` chunk with the 300 euros. Set
   *question weight* to **2**: in *Fuse lists* the question's search (`query 2 ×2`)
   now decides more, a second `remote_work.md` chunk replaces the expenses one, and
   the answer has no money for furniture. Over the 54 setups with `top_k` 2 a weight
   of 2 lowered the setups with all the evidence from 47 to 44; 0.5 raised it to 49.
   The made-up answer can be the better search: measure before trusting either one.
5. **The big picture.** Over all 54 recorded setups (3 chunkers × 3 retrievers ×
   6 questions) the chunks given to the model held all the evidence in 45 for
   Simple RAG, 44 for HyDE with the question searched too, and 48 with the made-up
   answer alone. These short policies use the same words as the questions, so
   there is little for HyDE to fix, and six questions are too few to call a
   winner. HyDE is meant for questions worded far from the documents: measure it
   on yours with `rag-titan evaluate --rag-type simple hyde`.

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Chunker | `CHUNKER`, `CHUNK_SIZE` | how the documents are cut |
| Retriever | `RETRIEVER` | `dense`, `bm25` or `hybrid`; each search uses it |
| Reranker | `RERANKER` | orders the fused chunks against your question |
| top_k | `TOP_K` | chunks the fused search returns (default 5) |
| top_n | `TOP_N` | chunks the model reads (default 2) |
| search the question too | `HYDE_WITH_QUESTION` | on (default): the question's search is fused in, so a wrong guess cannot take over; off: the made-up answer alone, like the paper |
| question weight | `QUESTION_WEIGHT` | how much the question's search counts against the made-up answer's when fusing (default 1) |
| keep each search's #1 | `RRF_KEEP_TOP` | on: the best chunk of both searches stays in the fused `top_k` (default off) |
| vector / keyword search weight, keep each search's #1 | `HYBRID_DENSE_WEIGHT`, `HYBRID_BM25_WEIGHT`, `RRF_KEEP_TOP` | shown with retriever `hybrid`: how much each search counts when the two are fused (default 1), and keep each one's #1 (see [Retrieval](../retrieval.md#trusting-one-search-more-and-keeping-each-searchs-best)) |

## How the answers are made here

The searches, the fusing, the reranking and the scores run **live** in your
browser with the real rag-titan code; the model's replies (`gemma3:4b`) were
recorded once and are replayed. Replies were recorded with the defaults and with
one knob at a time on the values listed under it; the line under the knobs says
**● Answers recorded** when they replay. Elsewhere the run stops at the first model
step whose reply was not recorded, drawn dashed, and shows exactly what the model
would read. The code below the result runs the same setup in Python. The
[Compare page](../rag-playground.md) runs several RAG types side by side.
