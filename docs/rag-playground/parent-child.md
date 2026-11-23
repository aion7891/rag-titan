---
hide:
  - toc
---

# Parent-Child RAG playground

**Parent-Child RAG** searches small chunks, which match a question precisely,
then gives the model the **whole section** each good chunk sits in:

```
Question -> Retrieve small children (top_k) -> Rerank, keep top_n
         -> Expand each child to its parent section (each section once, within the budget)
         -> Generate the answer from the sections
```

Move any knob below and the question runs again. **Click "Expand to parents"** to
see which children were found, which section each one belongs to, and the text the
model is given.

<div id="rag-type-app" data-rag-type="parent_child"><p><em>The playground needs JavaScript.</em></p></div>

## Things to try

Every result below is a real recorded run you can repeat with the knobs. The
sample policies have short sections (130 to 300 characters), so here a matched
child usually holds the fact already and its section adds the rest around it; the
[Parent-Child example](../rag-types.md#parent-child-rag-parent_child) shows a longer document where only the
section holds the answer.

1. **See the expansion.** Chunker `word`, question *When do I need a note from the
   doctor?*, then click **Expand to parents**: the 4 children found become 4
   sections, each with the child that led to it and its length; the *Sick leave*
   section (178 characters) is the one the model answers from.
2. **Two files with the defaults.** Question *What do I get for working from home:
   equipment and money for furniture?*, chunker `paragraph`, retriever **`bm25`**.
   The 4 children come from 4 sections, the *Home office* section of
   `expenses.md` among them, and all fit in the 3000-character budget: the answer
   has the laptop, monitor and headset **and** the 300 euros. Simple RAG with its
   defaults gets only `remote_work.md` for this question, partly because it keeps
   2 chunks where Parent-Child keeps 4.
3. **The budget trade-off.** Same setup, **parent budget 600**: the first three
   sections (171 + 173 + 180 = 524 characters) fit, the *Home office* section would
   pass 600, so it is left out and the model says it does not know. At **300** only
   one section is left. Whole sections cost prompt space: the budget decides how
   many fit.
4. **Smaller parents.** Same setup, **parent size 200**: the *Leave policy* intro
   splits into two parents, the search finds a different set of children, the
   *Home office* section is no longer among them, and the answer covers only the
   equipment.

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Child chunker | `CHUNKER`, `CHUNK_SIZE` | how the small, searched children are cut |
| Parent size | `PARENT_SIZE` | the largest parent section in characters, used when indexing (default 1500) |
| Retriever | `RETRIEVER` | `dense`, `bm25` or `hybrid` |
| Reranker | `RERANKER` | orders the children before they are expanded |
| top_k | `TOP_K` | children searched (default 8: children are small, so take more) |
| top_n | `TOP_N` | children kept (default 4); children of the same section become one parent |
| parent budget | `PARENT_BUDGET` | most characters of section text given to the model (default 3000). The first section is always kept |
| vector / keyword search weight, keep each search's #1 | `HYBRID_DENSE_WEIGHT`, `HYBRID_BM25_WEIGHT`, `RRF_KEEP_TOP` | shown with retriever `hybrid`: how much each search counts when the two are fused (default 1), and keep each one's #1 (see [Retrieval](../retrieval.md#trusting-one-search-more-and-keeping-each-searchs-best)) |

The budget is the handbook's warning made into a knob: whole sections can make the
prompt long ("token inflation"), so sections are added best first until the
budget is used up.

## How the answers are made here

Search, reranking and the expansion to parents run **live** in your browser with
the real rag-titan code; the model's answers (`gemma3:4b`) were recorded once and
are replayed. Answers were recorded with the defaults and with one knob at a time
on the values listed under it (parent size 1500 or 200); the line under the knobs
says **● Answers recorded** when they replay. Elsewhere the run stops at *Generate
answer*, drawn dashed, and shows exactly the sections the model would read. The code below the result builds the same setup in
Python. The [Compare page](../rag-playground.md) runs several RAG types side by side.
