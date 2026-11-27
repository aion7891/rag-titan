# Changelog

All notable changes to this project are written here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.15.3] - 2026-11-27

Open Knowledge Format (OKF) bundles: check them, and index them so answers come
from current knowledge and show how far each passage can be trusted.

### Added
- **Read OKF bundles** (`rag_titan.okf`, OKF v0.3; v0.2 and unversioned bundles
  are read with the spec's fallback): a folder of Markdown files with YAML
  frontmatter, or a `.zip` of one. Concepts, skipped files, assets, reserved
  files (`index.md`, `log.md`) and the bundle metadata (`okf_version`, `title`,
  `description`, `license`) are sorted as the spec says. Reading is safe: dot
  paths are skipped, a symbolic link or zip entry that leads out of the bundle is
  never read, and YAML follows the 1.2 rule for booleans (`on`, `yes`, `no` stay
  text) and refuses duplicate keys.
- **Trust, freshness and lifecycle:** trust tiers (unverified, machine-confirmed,
  human-reviewed) from current verification events only, so a tier drops when the
  content changed after a review; staleness from `stale_after` (a value that cannot
  be read is never fresh); status, and `superseded_by` followed to the concept that
  replaced an old one. Links and path fields resolve from the bundle root or the
  file's folder, never outside the bundle.
- **`rag-titan okf validate`** (errors and warnings; exit 1 on errors, `--strict`
  for warnings too, `--json`) and **`rag-titan okf info`** (concepts by type,
  trust tier, status; stale; broken links). In Python: `validate_bundle()`.
- **Index a bundle:** give its folder or `.zip` to `index()` or `--files`. Each
  concept becomes a document, and every chunk keeps the concept's facts in
  `chunk["metadata"]` (also in a saved index). `OKF_INCLUDE=current` (default)
  leaves out deprecated concepts; `all` keeps every one. `OKF_MIN_TRUST` leaves out
  concepts below a trust tier. CLI: `--okf-include`, `--okf-min-trust`. The model
  sees each OKF passage's trust tier, status and staleness in its source line.
- `pip install "rag-titan[okf]"` (PyYAML).
- `examples/okf_rag.py` with an example bundle (`examples/okf_bundle/`) where three
  policies changed and the old versions stay deprecated. Guide:
  [OKF knowledge bundles](https://aion7891.github.io/rag-titan/okf/). With
  `gemma3:4b`, three runs: with `OKF_INCLUDE=current` 12 of 12 answers gave the
  current value; with `all` the old version was always among the passages read,
  and 5 of 12 answers also gave the old value.

### Changed
- A document's `metadata` is copied onto each of its chunks (every chunker,
  parent-child too). Documents without metadata give exactly the same chunks.
- A folder or `.zip` in the files to index is read as an OKF bundle.

## [0.15.2] - 2026-11-25

More model providers through one built-in client. Ollama and Claude work as before.

### Added
- **OpenAI-compatible providers** for answers, agents and embeddings, with no SDK
  and nothing extra to install: `LLM_PROVIDER` / `EMBED_PROVIDER` accept `openai`,
  `gemini`, `openrouter`, `huggingface` (`hf`, chat only), `lmstudio`, `vllm`, and
  `openai_compatible` for any other server (`LLM_BASE_URL`, `EMBED_BASE_URL`).
  Keys come from `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`,
  `HF_TOKEN`, or `LLM_API_KEY` / `EMBED_API_KEY`. Default models (when `LLM_MODEL`
  / `EMBED_MODEL` are unset): `gpt-5-mini` and `text-embedding-3-small` (OpenAI),
  `gemini-3.7-flash` and `gemini-embedding-001` (Gemini), `openai/gpt-5-mini` and
  `openai/text-embedding-3-small` (OpenRouter), `openai/gpt-oss-120b` (Hugging
  Face). [Setup](https://aion7891.github.io/rag-titan/configuration/#openai-gemini-openrouter-hugging-face-and-local-servers).
- `OpenAICompatibleLLM` (`rag_titan.llm`): Chat Completions with tool calls, so
  agents work; clear errors for a missing key (naming the variable), a reply
  without an answer, and a reply cut off at the length limit.
- `OpenAICompatibleEmbedder` (`rag_titan.embeddings`): batches, vectors put in
  order by their index, and a check that every vector has the same size.
- `rag_titan.providers`: the providers, their addresses, key variables and
  default models in one place.
- Checked against a real server: pointed at Ollama's own OpenAI-compatible
  endpoint, chat, agent tool calls and RAG answers work, and search found the same
  top-3 chunks for every help-desk question as the native Ollama client (recall@3
  1.000, MRR 0.917 for both).

### Changed
- **API keys never enter the settings.** They are read from the environment only
  when a client is made, so they cannot appear in reports, traces or logs.
- `EMBED_MODEL` now defaults to the embedding provider's own model
  (`nomic-embed-text` for Ollama, as before). In `.env.example` it is commented
  out, so switching `EMBED_PROVIDER` alone is enough.
- An unknown provider name lists every provider.

## [0.15.1] - 2026-11-23

Two chunkers that need no model. Nothing changes for the existing chunkers.

### Added
- **Markdown chunker** (`CHUNKER=markdown`, `MarkdownChunker`): element-aware
  chunking (chunking handbook 3.5 and 3.7). It reads Markdown as blocks
  (paragraphs, lists, tables, code blocks, quotes such as `> **Warning**`) and packs
  whole blocks per heading section, with the heading path in front. A code block
  with blank lines, a table row or a list item is never cut, and a lead-in line
  ("You get:") stays with the list, table or code after it. A block too big for a
  chunk is cut at its own lines; a big table gets chunks of its own that each
  start with the lead-in and the header row (the offsets point at the rows).
- **TextTiling chunker** (`CHUNKER=texttiling`, `TextTilingChunker`): cuts text
  without headings where the topic changes, from the words alone (Hearst 1997;
  handbook 5.1). The words of `TEXTTILING_WINDOW` sentences (default 3) on each
  side of every gap are compared (cosine, BM25's stopwords left out); it cuts at
  the deepest valleys (Hearst's cutoff, mean − std / 2), moves a cut onto a
  nearby paragraph break, and packs each topic on its own, so overlap never
  crosses a topic change. `tiling()` shows the score and depth of every gap.
  Simplified from the paper: whole sentences, no smoothing (measured: smoothing
  moved most cuts by one sentence), cuts only at valleys.
- `TEXTTILING_WINDOW` setting and `--texttiling-window` (`ask`, `chat`, `index`,
  `evaluate`); `evaluate --chunker` accepts `markdown` and `texttiling`.
- Chunk playground: `markdown` and `texttiling` modes, two example texts, a window
  slider and the topics found; added text (heading path, repeated table header)
  is shown in grey.
- `examples/compare_chunkers.py` with a Markdown handbook and meeting notes
  (`examples/structured_docs/`), gold blocks and topics, and eight questions.
  Guide with the real run:
  [Chunking methods](https://aion7891.github.io/rag-titan/chunking/#a-real-comparison).
  At 300 characters per chunk, `markdown` kept all 13 blocks whole (10 table rows,
  the list, the code and the warning) and repeated the table header in 5 of 5
  table chunks (`recursive` 92% and 1 of 4, `paragraph` and `heading` 69% and
  1 of 3); with
  `nomic-embed-text` it found the whole answer for 8 of 8 questions in the top 2
  chunks, like `recursive` (7 of 8 for `paragraph`, `heading` and `texttiling`,
  which cut a table row in half). On the meeting notes TextTiling scored a
  boundary F1 of 1.00 with window 4 but 0.29 with the default 3 (`paragraph`
  0.00): on short texts, try a few windows.
- A test runs every chunk playground chunker and its Python twin on 407
  generated texts and requires the same chunks (needs Node; skipped without it).

### Changed
- `HeadingChunker` has a `section_packer()` hook (what packs one section's text);
  its chunks are unchanged.

## [0.15.0] - 2026-11-23

The first of the 0.15.x series. Every new setting is off by default: with the
defaults, every search, answer and recorded playground run is the same as in 0.14.0.

### Added
- **Weighted RRF.** Trust one search more when result lists are fused: a list's
  votes count `weight / (k + rank)`. `HYBRID_DENSE_WEIGHT` and `HYBRID_BM25_WEIGHT`
  (default 1.0) for hybrid search; `QUESTION_WEIGHT` (default 1.0) for the
  question's own search in HyDE and RAG-Fusion. In Python:
  `HybridRetriever(..., dense_weight=, bm25_weight=)`,
  `HyDERAG(..., question_weight=)`, `RAGFusionRAG(..., question_weight=)`, and
  `rrf_fuse` / `fuse_ranked(..., weights=)`, `search_and_fuse(..., weights=)`.
- **Keep each search's #1** (`RRF_KEEP_TOP`, default off): the best chunk of every
  fused search always stays in the top results, for hybrid search, HyDE and
  RAG-Fusion. A chunk kept this way says so in its `score_trace`
  (`"kept_as_top_of": ["query_2"]`).
- CLI flags `--hybrid-dense-weight`, `--hybrid-bm25-weight`, `--question-weight`
  and `--rrf-keep-top` / `--no-rrf-keep-top` (`ask`, `chat`, `evaluate`). They
  change only the search, so `evaluate` never re-indexes for them.
- `rag_titan.retrieval.fusion_options`: `check_weight`, `check_weights`,
  `keep_list_tops`.
- Playground: question weight and keep-each-search's-#1 knobs on the HyDE and
  RAG-Fusion pages; vector and keyword search weights on every page while the
  retriever is `hybrid`. The steps show a list's weight and why a chunk was kept
  (`query 1 ×2 #1`, `kept as #1 of query 4`).
- Guides with real numbers:
  [Retrieval](https://aion7891.github.io/rag-titan/retrieval/#trusting-one-search-more-and-keeping-each-searchs-best)
  and [RAG types](https://aion7891.github.io/rag-titan/rag-types/#tuning-the-fusion).
  On the playground's 54 recorded setups per type, keeping each search's #1 with
  `top_k` 2 raised the setups that found all the evidence from 47 to 49 (HyDE) and
  from 50 to 53 (RAG-Fusion) and lowered none; with the default `top_k` 5 those
  counts did not change. `QUESTION_WEIGHT=2` lowered HyDE's count from 47 to 44,
  and on the help desk example `HYBRID_DENSE_WEIGHT=3` lost one error-code
  question (Recall@1 1.000 → 0.833): measure before changing a weight.

### Changed
- **Adaptive RAG** runs its retries under the same `LoopLimit` as Corrective and
  Iterative RAG, so `result["loop"]` says how many searches ran and why they
  stopped (`evidence is good` or `max steps`); its playground page shows the loop.
  The steps and model calls are unchanged: all 540 recorded Adaptive runs replay
  exactly as before.

## [0.14.0] - 2026-11-23

### Added
- **Corrective RAG** (`rag_type="corrective"`): the model grades every retrieved
  chunk from 0 to 10 in one call. *Correct* evidence (a chunk at least
  `CRAG_UPPER`, default 7) is kept; *ambiguous* evidence keeps its good chunks and
  is searched again with a rewritten question; *incorrect* evidence (every chunk
  below `CRAG_LOWER`, default 3) is thrown away and searched again. Only the
  useful sentences of the chunks the model reads are kept, word for word
  (`CRAG_REFINE`, default on), so citations still point to real text. At most
  `CRAG_MAX_RETRIES` searches again (default 1); with nothing good left the answer
  is "I don't know". `CorrectiveRAG(..., fallback_retriever=...)` searches
  somewhere else when the evidence is weak (for example your own web search).
  Inspired by Yan et al. (2024).
- **Iterative RAG** (`rag_type="iterative"`): search, let the model write what is
  still missing and its next search query (or DONE), search again, and answer once
  from everything found. Every search keeps its own best `TOP_N` chunks, so a
  second hop worded unlike the question is not dropped. At most
  `ITERATIVE_MAX_STEPS` rounds (default 3). Inspired by IRCoT (Trivedi et al. 2022).
- **Loop limits** (`rag_titan.rag_types.loops.LoopLimit`): both types cap their
  rounds, stop when a search finds no new chunk, and say why they stopped in
  `result["loop"]` (`max steps`, `no new chunks`, `model said done`,
  `evidence is good`).
- Settings, `.env` and CLI flags: `CRAG_UPPER`, `CRAG_LOWER`, `CRAG_REFINE`,
  `CRAG_MAX_RETRIES`, `ITERATIVE_MAX_STEPS` (`--crag-upper`, `--crag-lower`,
  `--crag-refine`/`--no-crag-refine`, `--crag-max-retries`,
  `--iterative-max-steps`). They change only the search, so `evaluate` never
  re-indexes for them.
- Playground pages for both types with their knobs: the grade of every chunk, the
  sentences kept, each thought and next search, and how the loop ended.
- `examples/corrective_rag.py`, `examples/iterative_rag.py`, and two-hop help desk
  documents (`examples/multihop_docs/`, `examples/multihop_questions.json`). Guide
  with a real comparison:
  [RAG types](https://aion7891.github.io/rag-titan/rag-types/#corrective-rag-corrective).
  On the two-hop questions with keyword search, Iterative RAG got every needed file
  (recall 1.000, Simple RAG 0.700); on the help desk questions with keyword search,
  Corrective RAG lifted recall from 0.500 to 0.833.

### Changed
- Flow diagrams route every arrow that skips a row around a side, so it never
  crosses a box. The diagram code is split into `rag-flow.js` (layout, drawing)
  and `rag-flow-play.js` (lighting up a run); each RAG type's playground knobs live
  in `rag-knobs.js`.
- Playground runs include `"loop"` (`None` for RAG types without a loop).

## [0.13.0] - 2026-11-22

### Added
- **HyDE RAG** (`rag_type="hyde"`): the model writes a short made-up answer, which
  is searched together with the question; the two result lists are fused (RRF) and
  the answer comes from the real chunks only. `HYDE_WITH_QUESTION` (default on;
  off searches with the made-up answer alone, like the paper). Inspired by Gao et
  al. (2022).
- **RAG-Fusion** (`rag_type="rag_fusion"`): the model writes `FUSION_QUERIES`
  rewordings (default 3) in one call; the question and every rewording are
  searched and the lists fused with Reciprocal Rank Fusion. The model's reply is
  read robustly (numbering, bullets, intro lines, duplicates, copies of the
  question), and an unusable reply falls back to the question. Inspired by
  Raudaschl (2023).
- Both work with every retriever and reranker, `TOP_K` and `TOP_N`, and on the
  command line (`--rag-type hyde rag_fusion`, `--no-hyde-with-question`,
  `--fusion-queries`). Guide with a real comparison:
  [RAG types](https://aion7891.github.io/rag-titan/rag-types/#hyde-rag-hyde). On
  the help desk example with keyword search, both found the right files for 0.833
  of the questions, Simple RAG for 0.500.
- `rag_titan.retrieval.fusion`: `rrf_fuse`, `fuse_ranked` and `search_and_fuse`
  (search several queries, fuse with RRF); `QueryTransformRAG`, the shared base
  for RAG types that search with queries made from the question.
- Playground pages for HyDE RAG and RAG-Fusion with their knobs; fused chunks show
  their rank for every query. `examples/hyde_rag.py` and
  `examples/rag_fusion_rag.py`.
- Recording the playground again reuses the replies and vectors already recorded
  (`record_traces.py --fresh` asks the model for everything), so adding a RAG type
  never changes the runs recorded before. `RecordingLLM(..., known=...)`,
  `RecordingEmbedder(..., known=...)`, `record_presets(..., earlier=...)`.

### Changed
- Hybrid search's RRF moved to `rag_titan.retrieval.fusion`; its results are
  exactly the same.
- Flow diagrams draw a step that repeats as a small loop with its step numbers
  ("2–5"), and an arrow that skips a row around the side.

## [0.12.1] - 2026-11-22

### Added
- **One install for every feature**: `pip install "rag-titan[all]"` installs the
  `tokens`, `chroma`, `qdrant` and `rerank` extras together. The README and
  [Getting started](https://aion7891.github.io/rag-titan/getting-started/) list what
  each extra adds, with the CPU-only PyTorch command for `rerank`. A test makes sure
  every new extra is also part of `all`.

## [0.12.0] - 2026-11-22

### Added
- **Parent-Child RAG** (`rag_type="parent_child"`): searches small chunks, then gives
  the model the whole section each one sits in. Sections are used once each and
  added best first within `PARENT_BUDGET` characters. Guide with a real run:
  [RAG types](https://aion7891.github.io/rag-titan/rag-types/#parent-child-rag-parent_child).
  On its example (six steps in one section) Simple RAG listed 1 step and
  Parent-Child RAG all 6, with the same piece size, search and `top_n`.
- `ParentChildChunker` in `rag_titan.chunking`: parents are heading sections (or
  paragraph groups) of at most `PARENT_SIZE` characters, children are cut inside
  them with any chunker; each child keeps its parent's id, text and place.
  `RAGTitan(rag_type="parent_child")` and `rag-titan index --rag-type parent_child`
  index this way; `rag-titan evaluate` gives Parent-Child RAG an index of its own.
- **Two rerankers**: `RERANKER=llm` (the chat model rates every chunk 0-10 in one
  call; chunks are fenced as untrusted data; an unreadable reply keeps the search
  order and says so) and `RERANKER=cross_encoder` (a cross-encoder model,
  `pip install "rag-titan[rerank]"`, `RERANK_MODEL`, scores mapped to 0-1).
  `rag-titan evaluate --reranker ...` compares them.
- **Every RAG knob is a setting**: `TOP_K`, `TOP_N` (empty: each type's own
  defaults), `ADAPTIVE_MAX_ATTEMPTS`, `PARENT_SIZE`, `PARENT_BUDGET`, in `.env`,
  `RAGTitan(settings={...})` and on the command line (`--top-k`, `--top-n`,
  `--max-attempts`, `--parent-size`, `--parent-budget`, and `--reranker` on `ask`
  and `chat`). Guide: [Tuning a RAG type](https://aion7891.github.io/rag-titan/configuration/#tuning-a-rag-type).
- **A playground page per RAG type** (Simple, Adaptive, Parent-Child) with tuning
  knobs, its own diagram, "Things to try" and the same setup as Python code.
  Search, reranking and parents run live for any value; answers replay for the
  recorded settings (the defaults and one knob at a time), and elsewhere the run
  stops at the model step and shows exactly what the model would read. Recorded
  again with `gemma3:4b`: 1,566 runs.
- `examples/simple_rag.py`, `examples/adaptive_rag.py` and
  `examples/parent_child_rag.py`: every knob set and explained, with real output
  on the RAG types page.
- `run(query, trace=None)` on every RAG type, and `Trace.record()` keeps a step
  that failed; `NotRecorded` (a `ValueError`) when a replay has no recording.

### Changed
- `top_n` larger than `top_k`, and sizes that are not whole numbers above 0, now
  raise a clear `ValueError` in every RAG type.
- Adaptive RAG's search sizes come from `top_k` / `top_n` (complex questions 2x,
  +1x per retry); the defaults give the same 4/2, 8/4 and +4/+2 as before.
- `build_rag()` and `RAG_TYPES` live in `rag_titan.rag_types.rag_registry`
  (`from rag_titan.factory import build_rag` still works);
  `build_rag_parts(settings, rag_type=None)`. Evaluation's per-question scoring
  moved to `rag_titan.evaluation.run_scoring` (old imports still work).
- The playground's recording format is now keyed by tuning as well
  (`rag_titan.playground.recording`); the compare page is unchanged.

### Fixed
- `MMRReranker` returned numpy numbers as `rerank_score`; they are plain floats now.

## [0.11.0] - 2026-11-21

### Added
- **Three retrievers, one setting**: `RETRIEVER=dense` (meaning, the default),
  `bm25` (exact words) or `hybrid` (both, fused with Reciprocal Rank Fusion).
  Every RAG type works with all of them. Guide with a real comparison:
  [Retrieval](https://aion7891.github.io/rag-titan/retrieval/). On its help-desk
  example the right file came first for 6 of 6 questions with hybrid, 5 with
  dense and 3 with BM25.
- `BM25Retriever`, `BM25Index`, `tokenize`, `STOPWORDS` and `HybridRetriever` in
  `rag_titan.retrieval`. The keyword index is built from the vector store's chunks
  and rebuilt by itself when the store changes, also for saved stores.
  `BaseStore.all_chunks()` on every store.
- `MMRReranker` (`RERANKER=mmr`): relevant chunks that do not repeat each other.
- `rag-titan evaluate --retriever dense bm25 hybrid` (also with `--retrieval-only`),
  and `--retriever` on `ask` and `chat`. Setups that only differ in retriever or
  reranker share one index.
- `RAGTitan.count()`: how many chunks are indexed.
- RAG playground: a Retriever choice; hybrid search runs as three steps on the
  flow diagram (*Vector search → Keyword search → Fuse ranks (RRF)*), and the fuse
  step shows each chunk's rank in both searches. Recorded again with `gemma3:4b`.
- `rag.flow()` draws any RAG type with its retriever, and `Trace.search()` records
  a retriever's own steps, so new RAG types get this for free.
- `examples/compare_retrievers.py` with `examples/retrieval_docs/`.

### Changed
- Every retriever's `score` is a relevance from 0 to 1, so rerankers work the same
  with all of them. Dense scores (cosine) are unchanged; BM25 is divided by the best
  score for the question and hybrid by the best possible RRF. Raw scores are in
  `score_trace`.
- Flow diagrams name the search step after the retriever ("Vector search" instead
  of "Retrieve"); the step id in traces stays `retrieve`.
- `Playground.info()["flows"]` is keyed by RAG type, then retriever;
  `Playground.run()` / `evaluate()` and `LivePlayground.run()` take a retriever.
- `build_reranker()` and `build_retriever()` live in
  `rag_titan.retrieval.retriever_registry` (`from rag_titan.factory import
  build_reranker` still works); `rag-titan evaluate` moved to `cli_evaluate.py`.

## [0.10.0] - 2026-11-21

### Added
- **Saved vector stores: index once, ask later.** `VECTOR_STORE=sqlite` keeps the
  index in one file (`rag_titan.db`) with nothing to install; `VECTOR_STORE=chroma`
  (`pip install "rag-titan[chroma]"`) and `VECTOR_STORE=qdrant`
  (`pip install "rag-titan[qdrant]"`) keep it in a folder or on a server. Guide:
  [Vector stores](https://aion7891.github.io/rag-titan/vector-stores/).
- `SQLiteStore`, `ChromaStore` and `QdrantStore` in `rag_titan.vector_stores`.
  Every store gives cosine scores and passes the same checks
  (`tests/store_contract.py`, reusable for your own store).
- `rag-titan index --files ...` (`--add` to keep what is indexed); `ask` and `chat`
  use the saved index when `--files` is left out. `--vector-store` and
  `--store-path` on `index`, `ask` and `chat`.
- Adding to an index (`rag.index(files, reset=False)`): a file indexed before is
  replaced instead of stored twice, and chunk ids go on counting, so citations
  stay right.
- The index remembers its embedding model and vector size: searching or adding
  with another embedder stops with a clear message instead of wrong results.
- Settings `VECTOR_STORE_PATH`, `VECTOR_STORE_URL`, `COLLECTION` and
  `QDRANT_API_KEY`.
- `BaseStore.delete_source()`, `get_info()`, `set_info()` and `persistent`, with
  defaults, so stores written for 0.9 keep working.

### Changed
- Evaluation never changes a saved index: `rag-titan evaluate` (also
  `--retrieval-only`), `compare_setups()`, `evaluate_stages()` and
  `collect_dataset()` index into a temporary in-memory copy.
- `vector_store_path` follows the store when it is not set: `rag_titan.db`,
  `rag_titan_chroma` or `rag_titan_qdrant` (`None` for `memory`).
- `build_store()` and `VECTOR_STORES` live in `rag_titan.vector_stores.store_registry`;
  `from rag_titan.factory import build_store` still works.

## [0.9.0] - 2026-11-20

### Added
- **Claude (Anthropic) as the LLM**: `LLM_PROVIDER=claude` (or `anthropic`) and
  your `ANTHROPIC_API_KEY`. Works for RAG answers and for agents (tool calling).
  Plain HTTP with the standard library, so nothing extra to install. Setup:
  [Claude (Anthropic)](https://aion7891.github.io/rag-titan/configuration/#claude-anthropic).
  Thanks to @suvi-tiwary for the first version (#23).
- `AnthropicLLM(model="claude-haiku-5-5", api_key=None, max_tokens=2048,
  temperature=None)` in `rag_titan.llm`. Errors show Anthropic's own reason
  (like "invalid x-api-key") and never the key; a reply cut off at `max_tokens`
  logs a warning.
- Default models per provider: leave `LLM_MODEL` / `AGENT_MODEL` unset and they
  follow `LLM_PROVIDER` (Ollama: `gemma3:4b` / `qwen3:4b`, Claude:
  `claude-haiku-5-5`), so changing the provider alone is enough.
- An unknown `LLM_PROVIDER` now lists the valid choices.

### Changed
- `.env.example` leaves `LLM_MODEL` and `AGENT_MODEL` commented out, so a copied
  file follows the provider's defaults. Setting them works as before.

## [0.8.5] - 2026-11-13

### Added
- Docs: [Evaluate your RAG: the complete guide](https://aion7891.github.io/rag-titan/cookbook/complete-guide/),
  one page from the smallest input file to a report on all seven stages: prepare
  documents, golden questions, capture your RAG's logs, `build_dataset()`, chunking
  and embedding inputs, read the report, improve and compare with a baseline. With a
  step diagram, a cheat sheet of what to pass, a complete data file that uses every
  input (`full-data.json`, 100% measurement coverage) with every key explained,
  and a runnable example
  (`your-rag/` in the starter ZIP) that needs no model. Every command, output and
  score on the page is checked against a real run by the tests.

### Fixed
- A correct refusal is no longer scored as a wrong answer. For a question the
  documents cannot answer (`"answerable": false`) Exact Match, Answer F1 and ROUGE-L
  are now *not measured*, in the generation and end-to-end stages: Abstention
  Accuracy scores the reply. Before, "I don't know" got Answer F1 0 against the empty
  reference and was listed under "Queries to look at".
- A refusal (`"abstained": true`) is no longer checked against the context: Context
  Grounding and Unsupported Number Rate are *not measured* for it. Before, a RAG that
  learned to refuse correctly lost Context Grounding, and `--fail-on-regression`
  reported a regression for an improvement.

### Changed
- Scores can move for data with refusals or unanswerable questions. In the cookbook's
  `missed-evidence.json` (one question is refused) the overall score goes from 62.5
  to 63.2. Reports saved with an older version may show a difference in
  `context_grounding` or `answer_f1` when used as a `--baseline`: save a new baseline.

## [0.8.4] - 2026-11-12

### Fixed
- Windows: `RAGTitan(...)` no longer crashes with "DLL load failed while importing
  _pydantic_core: An Application Control policy has blocked this file". rag-titan
  now talks to Ollama over plain HTTP with the Python standard library, so it no
  longer needs pydantic and its compiled `_pydantic_core` file, which Smart App
  Control and company application-control policies block.
- Clearer errors when Ollama is not running ("Could not reach Ollama at ... Is
  Ollama running?"), when Ollama returns an error (its own message, like "model not
  found", is shown), and when it sends a different number of embeddings than texts.

### Changed
- `OllamaLLM` and `OllamaEmbedder` send exactly the same requests as before
  (same JSON, same tool descriptions, same host rules, `OLLAMA_HOST` and
  `OLLAMA_API_KEY` work as before). Their `.client` attribute is gone; use `.host`
  to see the server address.
- `OllamaLLM.chat()["message"]` is now a plain dict instead of an `ollama.Message`
  object. `message["content"]` and the other keys work as before; attribute access
  like `message.content` does not.

### Removed
- The `ollama` dependency, and with it `pydantic`, `pydantic-core`, `httpx`,
  `httpcore`, `anyio`, `h11` and `typing-inspection`.

## [0.8.3] - 2026-11-12

### Fixed
- Windows: `python -m rag_titan ...` now runs the command line tool, with the same
  commands and options as `rag-titan`. Use it when Smart App Control or an
  application-control policy blocks the unsigned `rag-titan.exe` launcher that pip
  creates. Before, `python -m rag_titan` failed with "is a package and cannot be
  directly executed".
- Docs: Getting started, the cookbook's Windows install steps and the Command line
  page use `python -m rag_titan` instead of `rag-titan.exe`.

## [0.8.2] - 2026-11-12

Helpers for users who already have a RAG, from a new-user test. Nothing existing
was removed or renamed.

### Added
- `rag-titan golden --questions questions.txt --files docs.md` and
  `make_golden_template()`: a first draft of your golden questions, with a
  suggested evidence sentence per question. Every record starts unvalidated, so
  `check_golden()` is not ready until a person has reviewed it. Never overwrites
  an existing file without `--force`.
- `build_dataset(chunks, runs, golden, documents)`: turns the logs of any RAG and
  your golden questions into an evaluation data file. It places chunks in their
  documents, takes labels only from the golden questions, finds gold chunk ids
  from evidence quotes and turns `[p1]` markers into citations. On the new-user
  test data, measurement coverage went from 32% to 59%.
- Docs: "Prepare golden data step by step", "Why is a value null?", "A regression
  test in plain words" with a diagram, and `build_dataset()` in "Evaluate your own RAG".

### Changed
- The `rag-titan evaluate --input` summary also prints "Queries to look at" (each
  failed query and its stage) and "Not measured" (what input would measure each
  missing part).
- Regression checks ignore latency noise: a latency counts as worse only when it is
  more than 10% and more than 5 ms slower. `find_regressions()` and
  `ReportManager.regressions()` take `latency_share=0, latency_min_ms=0` for the
  old strict rule.

## [0.8.1] - 2026-11-12

Docs-only release: no change to the Python API.

### Added
- Beginner evaluation cookbook: six runnable recipes, downloadable teaching data and
  live Ollama scripts, interactive stage walkthroughs using real evaluator results,
  and Bash/PowerShell command builders for scoring and regression checks.
- Animated terminal demos of the evaluation recipes, generated from real
  `rag-titan evaluate` output, including an end-to-end demo from golden questions
  to a scorecard.
- Animated evaluation flow image: your logs in, the score of every stage and the
  first failing stage out, drawn from a real evaluator run.
- "Choose your path" on the cookbook start page, with six steps for users who
  already have a RAG and only want to evaluate it; buttons on the Home page.

### Changed
- Docs sidebar regrouped into collapsible sections: Get started, Evaluate your RAG,
  Evaluation reference, Build a RAG, Playgrounds and Advanced.

## [0.8.0] - 2026-11-12

### Added
- Stage-by-stage evaluation for any RAG system from logged outputs and gold labels,
  without an LLM judge or new dependencies: `ChunkEvaluator`, `EmbeddingEvaluator`,
  `RetrievalEvaluator`, `RerankingEvaluator`, `AugmentationEvaluator`,
  `GenerationEvaluator` and `EndToEndEvaluator` from `rag_titan.evaluation`.
  Metrics cover chunk quality, vectors, ranking, context, answers and citations,
  latency, cost and tokens, with multi-document support.
- `RAGEvaluator`, `ScoreAggregator` and `RootCauseAnalyzer`: a weighted 0–100
  scorecard with quality gates, measurement coverage, provisional status and
  root-cause hints. Missing measurements stay unmeasured rather than scoring zero.
- `ReportManager` and `evaluate_file()`: JSON and Markdown reports, comparisons
  against a baseline and trends, with regression direction taken from the score policy.
- `rag-titan evaluate --input data.json` with `--output`, `--baseline`, `--k` and
  `--fail-on-regression` (non-zero exit on regression). Existing
  `--files` / `--questions` evaluation remains supported.
- `collect_dataset()` and `RAGTitan.evaluate_stages()` to collect and evaluate
  rag-titan runs, plus `load_golden()` for gold labels and expected evidence.
- Dataset helpers: `profile_documents()` for empty, short and duplicate documents,
  `sample_documents()` for random, stratified or risk-based samples, and
  `check_golden()` for golden-set checks and human-validation flags.
- Stage evaluation guide, metric formula reference, architecture diagram and
  runnable examples with sample data and golden questions.

## [0.7.0] - 2026-11-11

### Added
- **RAG playground** on the docs site: the real package runs in your browser
  (Pyodide 314.0.7 in a Web Worker). Pick chunkers and RAG types, ask a question,
  and every setup runs side by side on its own live flow diagram: steps light up
  in order with step numbers, loops are drawn, and clicking a step shows its input
  and output (chunks with files and scores, rewritten queries, evidence checks).
  - Recorded mode: real `gemma3:4b` replies and `nomic-embed-text` vectors,
    replayed instantly; "Evaluate all questions" gives the same report as
    `rag-titan evaluate`.
  - Live mode: your own Ollama (`OLLAMA_ORIGINS`), any question, the sample
    documents or your own pasted text.
- `rag-flow.js`: draws any RAG type from its `FLOW` (automatic layout, loops),
  so new RAG types get a diagram without JavaScript.
- `HashEmbedder` (`EMBED_PROVIDER=hash`): word-hash vectors, no model needed.
- `FunctionLLM` and `FunctionEmbedder`: any function becomes a model.
- `RecordingLLM` / `ReplayLLM` and `RecordingEmbedder` / `ReplayEmbedder`:
  record real model replies and vectors once, replay them anywhere.
- `rag_titan.playground`: `Playground`, `LivePlayground`, `record_presets()`,
  `check_presets()`; `tools/playground/record_traces.py` records the examples
  (`--check` replays them, and a test fails when they are out of date).
- Docs: Ollama Cloud setup (`OLLAMA_HOST=https://ollama.com` + `OLLAMA_API_KEY`).

### Changed
- `ollama`, `pypdf`, `python-docx` and `python-dotenv` are imported only when
  used, so the package also loads without them (for example in the browser).
  Nothing changes when they are installed.

## [0.6.1] - 2026-11-11

### Fixed
- `pip install rag-titan` no longer installs pytest: it was listed as a runtime
  dependency by mistake. It stays in the `dev` extra.

### Changed
- README shows the 0.6 features: `rag-titan evaluate` with a real report,
  `run()` / `--trace` step logs, and the Evaluating RAG page.

## [0.6.0] - 2026-11-11

### Added
- End-to-end evaluation (RAG handbook step 0): compare any ingestion with any
  RAG type on your own test questions.
  - `rag-titan evaluate --files ... --questions questions.json --rag-type simple
    adaptive --chunker word paragraph` runs every combination and prints a
    report (Recall, MRR, nDCG, answered rate, steps, p50/p95 time) and the
    questions that went wrong. `--retrieval-only --top-k 5` scores search alone.
  - `compare_setups(files, questions, setups)`, `evaluate_rag()` and
    `format_rag_report()`. Evidence is scored on the chunks given to the LLM;
    setups with the same settings share one index.
  - Retrieval metrics: `recall_at_k()`, `reciprocal_rank()`, `ndcg_at_k()`,
    `evaluate_retriever()`, `compare_retrievers()`, `format_report()`.
  - Questions file loader `load_questions()`. Evidence ids name a file
    (`"hr.pdf"`) or a page (`"hr.pdf#p4"`), so labels keep working when the
    chunker changes; each evidence counts once.
- `run(question)` on every RAG type: the answer, the chunks used (`sources`) and
  every step with its input, output and time. `describe_steps()` prints them as
  a readable log; `rag-titan ask ... --trace` shows it in the terminal.
- `FLOW` on every RAG type: its steps and the arrows between them, for flow
  diagrams. A test checks that every run follows its FLOW.
- `RAGTitan.run()`, `RAGTitan.evaluate()` and `RAGTitan.evaluate_retrieval()`.
- `BaseRetriever`, so any retriever can be used by every RAG type, and
  `DenseRetriever` (the retriever from before; `Retriever` still works).
  Results also say which retriever found them: `"retriever"` and
  `"score_trace"`, e.g. `{"dense": 0.82}`.
- `BaseRAG` and `Trace` for writing new RAG types; `CHUNKERS` in the factory.
- Example `examples/evaluate_retrieval.py` with sample policies and questions.
- Docs: "Evaluating RAG" page with an animated end-to-end flow diagram.

### Changed
- `retrieve()` raises `ValueError` for a `top_k` that is not a whole number of
  at least 1 (it returned an empty list for 0 before).
- Search results have two more keys, `"retriever"` and `"score_trace"`. Code
  that compares whole result dicts must ignore them.

## [0.5.0] - 2026-11-11

### Added
- Search results no longer hold near-copies (chunking handbook 2.4). With
  overlap, neighbouring chunks share text, so a search could return the same
  passage twice. The retriever now fetches twice as many results and drops one
  that overlaps a better result from the same file by at least half of the
  shorter chunk, using the chunk offsets.
- Setting `DEDUPE_OVERLAP` (default `0.5`; `off` keeps every result), and
  `Retriever(..., dedupe_overlap=0.5)`.
- `rag_titan.retrieval.overlap_fraction()` and `remove_duplicates()`.

### Changed
- Retrieval returns different results than 0.4.x when chunks overlap: fewer
  near-copies, more different passages. Set `DEDUPE_OVERLAP=off` for the old
  behaviour.

### Fixed
- Docs site: the 0.4.0 docs were not published, because a docstring
  (`BaseChunker.chunk`) was read as a broken link in the API reference.

## [0.4.0] - 2026-11-11

### Added
- Chunk offsets (chunking handbook section 1): every chunk gets
  `start_offset` and `end_offset`, so `document["text"][start:end]` is where it
  comes from.
  - Exact for the character, word, recursive, sentence, paragraph, heading and
    page chunkers, even in very repetitive text.
  - `TokenChunker` and chunkers you write yourself get offsets by searching for
    each chunk in order (`None` when a chunk cannot be found).
- `page_numbers` on chunks of documents with pages (PDFs), and `section_path`
  on chunks of documents with headings, with any chunker.
- The answer context shows where each chunk comes from, for example
  `[chunk_4] Source: handbook.pdf, page 3, section: Leave policy > Sick leave`.
- `BaseChunker.split_spans(text)`: override it in your own chunker to report
  exact positions.
- New modules: `rag_titan.chunking.offsets`, `rag_titan.chunking.locations` and
  `rag_titan.chunking.headings`.

### Changed
- The answer prompt lets the LLM cite pages, such as `[chunk_0, page 3]`.
- The Markdown heading rules moved to `rag_titan.chunking.headings`; importing
  them from `heading_chunker` still works. Chunk text is unchanged.

## [0.3.1] - 2026-11-10

### Added
- Docs: a "Chunk playground" page. Try the chunkers on your own text in the
  browser, load example texts with one click, and follow "things to try".
- Docs: an animated diagram for each of the 8 chunkers, showing it run on a
  real text one chunk at a time, with the overlap highlighted.
- `tools/diagrams/`: draws those diagrams from the chunkers' real output, and
  `--check` fails when a diagram no longer matches the code. Not part of the
  package.

### Changed
- README: links to the chunk playground and the chunking guide.
- Docs: the chunking page links to the playground instead of embedding it.

## [0.3.0] - 2026-11-10

### Added
- Structure-aware chunking (chunking handbook section 3), in
  `rag_titan.chunking.structure_aware`:
  - `SentenceChunker`: packs whole sentences.
  - `ParagraphChunker`: packs whole paragraphs; a long one falls back to
    sentences, then words.
  - `HeadingChunker`: never mixes two sections and starts each chunk with its
    heading path, for example "Leave policy > Sick leave" (`include_path=False`
    keeps the original heading lines instead).
  - `PageChunker`: never puts two PDF pages in one chunk.
  - `split_sentences()` and `split_paragraphs()`: no library needed; handle
    abbreviations, initials, list numbers, decimals and CJK/Hindi end marks,
    and never lose a character.
- Settings: `CHUNKER=sentence`, `paragraph`, `heading` or `page`.
- `TextLoader` for `.txt`, `.md` and `.markdown` files.
- `rag_titan.chunking.packing.pack()`: the shared "whole pieces up to a size"
  step, now also used by `RecursiveChunker` (same output as before).
- Docs: a "Structure-aware chunkers" section on the Chunking methods page, and
  sentence and paragraph modes in the chunk playground.

### Changed
- Loaders keep the document's structure. DOCX paragraphs are separated by a
  blank line and Word headings become Markdown headings (`#`, `##`, ...); PDF
  pages are separated by a page break (`\f`) instead of a space.
  - `WordChunker` gives the same chunks as before, except that DOCX headings now
    add `#` marks.
  - `CharacterChunker`, `TokenChunker` and `RecursiveChunker` see the real
    paragraphs, so their chunks of PDF and DOCX files differ from 0.2.x.
    Re-index your files after upgrading.

## [0.2.1] - 2026-11-10

### Added
- Docs: a "Chunking methods" page that explains every chunker with
  formulas (stride, chunk count, overlap cost), runnable examples with real
  output, three animated diagrams, and an interactive chunk playground that
  runs in the browser.

### Changed
- Docs: "Building blocks" links to the new page instead of repeating it; the
  docs site can now show maths.

### Fixed
- README: the PyPI version badge showed an old cached version.

## [0.2.0] - 2026-11-10

### Added
- Fixed-length chunking (chunking handbook section 2):
  - `CharacterChunker`: fixed windows of characters, text kept exactly.
  - `TokenChunker`: fixed windows of model tokens. Uses tiktoken
    (`pip install "rag-titan[tokens]"`) or your own tokenizer with
    `encode()` / `decode()`.
  - `RecursiveChunker`: splits on paragraphs, lines, sentences and words
    before cutting, with whole-piece overlap; counts characters or tokens.
  - `fixed_windows()`: the shared sliding-window helper.
  - `count_windows()`: how many chunks a text makes, to see the cost of overlap.
- Settings: `CHUNKER=character`, `token` or `recursive`, and `CHUNK_ENCODING`.
- Optional install extra `tokens` (tiktoken).

### Changed
- Chunkers are grouped by family: fixed-size chunkers live in
  `rag_titan.chunking.fixed_size`. `from rag_titan.chunking import ...` and the
  old `rag_titan.chunking.word_chunker` path keep working.
- `WordChunker` is built on `fixed_windows()`; its output is unchanged.
- Docs: fixed-length chunkers, overlap cost, settings for every chunker, and
  where new chunkers go.
- README: banner header, a short summary, and links to the docs site.
- README and docs: a clear "Supported RAG types" table (simple, adaptive) and a
  separate list of planned RAG types.
- Docs: logo and favicon, banner on the home page.
- Fixed the number of LLM calls for adaptive RAG in the docs and diagram (2 to 5).

## [0.1.0] - 2026-11-10

### Added
- First release of `rag-titan`.
- RAG types: `SimpleRAG` and `AdaptiveRAG`.
- `RAGTitan` class: index files and ask questions in two lines.
- `rag-titan` command line tool (`ask` and `chat`).
- Building blocks: PDF/DOCX loaders, word chunker, Ollama LLM and embedder,
  in-memory vector store, keyword reranker, grounded generator.
- Tool-calling `BaseAgent` with an example `MathAgent`.
- Tests that run without Ollama, docs site, and PyPI publish workflow.
- Animated flow diagrams (SVG) for every capability, with worked examples.
