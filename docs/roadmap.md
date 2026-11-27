# Roadmap

`rag-titan` aims to cover every common RAG type with the same simple building blocks.

## RAG types

- [x] Simple RAG (`"simple"`), available since 0.1.0
- [x] Adaptive RAG (`"adaptive"`), available since 0.1.0
- [x] Parent-Child RAG (`"parent_child"`): search small chunks, answer with their sections, available since 0.12.0
- [x] HyDE RAG (`"hyde"`): search with a made-up answer and the question, available since 0.13.0
- [x] RAG-Fusion (`"rag_fusion"`): search several rewordings, fuse with RRF, available since 0.13.0
- [x] Corrective RAG (`"corrective"`): grade every chunk, search again when the evidence is weak, keep the key sentences, available since 0.14.0
- [x] Iterative RAG (`"iterative"`): search, think about what is missing, search again, available since 0.14.0
- [x] Adaptive RAG's retries on the shared loop limit, with `result["loop"]`, available since 0.15.0
- [ ] Corrective RAG with a built-in web search fallback (today: plug in any retriever as `fallback_retriever`)
- [ ] Self-RAG: the model critiques its own answer
- [ ] Graph RAG: use a knowledge graph
- [ ] Agentic RAG: an agent decides when and what to search

## Building blocks

- [x] LLMs: Anthropic Claude (`LLM_PROVIDER=claude`), available since 0.9.0
- [x] LLMs and embeddings: OpenAI, Gemini, OpenRouter, Hugging Face (chat), LM Studio, vLLM and any OpenAI-compatible server, one built-in client, available since 0.15.2
- [ ] Embedders: sentence-transformers (local, no server)
- [x] Saved vector stores: SQLite (built in), Chroma, Qdrant; `rag-titan index`, add/replace files, embedder check, available since 0.10.0
- [ ] Vector stores: FAISS, LanceDB, pgvector, Pinecone, then Weaviate, Milvus, Elasticsearch, Redis, MongoDB
- [x] Retrievers: BM25 and hybrid (RRF) next to dense, `RETRIEVER` setting, `evaluate --retriever`, available since 0.11.0
- [x] Weighted RRF and keep-each-search's-#1 for hybrid, HyDE and RAG-Fusion (`HYBRID_DENSE_WEIGHT`, `HYBRID_BM25_WEIGHT`, `QUESTION_WEIGHT`, `RRF_KEEP_TOP`), available since 0.15.0
- [x] Rerankers: MMR (`RERANKER=mmr`), available since 0.11.0
- [x] Rerankers: cross-encoder (`RERANKER=cross_encoder`, `[rerank]` extra) and LLM reranker (`RERANKER=llm`), available since 0.12.0
- [x] Tuning settings for every RAG type (`TOP_K`, `TOP_N`, `ADAPTIVE_MAX_ATTEMPTS`, `PARENT_SIZE`, `PARENT_BUDGET`) and a playground page per RAG type with knobs, available since 0.12.0
- [x] Loaders: TXT and Markdown (`TextLoader`), available since 0.3.0
- [ ] Loaders: HTML, CSV
- [x] Open Knowledge Format bundles: read and validate (v0.3, v0.2 readable), `rag-titan okf validate` / `info`, index with trust tiers and current-only filtering (`OKF_INCLUDE`, `OKF_MIN_TRUST`), available since 0.15.3
- [x] Chunkers: character, token, recursive (`"character"`, `"token"`, `"recursive"`), available since 0.2.0
- [x] Chunkers: sentence, paragraph, heading, page (`"sentence"`, `"paragraph"`, `"heading"`, `"page"`), available since 0.3.0
- [x] Chunkers: element-aware Markdown (`"markdown"`: whole lists, tables, code) and TextTiling (`"texttiling"`: topic changes, no model), available since 0.15.1
- [ ] Chunkers: semantic (embedding similarity), LLM boundaries
- [x] Chunk offsets, page numbers and section paths, with page citations in answers, available since 0.4.0
- [x] Remove duplicate overlapping chunks from search results (`DEDUPE_OVERLAP`), available since 0.5.0
- [x] Evaluation helpers: retrieval and end-to-end (`rag-titan evaluate`), available since 0.6.0
- [x] Step traces and flows for every RAG type (`run()`, `FLOW`, `ask --trace`), available since 0.6.0
- [x] Swappable retrievers (`BaseRetriever`, `DenseRetriever`), available since 0.6.0
- [x] RAG playground: run RAG types in the browser with live flow diagrams, recorded or live with your Ollama, available since 0.7.0
- [x] Answer grading: deterministic answer, fact coverage, grounding and citation metrics (`GenerationEvaluator`), available since 0.8.0
- [x] Stage evaluation: chunking through end to end, scorecards, root-cause hints, reports and regression checks (`RAGEvaluator`, `rag-titan evaluate --input`), available since 0.8.0
- [x] Evaluation cookbook: six recipes, starter kit and animated walkthroughs, available since 0.8.1
- [x] Beginner evaluation helpers: golden questions template (`rag-titan golden`), `build_dataset()` for any RAG's logs, clearer summary, available since 0.8.2

## Later ideas

- [ ] LLM-based answer and citation grading
- [ ] Stage evaluation directly from CLI runs (`rag-titan evaluate --files ... --questions golden.json --stages`)
- [ ] LLM-drafted golden questions, validated by a human
- [ ] S3 / cloud loaders
- [ ] Confidence intervals and repeated evaluation runs
- [ ] Scorecard breakdown by task type
