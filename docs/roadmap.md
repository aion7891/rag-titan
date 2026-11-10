# Roadmap

`rag-titan` aims to cover every common RAG type with the same simple building blocks.

## RAG types

- [x] Simple RAG (`"simple"`), available since 0.1.0
- [x] Adaptive RAG (`"adaptive"`), available since 0.1.0
- [ ] Corrective RAG (CRAG): grade chunks, fall back to web search
- [ ] Multi-query RAG: ask several versions of the question
- [ ] HyDE: search with a hypothetical answer
- [ ] Hybrid RAG: keyword (BM25) + vector search
- [ ] Self-RAG: the model critiques its own answer
- [ ] Parent-document RAG: search small chunks, answer with big ones
- [ ] Graph RAG: use a knowledge graph
- [ ] Agentic RAG: an agent decides when and what to search

## Building blocks

- [ ] LLMs: OpenAI, Anthropic, Hugging Face
- [ ] Embedders: OpenAI, sentence-transformers
- [ ] Vector stores: Chroma, Qdrant, FAISS, pgvector
- [ ] Rerankers: cross-encoder, LLM reranker
- [x] Loaders: TXT and Markdown (`TextLoader`), available since 0.3.0
- [ ] Loaders: HTML, CSV
- [x] Chunkers: character, token, recursive (`"character"`, `"token"`, `"recursive"`), available since 0.2.0
- [x] Chunkers: sentence, paragraph, heading, page (`"sentence"`, `"paragraph"`, `"heading"`, `"page"`), available since 0.3.0
- [ ] Chunkers: element-aware (lists, tables, code), semantic
- [x] Chunk offsets, page numbers and section paths, with page citations in answers, available since 0.4.0
- [x] Remove duplicate overlapping chunks from search results (`DEDUPE_OVERLAP`), available since 0.5.0
- [ ] Evaluation helpers
