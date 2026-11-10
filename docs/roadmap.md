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
- [ ] Loaders: TXT, Markdown, HTML, CSV
- [ ] Chunkers: sentence, recursive, semantic
- [ ] Evaluation helpers
