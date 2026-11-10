# Changelog

All notable changes to this project are written here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

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
