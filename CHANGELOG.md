# Changelog

All notable changes to this project are written here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
