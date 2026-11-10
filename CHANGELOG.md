# Changelog

All notable changes to this project are written here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
