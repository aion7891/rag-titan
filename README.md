# rag-titan

**A Swiss Army knife for RAG (Retrieval-Augmented Generation).**

`rag-titan` puts many types of RAG in one small, beginner-friendly Python
package. Every part (loader, chunker, embedder, vector store, reranker,
generator) is a small class you can read in a minute and replace with your own.

[![PyPI](https://img.shields.io/pypi/v/rag-titan.svg)](https://pypi.org/project/rag-titan/)
[![Python](https://img.shields.io/pypi/pyversions/rag-titan.svg)](https://pypi.org/project/rag-titan/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Features

| RAG type   | How it works                                                        |
|------------|---------------------------------------------------------------------|
| `simple`   | Retrieve → Rerank → Generate                                        |
| `adaptive` | Route the question → Retrieve → Check evidence → Rewrite and retry  |

More RAG types (corrective, hybrid, multi-query, graph, agentic, ...) are
planned. See the [roadmap](docs/roadmap.md).

- Runs **100% locally** with [Ollama](https://ollama.com), so your data stays on your machine.
- Reads **PDF** and **DOCX** files.
- Answers are **grounded** in your documents and **cite** the chunks they used.
- Every part is swappable. You only need to inherit from one base class.
- Includes a `rag-titan` command line tool.

## Installation

```bash
pip install rag-titan
```

Then install [Ollama](https://ollama.com/download) and pull the default models:

```bash
ollama pull gemma3:4b          # chat model
ollama pull nomic-embed-text   # embedding model
```

## Quick start

```python
from rag_titan import RAGTitan

rag = RAGTitan(rag_type="simple")        # or "adaptive"
rag.index(["handbook.pdf", "policy.docx"])

print(rag.ask("How many leave days do I get?"))
# You get 20 leave days every year [chunk_0].
```

Change any setting in code:

```python
rag = RAGTitan(
    rag_type="adaptive",
    settings={"llm_model": "llama3.2", "chunk_size": 200, "reranker": "none"},
)
```

## Command line

```bash
rag-titan ask  --files handbook.pdf --question "How many leave days?"
rag-titan chat --files handbook.pdf policy.docx --rag-type adaptive
```

## Configuration

Settings come from environment variables or a `.env` file. Copy
[`.env.example`](.env.example) to get started. Every setting has a default.

| Variable        | Default                  | Meaning                      |
|-----------------|--------------------------|------------------------------|
| `LLM_MODEL`     | `gemma3:4b`              | Chat model used for answers  |
| `EMBED_MODEL`   | `nomic-embed-text`       | Embedding model              |
| `OLLAMA_HOST`   | `http://localhost:11434` | Ollama server address        |
| `CHUNK_SIZE`    | `150`                    | Words per chunk              |
| `CHUNK_OVERLAP` | `30`                     | Words shared between chunks  |
| `RERANKER`      | `keyword`                | `keyword` or `none`          |

See the [configuration docs](docs/configuration.md) for the full list.

## How it works

```
Indexing  (once):    files -> load -> chunk -> embed -> vector store
Answering (each Q):  question -> retrieve -> rerank -> generate answer
```

## Project layout

```
src/rag_titan/
├── app.py            RAGTitan: the easy, all-in-one class
├── cli.py            the `rag-titan` command
├── config.py         settings from .env / environment
├── factory.py        builds the right class for each setting
├── prompts.py        every RAG prompt in one place
├── llm/              chat models (Ollama)
├── loaders/          PDF and DOCX readers
├── chunking/         split text into chunks
├── embeddings/       text -> vectors
├── vector_stores/    save and search vectors
├── retrieval/        retriever + rerankers
├── generation/       write the final answer
├── pipelines/        indexing pipeline
├── rag_types/        simple and adaptive RAG
├── agents/           tool-calling agents
└── utils/            logging
```

## Documentation

Full docs live in [`docs/`](docs/index.md). To view them locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
