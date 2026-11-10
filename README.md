<p align="center">
  <a href="https://aion7891.github.io/rag-titan/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/banner.svg" alt="rag-titan: a Swiss Army knife for RAG. Many RAG types, one simple API, runs locally." width="100%"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/rag-titan/"><img src="https://img.shields.io/pypi/v/rag-titan?label=pypi" alt="PyPI version"></a>
  <a href="https://pypi.org/project/rag-titan/"><img src="https://img.shields.io/pypi/pyversions/rag-titan" alt="Python versions"></a>
  <a href="https://github.com/aion7891/rag-titan/actions/workflows/tests.yml"><img src="https://github.com/aion7891/rag-titan/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://aion7891.github.io/rag-titan/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/aion7891/rag-titan/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/"><b>📖 Documentation</b></a> ·
  <a href="https://aion7891.github.io/rag-titan/getting-started/">Getting started</a> ·
  <a href="https://aion7891.github.io/rag-titan/rag-types/">Supported RAG types</a> ·
  <a href="https://aion7891.github.io/rag-titan/api/core/">API reference</a> ·
  <a href="https://pypi.org/project/rag-titan/">PyPI</a>
</p>

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/overview.svg" alt="rag-titan overview: index your files once, then ask questions" width="100%"></a>
</p>

## Why rag-titan?

- **Many RAG types, one API.** Two today (`simple`, `adaptive`), with more on the way. Switch with one word: `rag_type="adaptive"`.
- **Runs 100% locally** with [Ollama](https://ollama.com), so your documents never leave your machine.
- **Grounded answers.** The model answers only from your files and cites the chunks it used.
- **Fixed-length chunking your way.** Split by words, characters, model tokens (tiktoken or your own tokenizer) or recursive separators that keep paragraphs whole, with overlap.
- **Easy to read, easy to change.** Short files, plain Python, and a docstring on every class. Swap any part by inheriting one base class.

## Install

```bash
pip install rag-titan
ollama pull gemma3:4b && ollama pull nomic-embed-text
```

For token-based chunking, add the extra: `pip install "rag-titan[tokens]"`.

Needs Python 3.10+ and [Ollama](https://ollama.com/download). → [Full setup guide](https://aion7891.github.io/rag-titan/getting-started/)

## Quick start

```python
from rag_titan import RAGTitan

rag = RAGTitan(rag_type="simple")            # or "adaptive"
rag.index(["handbook.pdf", "policy.docx"])

print(rag.ask("How many leave days do I get?"))
# You get 20 paid leave days every year [chunk_0].
```

Or from the terminal:

```bash
rag-titan ask --files handbook.pdf --question "How many leave days do I get?"
```

## Supported RAG types

**rag-titan 0.1 supports 2 RAG types.** Pick one with `rag_type`:

| Status | RAG type | How it works | Best for | LLM calls |
|---|---|---|---|---|
| ✅ Available | [**Simple RAG**](https://aion7891.github.io/rag-titan/rag-types/#simple-rag-simple)<br>`rag_type="simple"` | Retrieve → Rerank → Generate | Fast answers to clear questions | 1 |
| ✅ Available | [**Adaptive RAG**](https://aion7891.github.io/rag-titan/rag-types/#adaptive-rag-adaptive)<br>`rag_type="adaptive"` | Route → Retrieve → Check evidence → Rewrite & retry | Mixed questions, fewer wrong answers | 2–5 |

```python
from rag_titan import RAG_TYPES
print(RAG_TYPES)        # ['simple', 'adaptive']
```

**Coming next** (not available yet): Corrective RAG · Multi-query RAG · HyDE · Hybrid RAG · Self-RAG · Parent-document RAG · Graph RAG · Agentic RAG. → [Roadmap](https://aion7891.github.io/rag-titan/roadmap/)

> Also included: a tool-calling [agent](https://aion7891.github.io/rag-titan/agents/) loop. It's a separate feature, not a `rag_type`.

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/rag-types/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/simple-rag.svg" alt="Simple RAG: retrieve, rerank, generate, with example scores" width="100%"></a>
</p>

<p align="center">
  <a href="https://aion7891.github.io/rag-titan/rag-types/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/adaptive-rag.svg" alt="Adaptive RAG: route, check evidence, rewrite and retry" width="100%"></a>
</p>

## Explore the docs

Click any picture to open its page.

<table>
  <tr>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/building-blocks/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/indexing.svg" alt="Indexing pipeline"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/building-blocks/">Building blocks</a></b><br>Load → chunk → embed → store, with the data at every step.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/extending/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/architecture.svg" alt="Swappable architecture"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/extending/">Extending rag-titan</a></b><br>Add your own LLM, vector store, chunker or RAG type.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/agents/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/agent.svg" alt="Agent loop"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/agents/">Agents</a></b><br>Let the LLM call your Python functions as tools.</p>
    </td>
    <td width="50%" valign="top">
      <a href="https://aion7891.github.io/rag-titan/cli/"><img src="https://cdn.jsdelivr.net/gh/aion7891/rag-titan@main/docs/assets/images/cli.svg" alt="Command line demo"></a>
      <p><b><a href="https://aion7891.github.io/rag-titan/cli/">Command line</a></b><br><code>rag-titan ask</code> and <code>rag-titan chat</code>.</p>
    </td>
  </tr>
</table>

| I want to... | Read |
|---|---|
| Install it and ask my first question | [Getting started](https://aion7891.github.io/rag-titan/getting-started/) |
| Pick the right RAG type | [RAG types](https://aion7891.github.io/rag-titan/rag-types/) |
| Change models, chunk size or reranker | [Configuration](https://aion7891.github.io/rag-titan/configuration/) |
| See every class and method | [API reference](https://aion7891.github.io/rag-titan/api/core/) |

## Contributing

Contributions are welcome! Read [CONTRIBUTING.md](https://github.com/aion7891/rag-titan/blob/main/CONTRIBUTING.md) to get set up. The tests run without Ollama.

## License

[MIT](https://github.com/aion7891/rag-titan/blob/main/LICENSE) © Kumar Nishant
