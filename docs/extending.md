# Extending rag-titan

Every part works the same way:

1. create a class that inherits from the part's **base class**
2. write the methods the base class asks for
3. (optional) add one `if name == ...` line in `factory.py` so it can be chosen from settings

## Example: a new chunker

```python
from rag_titan.chunking import BaseChunker


class SentenceChunker(BaseChunker):
    """Splits text into groups of sentences."""

    def __init__(self, sentences_per_chunk=5):
        self.sentences_per_chunk = sentences_per_chunk

    def split_text(self, text):
        """Return pieces of `sentences_per_chunk` sentences each."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        pieces = []

        for start in range(0, len(sentences), self.sentences_per_chunk):
            group = sentences[start:start + self.sentences_per_chunk]
            pieces.append(". ".join(group) + ".")

        return pieces
```

Then register it in `factory.py`:

```python
def build_chunker(settings):
    name = settings["chunker"]

    if name == "word":
        return WordChunker(settings["chunk_size"], settings["chunk_overlap"])
    if name == "sentence":
        return SentenceChunker()
    ...
```

and set `CHUNKER=sentence` in `.env`.

![Where your new class plugs in](assets/images/architecture.svg)

## What each base class needs

| Base class      | Methods to write                                  |
|-----------------|---------------------------------------------------|
| `BaseLLM`       | `chat(messages, tools)`, `tool_result_message()`  |
| `BaseLoader`    | `extensions` list and `extract_text(path)`        |
| `BaseChunker`   | `split_text(text)`                                |
| `BaseEmbedder`  | `embed_many(texts)`                               |
| `BaseStore`     | `add()`, `search()`, `clear()`, `count()`         |
| `BaseReranker`  | `score(query, match)`                             |
| `BaseGenerator` | `generate(query, chunks)`                         |

## Adding a new RAG type

1. create `src/rag_titan/rag_types/my_rag.py` with a class that has an
   `answer(query)` method
2. add it to `RAG_TYPES` and `build_rag()` in `factory.py`
3. export it in `rag_types/__init__.py`
4. add tests in `tests/` (use `FakeLLM` and `FakeEmbedder` from `conftest.py`)
5. document it in `docs/rag-types.md`

## Code style

- keep functions short (under about 50 lines) and files under about 200 lines
- no type hints, plain dicts and lists
- a docstring for every class and public method
- comments explain **why**, not what
