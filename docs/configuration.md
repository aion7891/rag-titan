# Configuration

Settings are read from **environment variables** or a **`.env` file** in the
folder where you run your code. Every setting has a default, so nothing is required.

![Settings pick a class for every part](assets/images/architecture.svg)

Copy the example file to start:

```bash
cp .env.example .env
```

## All settings

| Variable          | Settings key      | Default                  | Meaning                             |
|-------------------|-------------------|--------------------------|-------------------------------------|
| `LLM_PROVIDER`    | `llm_provider`    | `ollama`                 | Who runs the chat model             |
| `LLM_MODEL`       | `llm_model`       | `gemma3:4b`              | Chat model for answers              |
| `OLLAMA_HOST`     | `ollama_host`     | `http://localhost:11434` | Ollama server address               |
| `AGENT_MODEL`     | `agent_model`     | `qwen3:4b`               | Model for agents (needs tools)      |
| `AGENT_THINK`     | `agent_think`     | `true`                   | Turn on thinking for agents         |
| `AGENT_MAX_STEPS` | `agent_max_steps` | `5`                      | Max agent think/tool rounds         |
| `EMBED_PROVIDER`  | `embed_provider`  | `ollama`                 | Who runs the embedding model        |
| `EMBED_MODEL`     | `embed_model`     | `nomic-embed-text`       | Embedding model                     |
| `CHUNKER`         | `chunker`         | `word`                   | see [Chunking methods](chunking.md) |
| `CHUNK_SIZE`      | `chunk_size`      | `150`                    | Units per chunk (see below)         |
| `CHUNK_OVERLAP`   | `chunk_overlap`   | `30`                     | Units shared by neighbouring chunks |
| `CHUNK_ENCODING`  | `chunk_encoding`  | `cl100k_base`            | tiktoken encoding (`token` only)    |
| `VECTOR_STORE`    | `vector_store`    | `memory`                 | Where vectors are stored            |
| `RERANKER`        | `reranker`        | `keyword`                | `keyword` or `none`                 |
| `GENERATOR`       | `generator`       | `grounded`               | How answers are written             |
| `LOG_LEVEL`       | –                 | `INFO`                   | Used by `setup_logging()`           |

`CHUNKER` is one of `word`, `character`, `token`, `recursive`, `sentence`,
`paragraph`, `heading` or `page`. `CHUNK_SIZE` and `CHUNK_OVERLAP` count words
for `word`, model tokens for `token`, and characters for all the others. Change them together
when you switch: 150 words is roughly 200 tokens or 900 characters, so for
`recursive`, `paragraph` and `heading` try `CHUNK_SIZE=1000`, for `sentence`
`500`, and for `page` `2000`. `CHUNKER=token` needs
`pip install "rag-titan[tokens]"`; to use your own tokenizer, see
[Chunking methods](chunking.md#your-own-tokenizer).

## Changing settings in code

Pass a dict. Your values win over the environment and the defaults.

```python
from rag_titan import RAGTitan, get_settings

rag = RAGTitan(settings={"llm_model": "llama3.2", "chunk_size": 200})

settings = get_settings({"reranker": "none"})   # full settings dict
```

## Logging

`rag-titan` prints nothing until you ask it to:

```python
from rag_titan import setup_logging

setup_logging("DEBUG")   # DEBUG, INFO or WARNING
```
