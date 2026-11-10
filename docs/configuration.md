# Configuration

Settings are read from **environment variables** or a **`.env` file** in the
folder where you run your code. Every setting has a default, so nothing is required.

Copy the example file to start:

```bash
cp .env.example .env
```

## All settings

| Variable          | Settings key      | Default                  | Meaning                            |
|-------------------|-------------------|--------------------------|------------------------------------|
| `LLM_PROVIDER`    | `llm_provider`    | `ollama`                 | Who runs the chat model            |
| `LLM_MODEL`       | `llm_model`       | `gemma3:4b`              | Chat model for answers             |
| `OLLAMA_HOST`     | `ollama_host`     | `http://localhost:11434` | Ollama server address              |
| `AGENT_MODEL`     | `agent_model`     | `qwen3:4b`               | Model for agents (needs tools)     |
| `AGENT_THINK`     | `agent_think`     | `true`                   | Turn on thinking for agents        |
| `AGENT_MAX_STEPS` | `agent_max_steps` | `5`                      | Max agent think/tool rounds        |
| `EMBED_PROVIDER`  | `embed_provider`  | `ollama`                 | Who runs the embedding model       |
| `EMBED_MODEL`     | `embed_model`     | `nomic-embed-text`       | Embedding model                    |
| `CHUNKER`         | `chunker`         | `word`                   | Chunking strategy                  |
| `CHUNK_SIZE`      | `chunk_size`      | `150`                    | Words per chunk                    |
| `CHUNK_OVERLAP`   | `chunk_overlap`   | `30`                     | Words shared by neighbouring chunks|
| `VECTOR_STORE`    | `vector_store`    | `memory`                 | Where vectors are stored           |
| `RERANKER`        | `reranker`        | `keyword`                | `keyword` or `none`                |
| `GENERATOR`       | `generator`       | `grounded`               | How answers are written            |
| `LOG_LEVEL`       | –                 | `INFO`                   | Used by `setup_logging()`          |

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
