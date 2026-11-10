# Command line

Installing the package adds a `rag-titan` command.

![rag-titan command line demo](assets/images/cli.svg)

## Ask one question

```bash
rag-titan ask --files handbook.pdf --question "How many leave days do I get?"
```

## Chat in a loop

```bash
rag-titan chat --files handbook.pdf policy.docx --rag-type adaptive
```

Type `exit` to quit.

## See every step

```bash
rag-titan ask --files handbook.pdf --question "How many leave days do I get?" --trace
```

Prints every step the RAG type took (search results with their files and
scores, evidence checks, rewrites) before the answer.

## Evaluate and compare

```bash
rag-titan evaluate --files docs/*.md --questions questions.json \
    --rag-type simple adaptive --chunker word paragraph
```

Runs every RAG type with every chunker on your test questions and prints a
report, then the questions that missed evidence or got no answer. Add
`--retrieval-only` to score search alone (fast, no LLM). See
[Evaluating RAG](evaluation.md).

## Options

| Option        | Used by     | Meaning                                   |
|---------------|-------------|-------------------------------------------|
| `--files`     | ask, chat, evaluate | PDF, DOCX, MD or TXT files (required) |
| `--rag-type`  | ask, chat, evaluate | `simple` (default) or `adaptive`; evaluate takes several |
| `--log-level` | ask, chat, evaluate | `DEBUG`, `INFO` or `WARNING` (default) |
| `--question`  | ask         | the question to answer (required)         |
| `--trace`     | ask         | also print every step taken               |
| `--questions` | evaluate    | questions JSON file (required)            |
| `--chunker`   | evaluate    | one or more chunkers to compare           |
| `--chunk-size`, `--chunk-overlap` | evaluate | used by every setup  |
| `--retrieval-only` | evaluate | score search only, no LLM             |
| `--top-k`     | evaluate    | chunks per search with `--retrieval-only` (default 5) |
| `--version`   | –           | print the version                         |

The CLI also reads your `.env` file, see [Configuration](configuration.md).
