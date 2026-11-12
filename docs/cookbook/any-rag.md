# 4. Evaluate your own RAG

**Goal:** evaluate outputs from your existing RAG, regardless of its framework,
model or vector database. **Requirements:** the installation from
[recipe 1](first-evaluation.md) and access to your system's logged outputs.
Evaluation itself does not need access to the model or database.

## What you will see

![Evaluating a minimal file of your own logs](../assets/images/eval-terminal/own-rag.svg)

The terminal replays the **real output** of this recipe's commands on the starter data.

## 1. Start with the smallest useful dataset

Download [minimal.json](../assets/cookbook/minimal.json), or save the following as
`minimal.json` in your working folder. This is the complete runnable file:

```json
{
  "chunks": [
    {"id": "c1", "text": "Employees get 25 days of annual leave."}
  ],
  "queries": [
    {
      "id": "q1",
      "expected_chunk_ids": ["c1"],
      "retrieved_ids": ["c1"],
      "context_ids": ["c1"],
      "answer": "Employees get 25 days of annual leave.",
      "expected_answer": "Employees get 25 days of annual leave."
    }
  ]
}
```

```bash
rag-titan evaluate --input minimal.json --output minimal-report.json --k 1
python inspect_report.py minimal-report.json
```

Retrieval recall and answer F1 are **1.0**. Citation precision is **not measured**
because this file does not provide citations. Expect a provisional overall report:
there is much less measurement coverage than in the full starter example.

## 2. Map your actual logs

| Your system records | Evaluation key | Important detail |
|---|---|---|
| Every relevant indexed passage | `chunks` with `id` and `text` | Include gold chunks even if retrieval missed them |
| A stable question identifier | `queries[].id` | Unique per query; use the same IDs across comparisons |
| Search results in ranked order | `retrieved_ids` | Highest-ranked first, before reranking |
| Reranker output in ranked order | `reranked_ids` | Omit if there is no reranker |
| Chunks actually put into the prompt | `context_ids` | These may differ from search results |
| The actual generated text | `answer` | Do not replace it with your ideal answer |
| Human-reviewed reference answer | `expected_answer` | Supplies the target for answer metrics |
| Relevant chunk IDs | `expected_chunk_ids` | These are gold labels, not copied search results |
| Actual citation IDs / desired IDs | `citations` / `expected_citations` | Use IDs matching the chunks |
| Stage durations | `retrieval_latency_ms`, `generation_latency_ms` | Use milliseconds and the right stage |
| Total duration, cost and token usage | `latency_ms`, `cost_usd`, `total_tokens` | Measure them; leave unknown values absent |

A retrieved ID is an **observation**; an expected ID is a **label**. Copying retrieved
IDs into `expected_chunk_ids` would make a failed search appear correct. Likewise,
`verified_claims` and `task_success` are supplied judgements, not judgements the
package magically generates.

## 3. Evaluate from Python

Save this as `evaluate_my_logs.py` next to `minimal.json`:

```python
import json
from pathlib import Path

from rag_titan.evaluation import RAGEvaluator, ReportManager
from rag_titan.evaluation.reports import summary_text

data = json.loads(Path("minimal.json").read_text(encoding="utf-8"))
report = RAGEvaluator(k=1).evaluate(**data)

manager = ReportManager()
manager.save_json(report, "my-report.json")
manager.save_markdown(report, "my-report.md")
print(summary_text(report))
```

```bash
python evaluate_my_logs.py
```

Replace the content of `minimal.json` with your real observations and independently
reviewed labels. To evaluate many questions, append records to `queries`; keep one
entry per unique chunk in `chunks`.

## 4. Add measurements gradually

You do not need to implement everything at once:

1. **Retrieval:** chunk text, ranked IDs and expected IDs.
2. **Context and answers:** actual context IDs, answers and reference answers.
3. **Citations:** actual citation IDs and reviewed expected IDs.
4. **Chunking:** source documents, character offsets and expected evidence.
5. **Embeddings:** actual chunk/query vectors and index membership.
6. **Operations:** stage latency, end-to-end latency, cost and token usage.

For multiple documents, use `documents: {"handbook.md": "...", "expenses.md": "..."}`
and a matching `doc_id` on each chunk. `start` and `end` are character offsets into
that document, with `end` excluded. Do not mix token indices with character offsets.
See the complete [stage input format](../stage-evaluation.md) before adding these fields.

!!! warning "Keep missing data honest"
    Omit unavailable measurements instead of setting them to zero. A missing
    reference answer should not be an empty reference string. A missing latency
    should not be zero milliseconds. `None`/`null` in a report means “not measured”.

## 5. Generate a command for your files

<div class="cb-command" data-input="minimal.json" data-output="my-report.json">
<p>Without JavaScript, use: <code>rag-titan evaluate --input minimal.json --output my-report.json --k 3</code>.</p>
</div>

When evaluating a new run, save its input data as well as its report. That lets you
reproduce a score without rerunning an expensive model and inspect individual
failures later.

**Next:** [prepare better gold labels](golden-set.md) and
[compare against a baseline](regressions.md).
