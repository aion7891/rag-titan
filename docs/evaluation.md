# Evaluating retrieval

A RAG answer can only be as good as the chunks it is given. Before you change
the chunker, the retriever or the RAG type, **measure** how well retrieval finds
the right evidence for your own questions, then compare. No method is best for
every set of documents.

```python
from rag_titan.evaluation import compare_retrievers, format_report, load_questions

questions = load_questions("questions.json")
results = compare_retrievers({"dense": retriever}, questions, top_k=5)
print(format_report(results))
```

```text
Retrieval report: 6 questions, top_k=3

retriever        Recall@3       MRR    nDCG@3    p50 ms    p95 ms
-----------------------------------------------------------------
dense               1.000     1.000     0.987       7.0       8.5
dense_no_dedupe     1.000     1.000     0.987       7.1       8.4
```

This is the real output of `python examples/evaluate_retrieval.py`. It indexes
three short policies in `examples/sample_docs/` with Ollama embeddings and
compares the default retriever with one that keeps near-copies.

## The questions file

A JSON list. Each record says which evidence a good search must find:

```json
[
  {"id": "q_001",
   "question": "How many paid leave days do I get each year?",
   "required_evidence_ids": ["leave_policy.md"],
   "task_type": "fact"},
  {"id": "q_006",
   "question": "What do I get for working from home: equipment and money for furniture?",
   "required_evidence_ids": ["remote_work.md", "expenses.md"],
   "task_type": "multi_hop"}
]
```

| Key | Needed | Meaning |
|---|---|---|
| `id` | yes | a unique name, used in error messages and in the rows |
| `question` | yes | what to search for |
| `required_evidence_ids` | yes | one or more evidence ids (below); a multi-hop question lists several |
| `task_type` | no | your own label, like `fact`, `semantic`, `multi_hop` |
| anything else | no | kept as it is (for example `answer` or `tenant`) |

`load_questions()` checks every record and says which one is wrong:

```text
ValueError: Question 'q_007': Evidence id 'hr.pdf#4': write the page as '#p4'
```

### Evidence ids

| Evidence id | A chunk is evidence when |
|---|---|
| `"hr.pdf"` | it comes from `hr.pdf` |
| `"hr.pdf#p4"` | it comes from `hr.pdf` and its `page_numbers` include 4 |

Evidence ids name files and pages, **not chunk ids**. Chunk ids (`chunk_0`,
`chunk_1`, ...) change whenever you change the chunker, so labels written with
them would break. With files and pages you can compare chunkers on one file.

Each evidence counts **once**: when two overlapping chunks come from the same
page, only the first one counts, so near-copies never make a retriever look better.

## The metrics

Every metric is from 0 (bad) to 1 (perfect), computed for each question on the
first $k$ results, then averaged.

| Metric | Formula | Answers the question |
|---|---|---|
| **Recall@k** | $\dfrac{\lvert \text{Relevant} \cap \text{Retrieved}_k \rvert}{\lvert \text{Relevant} \rvert}$ | did we find **all** the evidence? |
| **MRR** | $\dfrac{1}{N} \sum_q \dfrac{1}{\text{rank}_q}$ | how high is the **first** piece of evidence? |
| **nDCG@k** | $\dfrac{\sum_r g_r / \log_2(r+1)}{\text{IDCG}}$ | is the evidence **near the top**? |
| **p50 / p95 ms** | median and 95th percentile time per search | how fast, including the slow searches |

For nDCG, $g_r$ is the number of new evidences found at rank $r$, and IDCG is
the best possible score: one new evidence at each of the first
$\min(\lvert \text{Relevant} \rvert, k)$ ranks. One chunk can hold two
evidences (pages 4 and 5), so the score is capped at 1.

**Worked example.** A question needs `a.pdf#p4` and `c.txt`. The search returns:

| Rank | Chunk from | New evidence |
|---|---|---|
| 1 | `a.pdf` page 1 | – |
| 2 | `b.pdf` page 2 | – |
| 3 | `a.pdf` page 4 | `a.pdf#p4` |
| 4 | `a.pdf` page 4 again | – (already found) |
| 5 | `c.txt` | `c.txt` |

- Recall@5 $= 2/2 = 1.0$, but Recall@3 $= 1/2 = 0.5$
- reciprocal rank $= 1/3$ (first evidence at rank 3)
- nDCG@5 $= \dfrac{1/\log_2 4 + 1/\log_2 6}{1 + 1/\log_2 3} = \dfrac{0.887}{1.631} \approx 0.544$

Latency is reported as p50 and p95, not only an average: the average hides the
few slow searches that users notice.

## The results in Python

`evaluate_retriever()` returns the averages and one row per question, so you
can find the questions that fail:

```python
from rag_titan.evaluation import evaluate_retriever

result = evaluate_retriever(retriever, questions, top_k=5)
result["recall"], result["mrr"], result["ndcg"], result["p95_ms"]

for row in result["rows"]:
    if row["recall"] < 1:
        print(row["id"], row["task_type"], row["sources"])
```

`compare_retrievers({"name": retriever, ...}, questions, top_k)` runs every
retriever on **the same questions with the same top_k**, and returns one such
result per retriever.

## Tips

- Write **30 to 100 questions** for each kind of question that matters to you.
  A handful of questions is a smoke test, not proof that one method is better.
- Include questions that need **several** pieces of evidence: a single label per
  question hides weak multi-hop retrieval.
- Use page labels (`#p4`) for long PDFs: a file label is too easy to match.
- Compare under identical conditions: same files, chunker, embedder and `top_k`.

What is measured today is **retrieval**. Scoring the final answers (correctness,
citations) is planned; see the [roadmap](roadmap.md).
