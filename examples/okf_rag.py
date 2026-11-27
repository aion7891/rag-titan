"""
Check and index an Open Knowledge Format (OKF) bundle, then ask about policies
that changed: does the answer give the current value or the old one?

examples/okf_bundle/ holds three policies that changed in 2026. Their old
versions stay in the bundle, marked `status: deprecated`, with `superseded_by`
pointing to the new ones. OKF_INCLUDE=current (the default) leaves them out;
OKF_INCLUDE=all indexes every concept.

Before running (Ollama installed and running):
    pip install "rag-titan[okf]"
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/okf_rag.py
"""

import json
import re
from pathlib import Path

from rag_titan import RAGTitan
from rag_titan.okf import validate_bundle

HERE = Path(__file__).parent
BUNDLE = HERE / "okf_bundle"
questions = json.loads((HERE / "okf_questions.json").read_text(encoding="utf-8"))


def says(answer, number):
    """True when `number` appears as a whole number ("20", not inside "2026" or "220")."""
    return re.search(rf"(?<!\d){number}(?!\d)", answer) is not None


def which_value(answer, question):
    """Does the answer give the current value, the old one, both or neither?"""
    current, old = says(answer, question["current"]), says(answer, question["old"])
    return {(True, False): "current", (False, True): "OLD", (True, True): "both"}.get(
        (current, old), "neither")


# 1. Check the bundle (no model needed)
report = validate_bundle(BUNDLE)
summary = report["summary"]
print(f"{report['metadata']['title']}: {summary['concepts']} concepts")
print(f"  trust: {summary['trust_tiers']}")
print(f"  status: {summary['statuses']}, stale: {summary['stale']}")
print(f"  conformant with OKF v0.3: {report['conformant']}, "
      f"{len(report['warnings'])} warning(s)")

# 2. The same questions with current knowledge only, then with every concept
for include in ["current", "all"]:
    rag = RAGTitan(settings={"okf_include": include, "chunker": "paragraph",
                             "chunk_size": 600, "top_k": 3, "top_n": 2})
    rag.index([BUNDLE])
    print(f"\nOKF_INCLUDE={include} ({rag.count()} chunks)")
    for question in questions:
        result = rag.run(question["question"])
        read = ", ".join(f"{chunk['source']} ({chunk['metadata']['status']}, "
                         f"{chunk['metadata']['trust_tier']})" for chunk in result["sources"])
        print(f"  {question['question']}")
        print(f"    {which_value(result['answer'], question)}: {result['answer'].strip()[:110]}")
        print(f"    read: {read}")
