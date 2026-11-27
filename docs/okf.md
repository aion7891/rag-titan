# OKF knowledge bundles

The **Open Knowledge Format** ([OKF v0.3](https://github.com/open-knowledge-format/open-knowledge-format))
is a simple, open way to ship knowledge: a folder of Markdown files with YAML
frontmatter. Each file is a **concept** (a policy, a metric, a table, a playbook),
and its frontmatter can say where it came from, who checked it, whether it is
still current and when it must be checked again.

rag-titan reads OKF bundles, checks them against the spec, and indexes them so
that answers come from **current** knowledge and say **how far to trust** each
passage.

```bash
pip install "rag-titan[okf]"          # adds PyYAML, nothing else
```

```text
examples/okf_bundle/
  index.md                                  okf_version, title, licence of the bundle
  log.md                                    history of changes
  policies/travel/hotel-limit.md            a concept (type: Policy)
  policies/travel/hotel-limit-2025.md       its old version: status deprecated, superseded_by
  computations/overtime-pay.md              an Attested Computation
  references/attesters/overtime_check.py    an asset (rag-titan never runs it)
  README.md                                 no `type`: not a concept, skipped
```

## Check a bundle

```bash
rag-titan okf validate examples/okf_bundle            # errors and warnings; exit 1 on errors
rag-titan okf validate bundle.zip --strict            # warnings fail too (for CI)
rag-titan okf info examples/okf_bundle                # concepts by type, trust, status
```

```text
Acme employee handbook (OKF 0.3, CC-BY-4.0), 8 concepts
types: Policy 6, Attested Computation 1, Process 1
trust: human-reviewed 4, machine-confirmed 1, unverified 3
status: stable 4, draft 1, deprecated 3
stale: 1 (stale_after not readable: 0)
links between concepts: 3, broken: 0
other files: 1 skipped .md (no type), 1 assets
conformant with OKF v0.3 (0 errors, 1 warning: see rag-titan okf validate)
```

In Python: `from rag_titan.okf import validate_bundle, read_bundle`.

The spec is lenient on purpose: consumers must not reject unknown types, unknown
keys or broken links. So only what section 11.1 names makes a bundle **not
conformant**; the rest is a warning worth fixing.

| Errors (not conformant) | Warnings (still read) |
|---|---|
| frontmatter that is not a YAML mapping (bad YAML, a list, an unclosed `---`, duplicate keys) | `.md` files without `type` (not concepts), a missing or unknown `okf_version` |
| an `index.md` in a subfolder, or a `log.md`, with frontmatter | timestamps that are not RFC 3339, `verified` events without `by` and `at`, `sources` without `resource` |
| a `log.md` date heading that is not `YYYY-MM-DD` | an unknown `status`, `superseded_by` without `deprecated`, an Attested Computation without `runtime` |
| | broken links, relative paths that only work from the bundle root (the v0.2 style) |

## Index a bundle

Give the folder (or a `.zip` of it) to `index()`, with your other files if you like:

```python
from rag_titan import RAGTitan

rag = RAGTitan(settings={"chunker": "paragraph"})
rag.index(["examples/okf_bundle", "handbook.pdf"])
print(rag.ask("What is the maximum hotel cost per night?"))
```

```bash
rag-titan ask --files examples/okf_bundle --question "What is the maximum hotel cost per night?"
```

Each concept becomes one document: its title, its description, its body. Every
chunk keeps the concept's facts in `chunk["metadata"]` (concept id, type, tags,
status, trust tier, stale, and for an old concept what replaced it), also in a
saved index. The model sees them in the source line of each passage:

```text
[chunk_7] Source: policies/travel/hotel-limit.md, section: Hotel limit, trust: human-reviewed
[chunk_2] Source: it/laptop-refresh.md, section: Laptop refresh, trust: unverified, status: draft, stale: due for a check
```

| Setting | CLI | Default | What it does |
|---|---|---|---|
| `OKF_INCLUDE` | `--okf-include` | `current` | `current` leaves out deprecated concepts and unknown statuses (drafts stay, marked as drafts); `all` indexes every concept |
| `OKF_MIN_TRUST` | `--okf-min-trust` | `unverified` | leave out concepts below `machine-confirmed` or `human-reviewed` |

Both change what is indexed, so changing them indexes again.

## Trust, freshness and lifecycle

These come from the frontmatter, the way the spec says:

- **Trust tier.** Only *current* verification events count: a `verified` event at
  or after `generated.at`. When an agent rewrites a concept after a human review,
  the human did not read the new text, so the tier goes down. Only an actor
  written `human:<id>` gives **human-reviewed**; a `process:` or `team:` actor
  gives **machine-confirmed**; no current event means **unverified**.
- **Stale.** A concept is stale from its `stale_after` instant on. A value that
  cannot be read is never treated as fresh.
- **Status.** Missing means `stable`. `deprecated` (and any value OKF does not
  know) is not current; `superseded_by` is followed, over several hops, to the
  concept that replaced it.

Trust tiers are signals for the reader and the model, not access control.

## A real comparison

[`examples/okf_rag.py`](https://github.com/aion7891/rag-titan/blob/main/examples/okf_rag.py)
checks [`examples/okf_bundle/`](https://github.com/aion7891/rag-titan/tree/main/examples/okf_bundle),
a small handbook where three policies changed in 2026 and their old versions stay
in the bundle as deprecated concepts. It asks four questions whose answer changed,
once with current knowledge only and once with every concept.

```python
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
```

A real run (`gemma3:4b`, `nomic-embed-text`):

```text
Acme employee handbook: 8 concepts
  trust: {'human-reviewed': 4, 'machine-confirmed': 1, 'unverified': 3}
  status: {'stable': 4, 'draft': 1, 'deprecated': 3}, stale: 1
  conformant with OKF v0.3: True, 1 warning(s)

OKF_INCLUDE=current (5 chunks)
  What is the maximum hotel cost per night?
    current: The most a hotel night may cost on a business trip, from 2026 is **180 euros per night**, and up to 220 euros 
    read: policies/travel/hotel-limit.md (stable, human-reviewed), computations/overtime-pay.md (stable, human-reviewed)
  How much money do I get for home office furniture?
    current: You can claim up to 400 euros once for a desk and a chair [chunk_3].
    read: policies/remote/home-office.md (stable, machine-confirmed), policies/travel/hotel-limit.md (stable, human-reviewed)
  How many paid leave days do I get per year?
    current: Every full-time employee gets **25 paid leave days** per calendar year [chunk_2].
    read: policies/leave/annual-leave.md (stable, human-reviewed), computations/overtime-pay.md (stable, human-reviewed)
  What may a hotel night in London cost?
    current: Hotel costs are refunded up to 220 euros per night in London and Paris [chunk_4, page 1].
    read: policies/travel/hotel-limit.md (stable, human-reviewed), policies/remote/home-office.md (stable, machine-confirmed)

OKF_INCLUDE=all (8 chunks)
  What is the maximum hotel cost per night?
    current: The most a hotel night may cost on a business trip, from 2026, is up to 180 euros per night, and up to 220 eur
    read: policies/travel/hotel-limit.md (stable, human-reviewed), policies/travel/hotel-limit-2025.md (deprecated, human-reviewed)
  How much money do I get for home office furniture?
    both: Remote workers can claim up to 400 euros once [chunk_5] for a desk and a chair, or up to 300 euros once [chunk
    read: policies/remote/home-office.md (stable, machine-confirmed), policies/remote/home-office-2024.md (deprecated, unverified)
  How many paid leave days do I get per year?
    current: Every full-time employee gets 25 paid leave days per calendar year [chunk_3].
    read: policies/leave/annual-leave.md (stable, human-reviewed), policies/leave/annual-leave-2024.md (deprecated, unverified)
  What may a hotel night in London cost?
    current: Hotel costs are refunded up to 220 euros per night in London and Paris [chunk_7, page 1].
    read: policies/travel/hotel-limit.md (stable, human-reviewed), policies/travel/hotel-limit-2025.md (deprecated, human-reviewed)
```

What happened:

- **With current knowledge only** the old policies are not in the index: every
  answer gave the current value. In three runs, 12 of 12 answers did.
- **With every concept** the old version of the policy was one of the two passages
  the model read, for every question (the same passages in all three runs). The
  source line marks it `status: deprecated, replaced by ...`, and most answers
  still gave the current value, but in three runs 5 of 12 answers also gave the
  old one (*"... or up to 300 euros once"* in the run above): the home office question every time,
  the hotel question in two runs. No answer gave only the old value.
- Keep `OKF_INCLUDE=current` for questions about how things are now; use `all`
  when the history matters, and let the model see each passage's status.

## What is simplified

- rag-titan **reads** bundles; it does not write them or generate `index.md`.
- **Attested Computations** are indexed as text and validated, but their executors
  and attesters are never run, and no value is shown as attested.
- Links between concepts are resolved and checked (both `/root` and relative
  forms, never outside the bundle root), but search does not follow them yet.
- YAML is read with PyYAML and the YAML 1.2 rule for booleans (`yes`, `on`, `no`
  stay text), as OKF asks.
- Bundles of OKF v0.2 (or with no version) are read with the spec's fallback for
  paths written from the bundle root, and a warning says so.
- The OKF spec repository has no licence file at its root; rag-titan follows the
  spec and does not copy its text.
