"""Markdown test documents: fixed ones with every element, and seeded random ones."""

import random

HANDBOOK = """# IT handbook

Intro paragraph about the handbook.

## New laptop

You get:
- a laptop
- a headset
  with a spare cable

1. Open the portal.
2. Pick a model.

```bash
# not a heading

sudo setup --all
```

> **Warning** Never share your password.

## Error codes

| Code | Meaning |
|------|---------|
| E-1 | VPN down |
| E-2 | Password old |
| E-3 | Disk full |
| E-4 | Printer offline |
"""

CODE = """Run this once:

```bash
# not a heading

sudo setup --all
```
"""

TABLE = "# Errors\n\nError codes:\n| Code | Meaning |\n|------|---------|\n" + "".join(
    f"| E-{n} | {meaning} |\n" for n, meaning in enumerate(
        ["VPN down", "Password too old", "Disk full", "Printer offline", "Badge blocked",
         "Account locked", "Mailbox full", "Licence expired"], start=1))

WORDS = ("leave days laptop printer error vpn badge desk chair monitor policy manager "
         "office travel hotel note doctor week remote equipment").split()


def words(rng, count):
    return " ".join(rng.choice(WORDS) for _ in range(count))


def sentence(rng):
    text = words(rng, rng.randint(3, 12))
    return text[0].upper() + text[1:] + rng.choice([".", ".", "?", "!"])


def list_block(rng, atoms):
    items = []
    for _ in range(rng.randint(1, 6)):
        item = rng.choice(["- ", "* ", "1. ", "2) "]) + words(rng, rng.randint(1, 6))
        if rng.random() < 0.3:
            item += "\n  " + words(rng, rng.randint(1, 4))   # a wrapped line of the item
        items.append(item)
    atoms.extend(items)
    return "\n".join(items) + "\n\n"


def table_block(rng, atoms):
    rows = [f"| E-{rng.randint(1, 99)} | {words(rng, rng.randint(1, 3))} |"
            for _ in range(rng.randint(1, 8))]
    atoms.extend(rows)
    return "| Code | Meaning |\n|------|---------|\n" + "\n".join(rows) + "\n\n"


def code_block(rng, atoms):
    lines = [rng.choice(["# a comment", "x = 1", "", "run --all", "## not a heading"])
             for _ in range(rng.randint(1, 4))]
    block = "```\n" + "\n".join(lines) + "\n```"
    atoms.append(block)
    return block + "\n\n"


def random_document(seed):
    """
    A Markdown document full of traps, and its atoms: the list items, table rows
    and code blocks that must never be cut.
    """
    rng, parts, atoms = random.Random(seed), [], []
    for _ in range(rng.randint(3, 10)):
        kind = rng.choice(["heading", "paragraph", "paragraph", "list", "table", "code",
                           "quote", "lead"])
        if kind == "heading":
            parts.append("#" * rng.randint(1, 3) + " " + words(rng, 2).title() + "\n\n")
        elif kind == "paragraph":
            parts.append(" ".join(sentence(rng) for _ in range(rng.randint(1, 6))) + "\n\n")
        elif kind == "quote":
            parts.append("> **Warning** " + sentence(rng) + "\n\n")
        elif kind == "lead":
            parts.append("You get:\n" + list_block(rng, atoms))
        else:
            parts.append({"list": list_block, "table": table_block,
                          "code": code_block}[kind](rng, atoms))
    return "".join(parts), atoms
