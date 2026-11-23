"""
Read Markdown text as blocks: the units the markdown chunker keeps whole
(chunking handbook 3.5 and 3.7: "keep lists, code blocks, tables and callouts intact").

    split_blocks("You get:\\n- a laptop\\n- a headset\\n\\n| Code | Meaning |\\n...")
    -> [("paragraph", "You get:\\n"), ("list", "- a laptop\\n- a headset\\n\\n"),
        ("table", "| Code | Meaning |\\n...")]

Kinds: heading, paragraph, list, table, code, quote. Each block keeps the blank
lines after it, so joining the texts gives back the text exactly. Simple line
rules, no Markdown library: fenced code, "|" table rows, "-", "*", "+" and "1."
list items, "> " quotes and "#" headings.
"""

import re

from rag_titan.chunking.headings import FENCE, HEADING

# "- item", "* item", "+ item", "1. item", "2) item" (indented items are nested ones)
LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d{1,3}[.)])[ \t]+\S")
TABLE_ROW = re.compile(r"^[ \t]*\|")
QUOTE = re.compile(r"^[ \t]*>")

# blocks a lead-in line ("You get:") is kept with
LED_KINDS = ("list", "table", "code")


def is_blank(line):
    """True for an empty line or one with only spaces."""
    return not line.strip()


def is_indented(line):
    """True for a line that starts with a space or a tab and has text."""
    return line[:1] in (" ", "\t") and not is_blank(line)


def block_kind(line):
    """The kind of block that `line` starts."""
    if FENCE.match(line):
        return "code"
    if HEADING.match(line.rstrip("\r\n")):
        return "heading"
    if TABLE_ROW.match(line):
        return "table"
    if LIST_ITEM.match(line):
        return "list"
    if QUOTE.match(line):
        return "quote"
    return "paragraph"


def fence_end(lines, start):
    """The line after the fence that closes the code block opened at lines[start]."""
    for end in range(start + 1, len(lines)):
        # the same rule as the heading chunker: any fence line closes the block
        if FENCE.match(lines[end]):
            return end + 1
    return len(lines)   # an unclosed fence runs to the end, as in Markdown


def list_end(lines, start):
    """
    The line after a list: its items, their indented or wrapped lines, and the
    blank lines between items (a blank line ends it only when no item follows).
    """
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if is_blank(line):
            following = next((other for other in lines[end:] if not is_blank(other)), "")
            if not (LIST_ITEM.match(following) or is_indented(following)):
                break
        elif not (LIST_ITEM.match(line) or is_indented(line)
                  or block_kind(line) == "paragraph"):
            break   # a table, code block, quote or heading ends the list
        end += 1
    return end


def block_end(lines, start, kind):
    """The line after the block of `kind` that starts at lines[start]."""
    if kind == "heading":
        return start + 1
    if kind == "code":
        return fence_end(lines, start)
    if kind == "list":
        return list_end(lines, start)

    end = start + 1
    while end < len(lines) and not is_blank(lines[end]):
        follows = block_kind(lines[end])
        # tables and quotes continue with their own lines; a paragraph ends
        # where another block starts ("You get:" then "- a laptop")
        if follows != kind:
            break
        end += 1
    return end


def split_blocks(text):
    """
    Cut Markdown text into blocks.

    Returns:
        A list of (kind, text) pairs; "".join(texts) == text. Blank lines at the
        very start belong to the first block.
    """
    lines = text.splitlines(keepends=True)
    blocks, start, leading = [], 0, ""

    while start < len(lines):
        if is_blank(lines[start]):
            if blocks:
                blocks[-1][1] += lines[start]
            else:
                leading += lines[start]
            start += 1
            continue
        kind = block_kind(lines[start])
        end = block_end(lines, start, kind)
        blocks.append([kind, leading + "".join(lines[start:end])])
        start, leading = end, ""

    return [(kind, block) for kind, block in blocks]


def glue_lead_ins(blocks):
    """
    Keep a lead-in with what it introduces: a paragraph ending in ":" joins the
    list, table or code block right after it.

    Returns:
        A list of (kind, lead, body): lead is the lead-in text ("" for none),
        body the block's own text; lead + body is the original text.
    """
    units = []
    for kind, text in blocks:
        previous = units[-1] if units else None
        leads = (previous is not None and previous[0] == "paragraph" and not previous[1]
                 and previous[2].rstrip().endswith(":"))
        if leads and kind in LED_KINDS:
            units[-1] = (kind, previous[2], text)
        else:
            units.append((kind, "", text))
    return units
