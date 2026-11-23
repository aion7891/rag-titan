"""
Cut one Markdown block that is too big for a chunk, at its own natural lines
(the handbook's "defined fallback" for oversized atomic units):

    list      -> items (an item with its indented lines)
    code      -> lines (the fences stay with the first and the last line)
    quote     -> lines
    table     -> its header (first row and the |---| line) and its rows
    paragraph -> sentences (then words, done by the chunker)

Every function returns pieces that join back into the block exactly.
"""

import re

from rag_titan.chunking.markdown.blocks import LIST_ITEM, is_blank
from rag_titan.chunking.structure_aware.sentences import cut_after, split_sentences

# the line under a table's header: | --- | :---: | ---: |
TABLE_RULE = re.compile(r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")


def lines_with_blanks(text):
    """
    The lines of `text`; a blank line stays with the line before it, and blank
    lines at the very start (under a heading) stay with the first line.
    """
    pieces, leading = [], ""
    for line in text.splitlines(keepends=True):
        if not is_blank(line):
            pieces.append(leading + line)
            leading = ""
        elif pieces:
            pieces[-1] += line
        else:
            leading += line
    return pieces + [leading] if leading else pieces


def list_items(text):
    """Cut a list before every line that starts an item (nested items included)."""
    lines = text.splitlines(keepends=True)
    ends, position = [], 0
    for line in lines:
        if position and LIST_ITEM.match(line):
            ends.append(position)
        position += len(line)
    return cut_after(text, ends)


def code_lines(text):
    """Lines of a code block, so a ``` fence is never a chunk on its own."""
    lines = lines_with_blanks(text)
    if len(lines) >= 2:
        lines = [*lines[:-2], lines[-2] + lines[-1]]   # the closing fence
    if len(lines) >= 2:
        lines = [lines[0] + lines[1], *lines[2:]]      # the opening fence
    return lines


def table_parts(text):
    """
    Split a table into (header, rows).

    The header is the first row plus the |---| line under it when there is one;
    rows are the other lines (blank lines stay with the row before).
    """
    lines = lines_with_blanks(text)
    size = 2 if len(lines) > 1 and TABLE_RULE.match(lines[1].strip()) else 1
    return "".join(lines[:size]), lines[size:]


def block_pieces(kind, text):
    """The natural pieces of one block of `kind` (see the module docstring)."""
    if kind == "list":
        return list_items(text)
    if kind == "code":
        return code_lines(text)
    if kind in ("quote", "table"):
        return lines_with_blanks(text)
    return split_sentences(text)
