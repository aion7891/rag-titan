"""Helper that turns chunks into one text block for the LLM."""


def describe_source(chunk):
    """
    The header line of one chunk: its id, file, and page/section when known.

    Example:
        [chunk_4] Source: handbook.pdf, page 3, section: Leave policy > Sick leave
        [chunk_7] Source: metrics/margin.md, trust: human-reviewed      (an OKF concept)
    """
    parts = [str(chunk["source"])]

    pages = chunk.get("page_numbers")
    if pages:
        parts.append(f"page {pages[0]}" if len(pages) == 1 else f"pages {pages[0]}-{pages[-1]}")

    if chunk.get("section_path"):
        parts.append("section: " + " > ".join(chunk["section_path"]))

    parts += okf_notes(chunk.get("metadata") or {})
    return f"[{chunk['id']}] Source: {', '.join(parts)}"


def okf_notes(metadata):
    """
    What the model should know about an OKF concept: its trust tier, and
    whether it is a draft, deprecated (and by what replaced) or due for a check.
    Example: ["trust: human-reviewed", "status: deprecated, replaced by metrics/new"]
    """
    if "okf_concept" not in metadata:
        return []
    notes = [f"trust: {metadata.get('trust_tier', 'unverified')}"]
    status = metadata.get("status", "stable")
    if status != "stable":
        replaced = metadata.get("replaced_by")
        notes.append(f"status: {status}" + (f", replaced by {replaced}" if replaced else ""))
    if metadata.get("stale"):
        notes.append("stale: due for a check")
    return notes


def build_context(chunks):
    """
    Join chunks into one text block the LLM can read and cite.

    Each chunk looks like:
        [chunk_3] Source: handbook.pdf, page 2
        ...chunk text...
    """
    return "\n\n".join(f"{describe_source(chunk)}\n{chunk['text']}" for chunk in chunks)
