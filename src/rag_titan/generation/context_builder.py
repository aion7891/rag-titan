"""Helper that turns chunks into one text block for the LLM."""


def build_context(chunks):
    """
    Join chunks into one text block the LLM can read and cite.

    Each chunk looks like:
        [chunk_3] Source: handbook.pdf
        ...chunk text...
    """
    passages = []

    for chunk in chunks:
        passages.append(
            f"[{chunk['id']}] Source: {chunk['source']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(passages)
