"""Tests for the element-aware MarkdownChunker (chunking handbook 3.5 and 3.7)."""

import pytest
from markdown_docs import CODE, HANDBOOK, TABLE, random_document

from rag_titan.chunking import HeadingChunker, MarkdownChunker, ParagraphChunker
from rag_titan.chunking.markdown import glue_lead_ins, split_blocks
from rag_titan.chunking.markdown.block_pieces import code_lines, list_items, table_parts
from rag_titan.chunking.structure_aware import section_spans


def test_blocks_join_back_and_have_kinds():
    blocks = split_blocks(HANDBOOK)

    assert "".join(text for _, text in blocks) == HANDBOOK
    assert [kind for kind, _ in blocks] == ["heading", "paragraph", "heading", "paragraph",
                                            "list", "code", "quote", "heading", "table"]


def test_a_lead_in_joins_the_block_it_introduces():
    units = glue_lead_ins(split_blocks("Intro.\n\nYou get:\n- a laptop\n\nThe end:\n\n# Next\n"))

    assert units[1] == ("list", "You get:\n", "- a laptop\n\n")
    assert units[2] == ("paragraph", "", "The end:\n\n")       # a heading is not introduced


def test_code_with_blank_lines_and_hashes_stays_whole():
    chunks = MarkdownChunker(500).split_text(CODE)

    assert any("```bash\n# not a heading\n\nsudo setup --all\n```" in c for c in chunks)
    # the paragraph chunker cuts the same code block at its blank line
    assert not any("# not a heading\n\nsudo" in c for c in ParagraphChunker(40).split_text(CODE))


def test_a_big_table_repeats_its_header_and_never_cuts_a_row():
    chunks = MarkdownChunker(120).split_text(TABLE)
    header = "Error codes:\n| Code | Meaning |\n|------|---------|\n"

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith("Errors\n\n" + header)
        rows = chunk[len("Errors\n\n" + header):].splitlines()
        assert rows and all(row.startswith("| E-") and row.endswith("|") for row in rows)
    # the heading chunker cuts a row in half
    assert any(not c.rstrip().endswith("|") for c in HeadingChunker(120).split_text(TABLE))


def test_a_table_right_under_its_heading_repeats_its_header_too():
    # the commonest case: no lead-in, so the section body starts with a blank line
    text = "# Errors\n\n| Code | Meaning |\n|------|---------|\n" + "".join(
        f"| E-{n} | Something went wrong |\n" for n in range(1, 9))

    chunks = MarkdownChunker(120).split_text(text)

    assert len(chunks) > 1
    assert all(c.startswith("Errors\n\n| Code | Meaning |\n|------|---------|\n| E-")
               for c in chunks)


def test_the_offsets_of_a_repeated_header_point_at_the_rows():
    spans = MarkdownChunker(120).split_spans(TABLE)

    first, second = spans[0], spans[1]
    assert TABLE[first[1]:first[2]].startswith("Error codes:\n| Code |")
    assert TABLE[second[1]:second[2]].startswith("| E-")
    assert second[0].endswith(TABLE[second[1]:second[2]])


def test_small_chunks_cut_lists_at_items_and_code_at_lines():
    chunks = MarkdownChunker(70).split_text(HANDBOOK)

    assert "IT handbook > New laptop\n\nYou get:\n- a laptop" in chunks
    assert "IT handbook > New laptop\n\n- a headset\n  with a spare cable" in chunks
    assert all(chunk.split("\n\n", 1)[1] != "```" for chunk in chunks)   # no lone fence


def test_oversized_block_pieces_join_back():
    items = "- one\n  more\n- two\n  1. nested\n"
    assert list_items(items) == ["- one\n  more\n", "- two\n", "  1. nested\n"]
    assert code_lines("```py\na\nb\n```\n") == ["```py\na\n", "b\n```\n"]
    assert table_parts("| a |\n|---|\n| 1 |\n\n") == ("| a |\n|---|\n", ["| 1 |\n\n"])
    assert table_parts("| a |\n| 1 |\n") == ("| a |\n", ["| 1 |\n"])
    assert table_parts("\n| a |\n|---|\n| 1 |\n") == ("\n| a |\n|---|\n", ["| 1 |\n"])


def test_without_path_the_heading_lines_stay():
    chunks = MarkdownChunker(500, include_path=False).split_text(HANDBOOK)

    assert chunks[0].startswith("# IT handbook")


def test_overlap_repeats_whole_blocks():
    text = "One line here.\n\nTwo line here.\n\nThree line here.\n"
    chunks = MarkdownChunker(40, overlap=16).split_text(text)

    assert chunks == ["One line here.\n\nTwo line here.", "Two line here.\n\nThree line here."]


def test_chunk_gives_offsets_and_section_paths():
    chunks = MarkdownChunker(120).chunk([{"source": "it.md", "text": HANDBOOK}])

    laptop = [c for c in chunks if "a laptop" in c["text"]][0]
    assert laptop["section_path"] == ["IT handbook", "New laptop"]
    assert HANDBOOK[laptop["start_offset"]:laptop["end_offset"]].startswith("You get:")


@pytest.mark.parametrize("text", ["", "   \n\n  "])
def test_blank_text_gives_no_chunks(text):
    assert MarkdownChunker(100).split_text(text) == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (10, 10), (10, -1)])
def test_bad_sizes_raise(size, overlap):
    with pytest.raises(ValueError):
        MarkdownChunker(size, overlap)


def body_positions(text):
    """Every non-space position of the section bodies (heading lines are replaced by paths)."""
    positions = set()
    for _, body, start in section_spans(text):
        positions.update(start + i for i, char in enumerate(body) if not char.isspace())
    return positions


@pytest.mark.parametrize("seed", range(150))
def test_random_documents_keep_every_rule(seed):
    text, atoms = random_document(seed)
    size = 60 + seed * 3

    spans = MarkdownChunker(size, overlap=(seed % 3) * 10).split_spans(text)

    covered = set()
    for chunk, start, end in spans:
        assert len(chunk) <= size
        assert chunk.endswith(text[start:end]) and text[start:end] == text[start:end].strip()
        covered.update(range(start, end))
    assert body_positions(text) <= covered
    # a code block, a table row or a list item that fits is never cut
    for atom in atoms:
        if len(atom) <= size // 2:
            assert any(atom in chunk for chunk, _, _ in spans), (seed, atom)
