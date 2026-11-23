"""Tests for the TextTiling chunker (Hearst 1997; chunking handbook 5.1)."""

import pytest
from tiling_docs import THREE_TOPICS, planted_text

from rag_titan.chunking import TextTilingChunker
from rag_titan.chunking.statistical import boundaries, content_words
from rag_titan.chunking.statistical.cohesion import cosine, depth
from rag_titan.chunking.statistical.texttiling import snap_to_paragraphs


def test_a_planted_topic_change_is_found():
    chunks = TextTilingChunker(2000).split_text(THREE_TOPICS)

    assert [chunk.split()[:3] for chunk in chunks] == [
        ["The", "new", "laptop"], ["Travel", "is", "booked"], ["Leave", "days", "are"]]


def test_the_tiling_shows_the_scores_behind_the_cuts():
    tiling = TextTilingChunker(2000).tiling(THREE_TOPICS)

    assert tiling["cuts"] == [4, 8]
    gaps = tiling["gaps"]
    assert [gap["gap"] for gap in gaps] == list(range(1, 12))
    assert gaps[0]["score"] is None and gaps[-1]["score"] is None   # no full window there
    assert gaps[3]["score"] == 0.0 and gaps[3]["depth"] > 0.4       # laptop | travel


def test_content_words_drop_stopwords_and_case():
    assert content_words("The Laptop is in the DOCK.") == ["laptop", "dock"]


def test_cosine_and_depth_by_hand():
    left, right = {"laptop": 2, "dock": 1}, {"laptop": 1, "hotel": 1}
    assert cosine(left, right) == pytest.approx(2 / (5 ** 0.5 * 2 ** 0.5))
    assert cosine({}, right) is None
    # a valley at 0.1 between peaks 0.5 (left, climbing past 0.4) and 0.3
    assert depth([0.5, 0.4, 0.1, 0.3, 0.2], 2) == pytest.approx(0.4 + 0.2)


def test_text_of_stopwords_only_is_never_cut():
    text = "It is what it is. And so it was. It is. Was it? It was so. And it is. So it is."

    assert TextTilingChunker(500).tiling(text)["cuts"] == []
    assert len(TextTilingChunker(500).split_text(text)) == 1


def test_fewer_than_two_windows_of_sentences_are_never_cut():
    text = "Laptop dock. Laptop charger. Hotel train. Hotel flight. Tax bank."

    assert all(gap["score"] is None for gap in TextTilingChunker(500).tiling(text)["gaps"])
    assert len(TextTilingChunker(500).split_text(text)) == 1
    assert boundaries([], 3) == ([], [])


def test_a_cut_moves_onto_a_nearby_paragraph_break():
    pieces = ["One. ", "Two.\n\n", "Three. ", "Four. ", "Five.\n\n", "Six."]

    assert snap_to_paragraphs([3], pieces) == [2]        # break after "Two." is one before
    assert snap_to_paragraphs([4], pieces) == [5]        # break after "Five." is one after
    assert snap_to_paragraphs([2, 3], pieces) == [2]     # two cuts on one break become one
    assert snap_to_paragraphs([1], ["a. ", "b. ", "c. "]) == [1]


def test_a_long_topic_is_packed_and_never_mixed_with_the_next():
    chunks = TextTilingChunker(120).split_text(THREE_TOPICS)

    assert all(len(chunk) <= 120 for chunk in chunks)
    assert not any("repair" in chunk and "Travel" in chunk for chunk in chunks)
    assert not any("finance" in chunk and "Leave" in chunk for chunk in chunks)


def test_overlap_stays_inside_a_topic():
    chunks = TextTilingChunker(150, overlap=60).split_text(THREE_TOPICS)

    # sentences are repeated between chunks of one topic ...
    assert any(chunks[i].split(". ")[-1] in chunks[i + 1] for i in range(len(chunks) - 1))
    # ... but each topic's first chunk starts clean, with nothing of the topic before
    starts = [chunk.split(". ")[0] for chunk in chunks]
    assert "Travel is booked through the travel portal" in starts
    assert "Leave days are requested in the leave tool" in starts
    assert not any("repair" in chunk and "Travel" in chunk for chunk in chunks)


def test_chunk_gives_exact_offsets():
    chunks = TextTilingChunker(2000).chunk([{"source": "notes.txt", "text": THREE_TOPICS}])

    for chunk in chunks:
        assert THREE_TOPICS[chunk["start_offset"]:chunk["end_offset"]] == chunk["text"]


@pytest.mark.parametrize("window", [0, -1, 1.5, True, "3", None])
def test_a_bad_window_raises(window):
    with pytest.raises(ValueError, match="window must be a whole number of at least 1"):
        TextTilingChunker(500, window=window)


@pytest.mark.parametrize("text", ["", "  \n\n "])
def test_blank_text_gives_no_chunks(text):
    assert TextTilingChunker(500).split_text(text) == []


@pytest.mark.parametrize("seed", range(120))
def test_random_planted_texts_keep_every_rule(seed):
    text, changes = planted_text(seed)
    chunker = TextTilingChunker(80 + seed * 4, overlap=(seed % 3) * 15, window=1 + seed % 4)
    tiling = chunker.tiling(text)

    spans = chunker.split_spans(text)
    assert spans == chunker.split_spans(text)                       # same result every time
    covered = set()
    for chunk, start, end in spans:
        assert len(chunk) <= chunker.chunk_size and text[start:end] == chunk
        covered.update(range(start, end))
    assert {i for i, char in enumerate(text) if not char.isspace()} <= covered
    # no chunk runs across a cut
    starts, position = [], 0
    for piece in tiling["sentences"]:
        starts.append(position)
        position += len(piece)
    edges = {starts[cut] for cut in tiling["cuts"]}
    assert not any(start < edge < end for _, start, end in spans for edge in edges)
    scores = [gap["score"] for gap in tiling["gaps"] if gap["score"] is not None]
    assert all(0 <= score <= 1 + 1e-12 for score in scores)


def test_planted_changes_are_mostly_found():
    found = planted = 0
    for seed in range(120):
        text, changes = planted_text(seed)
        cuts = TextTilingChunker(5000).tiling(text)["cuts"]
        planted += len(changes)
        found += sum(any(abs(cut - change) <= 1 for cut in cuts) for change in changes)

    assert found / planted >= 0.9
