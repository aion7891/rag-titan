"""The markdown and texttiling chunkers are wired everywhere: CHUNKERS, .env, the CLI, RAGTitan."""

import argparse

import pytest

from rag_titan import RAGTitan, cli, get_settings
from rag_titan.chunking import MarkdownChunker, TextTilingChunker
from rag_titan.cli_options import add_tuning_options, tuning_settings
from rag_titan.evaluation.end_to_end import SEARCH_ONLY
from rag_titan.factory import CHUNKERS, build_chunker


def test_both_names_are_known():
    assert CHUNKERS[-2:] == ["markdown", "texttiling"]


def test_build_chunker_makes_them_with_the_settings(monkeypatch):
    monkeypatch.delenv("TEXTTILING_WINDOW", raising=False)
    markdown = build_chunker(get_settings({"chunker": "markdown", "chunk_size": 800,
                                           "chunk_overlap": 100}))
    tiling = build_chunker(get_settings({"chunker": "texttiling", "chunk_size": 900,
                                         "chunk_overlap": 0}))

    assert isinstance(markdown, MarkdownChunker)
    assert (markdown.chunk_size, markdown.overlap, markdown.include_path) == (800, 100, True)
    assert isinstance(tiling, TextTilingChunker)
    assert (tiling.chunk_size, tiling.window) == (900, 3)


def test_settings_without_the_window_use_the_default():
    settings = {"chunker": "texttiling", "chunk_size": 500, "chunk_overlap": 0}

    assert build_chunker(settings).window == 3


def test_the_window_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("TEXTTILING_WINDOW", "5")

    settings = get_settings({"chunker": "texttiling", "chunk_size": 500, "chunk_overlap": 0})

    assert settings["texttiling_window"] == 5 and build_chunker(settings).window == 5


def test_a_bad_window_says_which_setting(monkeypatch):
    monkeypatch.setenv("TEXTTILING_WINDOW", "wide")
    with pytest.raises(ValueError, match="Setting TEXTTILING_WINDOW must be a whole number"):
        get_settings()

    monkeypatch.setenv("TEXTTILING_WINDOW", "0")
    with pytest.raises(ValueError, match="window must be a whole number of at least 1"):
        build_chunker(get_settings({"chunker": "texttiling", "chunk_size": 500,
                                    "chunk_overlap": 0}))


@pytest.mark.parametrize("search", [True, False])   # False: `rag-titan index`
def test_the_cli_flag_is_an_indexing_option(search):
    parser = argparse.ArgumentParser()
    add_tuning_options(parser, search=search)

    assert tuning_settings(parser.parse_args(["--texttiling-window", "4"])) == {
        "texttiling_window": 4}
    with pytest.raises(SystemExit):
        parser.parse_args(["--texttiling-window", "0"])


def test_every_command_takes_the_flag():
    parser = cli.build_parser()
    for command in (["ask", "--question", "q"], ["chat"], ["index"],
                    ["evaluate", "--questions", "q.json"]):
        args = parser.parse_args([*command, "--files", "a.md", "--texttiling-window", "2"])
        assert args.texttiling_window == 2, command


def test_a_new_window_rebuilds_the_index():
    # the window changes the chunks, so evaluate must not reuse an index for it
    assert "texttiling_window" not in SEARCH_ONLY


@pytest.mark.parametrize("chunker", ["markdown", "texttiling"])
def test_ragtitan_indexes_with_them(tmp_path, monkeypatch, chunker):
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "it.md").write_text("# IT\n\nYou get:\n- a laptop\n- a headset\n\n"
                                    "| Code | Meaning |\n|---|---|\n| E-1 | VPN down |\n",
                                    encoding="utf-8")
    rag = RAGTitan(settings={"chunker": chunker, "chunk_size": 400, "chunk_overlap": 0})

    count = rag.index([tmp_path / "it.md"])

    assert count >= 1
    assert isinstance(rag.pipeline.chunker, (MarkdownChunker, TextTilingChunker))
