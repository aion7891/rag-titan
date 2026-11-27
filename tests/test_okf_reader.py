"""Tests for reading OKF bundles: frontmatter (OKF v0.3 section 4) and bundle files (3, 8)."""

import os
import sys
import zipfile
from datetime import date

import pytest

import rag_titan.okf.frontmatter as frontmatter_module
from rag_titan.okf import parse_frontmatter, read_bundle
from rag_titan.okf.frontmatter import split_frontmatter

CONCEPT = ("---\ntype: Metric\ntitle: Gross margin\ntags: [finance]\n---\n"
           "# Definition\nRevenue minus cost.\n")


def write(root, files):
    """Write {path: text or bytes} under root and return root."""
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return root


def messages(bundle, level=None):
    return [(p["path"], p["message"]) for p in bundle["problems"]
            if level is None or p["level"] == level]


# --- frontmatter -----------------------------------------------------------------------

def test_frontmatter_and_body_are_split():
    data, body = parse_frontmatter(CONCEPT)

    assert data == {"type": "Metric", "title": "Gross margin", "tags": ["finance"]}
    assert body == "# Definition\nRevenue minus cost.\n"


@pytest.mark.parametrize("text", ["# Just Markdown\n", "--- \ntype: X\n---\n",
                                  " ---\ntype: X\n---\n", ""])
def test_no_frontmatter_unless_the_first_line_is_exactly_three_dashes(text):
    assert parse_frontmatter(text) == (None, text)


def test_windows_line_endings_and_a_byte_order_mark_are_fine():
    assert parse_frontmatter("﻿---\r\ntype: Metric\r\n---\r\nbody")[0] == {"type": "Metric"}


@pytest.mark.parametrize("text, message", [
    ("---\ntype: Metric\n", "no line closes it"),
    ("---\ntype: [unclosed\n---\n", "not valid YAML"),
    ("---\n- a list\n---\n", "must be a YAML mapping, got a list"),
    ("---\ntype: A\ntype: B\n---\n", "duplicate key 'type'"),
])
def test_bad_frontmatter_is_a_clear_error(text, message):
    with pytest.raises(ValueError, match=message):
        parse_frontmatter(text)


def test_an_empty_block_is_an_empty_mapping():
    assert parse_frontmatter("---\n---\nbody") == ({}, "body")


def test_yaml_1_2_booleans_only_true_and_false():
    data, _ = parse_frontmatter(
        "---\nstatus: on\nlabel: yes\nflag: no\nok: true\nbad: FALSE\n---\n")

    assert data == {"status": "on", "label": "yes", "flag": "no", "ok": True, "bad": False}
    import yaml
    assert yaml.safe_load("x: yes") == {"x": True}         # PyYAML's own loader is untouched


def test_native_timestamps_are_kept_as_dates():
    data, _ = parse_frontmatter("---\nstale_after: 2026-09-23\nquoted: \"2026-09-23\"\n---\n")

    assert data["stale_after"] == date(2026, 9, 23) and data["quoted"] == "2026-09-23"


def test_split_keeps_the_body_after_the_closing_line():
    assert split_frontmatter("---\na: 1\n---\nline one\n---\nline two") == (
        "a: 1\n", "line one\n---\nline two")


def test_without_pyyaml_the_error_says_what_to_install(monkeypatch):
    monkeypatch.setattr(frontmatter_module, "_LOADER", None)
    monkeypatch.setitem(sys.modules, "yaml", None)     # "import yaml" now fails

    with pytest.raises(ImportError, match=r'pip install "rag-titan\[okf\]"'):
        parse_frontmatter(CONCEPT)


# --- bundles ----------------------------------------------------------------------------

@pytest.fixture
def bundle_dir(tmp_path):
    return write(tmp_path / "acme", {
        "index.md": '---\nokf_version: "0.3"\ntitle: Acme analytics\nlicense: CC-BY-4.0\n---\n'
                    "# Metrics\n\n* [Gross margin](metrics/gross-margin.md)\n",
        "log.md": "# Log\n\n## 2026-05-22\n* **Creation**: first concepts.\n",
        "metrics/gross-margin.md": CONCEPT,
        "metrics/index.md": "---\ntype: Metric\n---\nan index is never a concept\n",
        "tables/order_items.md": ("---\ntype: BigQuery Table\ntags: finance, sales\n"
                                  "x-acme-owner: ana\n---\n"),
        "playbooks/new-thing.md": "---\ntype: Totally New Type\ndescription: ''\n---\nbody\n",
        "README.md": "# About this bundle\n",
        "notes/draft.md": "---\ntitle: no type here\n---\n",
        "references/attesters/check.py": "print('never run')\n",
        "diagram.png": b"\x89PNG\r\n",
        ".git/config": "[core]\n",
        "metrics/.hidden.md": "---\ntype: Metric\n---\n",
    })


def test_concepts_are_the_md_files_with_a_type(bundle_dir):
    bundle = read_bundle(bundle_dir)

    assert [c["id"] for c in bundle["concepts"]] == [
        "metrics/gross-margin", "playbooks/new-thing", "tables/order_items"]
    assert bundle["skipped"] == ["README.md", "notes/draft.md"]
    assert sorted(bundle["reserved"]) == ["index.md", "log.md", "metrics/index.md"]
    assert bundle["assets"] == ["diagram.png", "references/attesters/check.py"]


def test_concept_fields(bundle_dir):
    concepts = {c["id"]: c for c in read_bundle(bundle_dir)["concepts"]}

    margin = concepts["metrics/gross-margin"]
    assert (margin["type"], margin["title"], margin["tags"]) == ("Metric", "Gross margin",
                                                                 ["finance"])
    assert margin["body"].startswith("# Definition")
    table = concepts["tables/order_items"]
    assert table["title"] == "Order items"                    # made from the file name
    assert table["tags"] == ["finance, sales"]                # a scalar is never split at commas
    assert table["frontmatter"]["x-acme-owner"] == "ana"      # unknown keys are kept
    assert concepts["playbooks/new-thing"]["description"] is None   # '' is the same as absent


def test_bundle_metadata_and_warnings(bundle_dir):
    bundle = read_bundle(bundle_dir)

    assert bundle["metadata"] == {"okf_version": "0.3", "title": "Acme analytics",
                                  "license": "CC-BY-4.0"}
    assert ("README.md", "not a concept: no frontmatter") in messages(bundle, "warning")
    assert ("notes/draft.md", "not a concept: no `type` in its frontmatter") in messages(bundle)
    assert not messages(bundle, "error")


def test_an_unquoted_version_is_kept_as_text_with_a_warning(tmp_path):
    bundle = read_bundle(write(tmp_path, {"index.md": "---\nokf_version: 0.3\n---\n"}))

    assert bundle["metadata"]["okf_version"] == "0.3"
    assert "okf_version should be a quoted string" in messages(bundle, "warning")[0][1]


def test_broken_files_are_errors_and_reading_goes_on(tmp_path):
    bundle = read_bundle(write(tmp_path, {
        "good.md": CONCEPT, "dup.md": "---\ntype: A\ntype: B\n---\n",
        "latin1.md": "---\ntype: Café\n---\n".encode("latin-1")}))

    assert [c["id"] for c in bundle["concepts"]] == ["good"]
    assert {path for path, _ in messages(bundle, "error")} == {"dup.md", "latin1.md"}


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need extra rights on Windows")
def test_a_symlink_out_of_the_bundle_is_never_read(tmp_path):
    secret = write(tmp_path / "outside", {"secret.md": "---\ntype: Secret\n---\nsecret\n"})
    root = write(tmp_path / "bundle", {"a.md": CONCEPT})
    (root / "leak.md").symlink_to(secret / "secret.md")
    (root / "alias.md").symlink_to(root / "a.md")             # inside the bundle: fine

    bundle = read_bundle(root)

    assert [c["id"] for c in bundle["concepts"]] == ["a", "alias"]
    assert ("leak.md", "a symbolic link out of the bundle: not read") in messages(bundle)


def test_a_zip_reads_like_the_folder_and_unsafe_entries_are_skipped(bundle_dir, tmp_path):
    archive = tmp_path / "acme.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in bundle_dir.rglob("*"):
            if path.is_file():
                zipped.write(path, "acme/" + path.relative_to(bundle_dir).as_posix())
        zipped.writestr("acme/../evil.md", "---\ntype: Evil\n---\n")

    from_zip, from_folder = read_bundle(archive), read_bundle(bundle_dir)

    assert [c["id"] for c in from_zip["concepts"]] == [c["id"] for c in from_folder["concepts"]]
    assert from_zip["metadata"] == from_folder["metadata"]
    assert ("acme/../evil.md", "a zip entry outside the bundle: not read") in messages(from_zip)


def test_a_source_that_is_not_a_bundle_raises(tmp_path):
    (tmp_path / "file.txt").write_text("hi")
    with pytest.raises(ValueError, match="is a folder or a .zip file"):
        read_bundle(tmp_path / "file.txt")
