"""
Tests for the optional extras in pyproject.toml: `all` must install every
user-facing feature, so a new extra cannot be forgotten in it.
"""

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")   # Python 3.11+; the check is the same on 3.10

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"

# extras for working on rag-titan itself, not features
DEVELOPER_EXTRAS = {"dev", "docs", "all"}

pytestmark = pytest.mark.skipif(not PYPROJECT.is_file(), reason="needs the repo's pyproject.toml")


def extras():
    """The optional-dependencies table of pyproject.toml."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]


def test_all_installs_every_feature_extra():
    table = extras()
    assert table["all"] == ["rag-titan[tokens,chroma,qdrant,rerank,okf]"]

    named = set(re.fullmatch(r"rag-titan\[(.+)\]", table["all"][0]).group(1).split(","))
    assert named == set(table) - DEVELOPER_EXTRAS     # a new extra must be added to all


def test_the_install_docs_list_every_feature_extra():
    root = PYPROJECT.parent
    for page in (root / "README.md", root / "docs" / "getting-started.md"):
        text = page.read_text(encoding="utf-8")
        for name in set(extras()) - {"dev", "docs"}:
            assert f'pip install "rag-titan[{name}]"' in text, f"{page.name}: {name}"
