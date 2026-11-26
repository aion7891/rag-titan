"""Tests for OKF trust, freshness, lifecycle (v0.3 section 5) and links (section 6)."""

from datetime import date, datetime, timezone

import pytest

from rag_titan.okf import read_bundle
from rag_titan.okf.links import body_links, bundle_graph, path_fields, replacement, resolve
from rag_titan.okf.trust import (
    lifecycle,
    parse_timestamp,
    staleness,
    tier_at_least,
    trust_tier,
)

GENERATED = {"by": "reference_agent/gemini-2.5-pro", "at": "2026-06-20T14:00:00Z"}
UTC = timezone.utc


# --- timestamps ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("2026-06-30T14:00:00Z", datetime(2026, 6, 30, 14, tzinfo=UTC)),
    ("2026-06-30T16:00:00+02:00", datetime(2026, 6, 30, 14, tzinfo=UTC)),
    ("2026-06-30", datetime(2026, 6, 30, tzinfo=UTC)),              # start of that day in UTC
    (date(2026, 6, 30), datetime(2026, 6, 30, tzinfo=UTC)),         # YAML-native date
    (datetime(2026, 6, 30, 14, tzinfo=UTC), datetime(2026, 6, 30, 14, tzinfo=UTC)),
    ("2026-06-30T14:00:00.5Z", datetime(2026, 6, 30, 14, 0, 0, 500000, tzinfo=UTC)),
    ("2026-06-30T14:00:00.123456789Z", datetime(2026, 6, 30, 14, 0, 0, 123456, tzinfo=UTC)),
])
def test_valid_timestamps(value, expected):
    assert parse_timestamp(value) == expected


@pytest.mark.parametrize("value", ["2026-06-30T14:00:00", datetime(2026, 6, 30, 14),
                                   "2026-13-01", "next week", 20260630, None, ""])
def test_invalid_timestamps(value):
    assert parse_timestamp(value) is None    # no offset, a naive datetime, month 13, ...


# --- trust tiers (5.3) --------------------------------------------------------------------

@pytest.mark.parametrize("verified, tier", [
    (None, "unverified"),
    ([{"by": "process:nightly", "at": "2026-06-26T02:00:00Z"}], "machine-confirmed"),
    ([{"by": "human:ana", "at": "2026-06-25T09:00:00Z"}], "human-reviewed"),
    ({"by": "human:ana", "at": "2026-06-25T09:00:00Z"}, "human-reviewed"),     # a bare mapping
    ([{"by": "team:finance", "at": "2026-06-25"}], "machine-confirmed"),       # not a person
    ([{"by": "robot:x", "at": "2026-06-25"}], "machine-confirmed"),            # unknown scheme
    ([{"by": "human:ana"}, {"at": "2026-06-25"}], "unverified"),               # missing keys
])
def test_trust_tiers(verified, tier):
    assert trust_tier({"generated": GENERATED, "verified": verified}) == tier


def test_a_review_before_the_last_change_does_not_count():
    frontmatter = {"generated": GENERATED, "verified": [
        {"by": "human:ana", "at": "2026-06-01T09:00:00Z"},           # before the rewrite
        {"by": "process:nightly", "at": "2026-06-21T02:00:00Z"}]}

    assert trust_tier(frontmatter) == "machine-confirmed"
    frontmatter["verified"][1]["at"] = "2026-06-10T02:00:00Z"
    assert trust_tier(frontmatter) == "unverified"


def test_without_generated_every_review_is_current():
    assert trust_tier({"verified": {"by": "human:ana", "at": "2020-01-01"}}) == "human-reviewed"


def test_minimum_trust():
    assert tier_at_least("human-reviewed", "machine-confirmed")
    assert not tier_at_least("unverified", "machine-confirmed")
    with pytest.raises(ValueError, match="Unknown trust tier 'trusted'"):
        tier_at_least("unverified", "trusted")


# --- freshness and lifecycle (5.4, 5.5) ---------------------------------------------------

NOW = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.mark.parametrize("value, state, stale", [
    (None, None, False), ("", None, False), ("2026-12-31", "fresh", False),
    ("2026-09-01T00:00:00Z", "stale", True),        # stale ON that instant
    ("2026-01-01", "stale", True), ("soon", "malformed", True),   # unparseable is never fresh
])
def test_staleness(value, state, stale):
    result = staleness({"stale_after": value}, now=NOW)
    assert (result["state"], result["stale"]) == (state, stale)


@pytest.mark.parametrize("status, current, known", [
    (None, True, True), ("stable", True, True), ("draft", True, True),
    ("deprecated", False, True), ("retired", False, False),        # unknown is not stable
])
def test_lifecycle(status, current, known):
    result = lifecycle({"status": status})
    assert (result["current"], result["known"]) == (current, known)
    assert result["status"] == (status or "stable")


# --- links (6.1) --------------------------------------------------------------------------

def test_only_real_links_are_links():
    body = ("See [margin](./margin.md \"title\") and [orders][ref].\n"
            "![chart](chart.png) and [[wikilink]] and `[code](x.md)`.\n\n"
            "```\n[in code](y.md)\n```\n\n"
            "[ref]: /tables/orders.md\n[^src]: a footnote, not a link\n")

    assert body_links(body) == ["./margin.md", "/tables/orders.md"]


FILES = {"metrics/margin.md", "metrics/revenue.md", "tables/orders.md", "my concept.md",
         "img/chart.png", "index.md"}


@pytest.mark.parametrize("value, kind, path", [
    ("https://example.com/x", "external", None),
    ("/tables/orders.md", "file", "tables/orders.md"),
    ("revenue.md", "file", "metrics/revenue.md"),
    ("../tables/orders.md#columns", "file", "tables/orders.md"),
    ("../my%20concept.md", "file", "my concept.md"),
    ("../../secrets.md", "external", None),                  # out of the root: never read
    ("/../../etc/passwd", "external", None),
    ("../img/", "folder", "img"),
    ("Revenue.md", "missing", "metrics/Revenue.md"),         # paths are case-sensitive
    ("#formula", "file", "metrics/margin.md"),
])
def test_resolve(value, kind, path):
    found = resolve(value, "metrics/margin.md", FILES, version="0.3")
    assert found["kind"] == kind and found.get("path") == path


def test_v02_bundles_fall_back_to_the_root():
    for version in (None, "0.2", "0.1"):
        found = resolve("tables/orders.md", "metrics/margin.md", FILES, version=version)
        assert found["path"] == "tables/orders.md" and found["fallback"] == "v0.2 root"
    assert resolve("tables/orders.md", "metrics/margin.md", FILES, "0.3")["kind"] == "missing"


def test_path_fields():
    frontmatter = {"resource": "bq://t", "superseded_by": "/m.md", "computation": "",
                   "executor": {"resource": "/ref/run.md"}, "attester": {"resource": "a.py"},
                   "sources": {"resource": "all queries in project X"}}

    assert path_fields(frontmatter) == [
        ("resource", "bq://t"), ("superseded_by", "/m.md"), ("executor.resource", "/ref/run.md"),
        ("attester.resource", "a.py"), ("sources.resource", "all queries in project X")]


# --- the bundle graph and superseded_by (6.4, 5.4) ----------------------------------------

def concept(type_, extra="", body=""):
    return f"---\ntype: {type_}\n{extra}---\n{body}"


@pytest.fixture
def bundle(tmp_path):
    files = {
        "index.md": '---\nokf_version: "0.3"\n---\n',
        "metrics/old.md": concept("Metric", "status: deprecated\nsuperseded_by: older.md\n"),
        "metrics/older.md": concept("Metric",
                                    "status: deprecated\nsuperseded_by: /metrics/new.md\n"),
        "metrics/new.md": concept("Metric", "sources:\n  - resource: ../tables/orders.md\n"
                                            "  - resource: all queries in project X\n",
                                  "Uses [orders](../tables/orders.md) and [a draft](future.md)."),
        "tables/orders.md": concept("BigQuery Table"),
        "loop/a.md": concept("Metric", "status: deprecated\nsuperseded_by: b.md\n"),
        "loop/b.md": concept("Metric", "status: deprecated\nsuperseded_by: a.md\n"),
    }
    for path, text in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    return read_bundle(tmp_path)


def test_the_bundle_graph_has_links_and_path_fields(bundle):
    graph = bundle_graph(bundle)

    edges = {(e["from"], e["to"], e["via"]) for e in graph["edges"]}
    assert ("metrics/new", "tables/orders", "link") in edges
    assert ("metrics/new", "tables/orders", "sources.resource") in edges
    assert ("metrics/old", "metrics/older", "superseded_by") in edges
    # a broken link is accepted and reported; a scope description is not broken
    assert graph["broken"] == [{"from": "metrics/new", "target": "future.md", "via": "link"}]


def test_superseded_by_is_followed_to_the_current_concept(bundle):
    by_id = {c["id"]: c for c in bundle["concepts"]}

    assert replacement(bundle, by_id["metrics/old"]) == "metrics/new"     # two hops
    assert replacement(bundle, by_id["metrics/new"]) is None              # not deprecated
    assert replacement(bundle, by_id["loop/a"]) is None                   # a loop never hangs
