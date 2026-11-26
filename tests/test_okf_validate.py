"""Tests for validating OKF bundles (v0.3 section 11) and the `rag-titan okf` commands."""

import json
import zipfile
from datetime import datetime, timezone

import pytest

from rag_titan import cli
from rag_titan.okf.validate import validate_bundle

GOOD = {
    "index.md": '---\nokf_version: "0.3"\ntitle: Acme\nlicense: CC-BY-4.0\n---\n# Metrics\n',
    "log.md": "# Directory Update Log\n\n## 2026-05-22\n* **Update**: margin.\n",
    "metrics/index.md": "# Metrics\n\n* [Margin](margin.md) - gross margin\n",
    "metrics/margin.md": (
        "---\ntype: Metric\ngenerated: { by: agent/x, at: \"2026-06-20T14:00:00Z\" }\n"
        "verified: { by: \"human:ana\", at: \"2026-06-25T09:00:00Z\" }\n"
        "stale_after: \"2026-01-01\"\n---\nUses [revenue](/computations/revenue.md).\n"),
    "computations/revenue.md": (
        "---\ntype: Attested Computation\nruntime: bigquery\n"
        "parameters:\n  - { name: year, type: integer, required: true }\n"
        "attester: { resource: /references/attesters/sql-equality.py }\n---\n"),
    "references/attesters/sql-equality.py": "print('never run by rag-titan')\n",
    "README.md": "# About\n",
}
NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def write(root, files):
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return root


def messages(items):
    return [(item["path"], item["message"]) for item in items]


def test_a_good_bundle_is_conformant_with_only_soft_warnings(tmp_path):
    report = validate_bundle(write(tmp_path, GOOD), now=NOW)

    assert report["conformant"] and report["errors"] == []
    assert messages(report["warnings"]) == [("README.md", "not a concept: no frontmatter")]
    assert report["summary"] == {
        "concepts": 2, "types": {"Metric": 1, "Attested Computation": 1},
        "trust_tiers": {"human-reviewed": 1, "unverified": 1}, "statuses": {"stable": 2},
        "stale": 1, "malformed_stale_after": 0, "links": 1, "broken_links": 0,
        "skipped": 1, "assets": 1}


@pytest.mark.parametrize("files, path, message", [
    ({"metrics/index.md": "---\ntitle: x\n---\n"}, "metrics/index.md",
     "a subfolder index file has no frontmatter (only the root index.md may)"),
    ({"log.md": "---\ntitle: x\n---\n## 2026-05-22\n"}, "log.md",
     "a log file has no frontmatter (only the root index.md may)"),
    ({"log.md": "# Log\n\n## May 22nd\n"}, "log.md",
     "a log date heading must be YYYY-MM-DD: 'May 22nd'"),
    ({"log.md": "# Log\n\n## 2026-02-30\n"}, "log.md",
     "a log date heading must be YYYY-MM-DD: '2026-02-30'"),
    ({"metrics/bad.md": "---\n- a list\n---\n"}, "metrics/bad.md",
     "the frontmatter must be a YAML mapping, got a list"),
])
def test_errors_are_only_what_breaks_conformance(tmp_path, files, path, message):
    report = validate_bundle(write(tmp_path, {**GOOD, **files}), now=NOW)

    assert not report["conformant"]
    assert (path, message) in messages(report["errors"])


@pytest.mark.parametrize("extra, message", [
    ("status: retired\n", "status 'retired' is not draft, stable or deprecated"),
    ("superseded_by: /metrics/margin.md\n",
     "superseded_by has a meaning only with status: deprecated"),
    ("stale_after: soon\n", "stale_after is not an RFC 3339 date"),
    ("verified: { by: \"human:ana\" }\n", "a verified event needs both `by` and `at`"),
    ("sources: [{ title: no resource }]\n", "a sources entry has no `resource`"),
    ("generated: { at: \"2026-06-20\" }\n", "generated has no `by`"),
    ("generated: { by: x, at: \"2026-06-20T14:00:00\" }\n", "generated.at is not an RFC 3339"),
])
def test_frontmatter_problems_are_warnings(tmp_path, extra, message):
    files = {**GOOD, "metrics/other.md": f"---\ntype: Metric\n{extra}---\nbody\n"}
    report = validate_bundle(write(tmp_path, files), now=NOW)

    assert report["conformant"]                      # consumers never reject for these (11.2)
    assert any(path == "metrics/other.md" and message in text
               for path, text in messages(report["warnings"]))


def test_attested_computation_checks(tmp_path):
    files = {**GOOD, "computations/revenue.md": (
        "---\ntype: Attested Computation\nparameters:\n  - { name: year }\n---\n")}
    found = [text for path, text in messages(validate_bundle(write(tmp_path, files))["warnings"])
             if path == "computations/revenue.md"]

    assert any("needs `runtime`" in text for text in found)
    assert any("a parameter needs `name` and `type`" in text for text in found)


def test_broken_links_unknown_types_and_keys_never_break_conformance(tmp_path):
    files = {**GOOD, "metrics/new.md": ("---\ntype: Brand New Type\nx-acme-owner: ana\n---\n"
                                        "See [nothing yet](missing.md).\n")}
    report = validate_bundle(write(tmp_path, files), now=NOW)

    assert report["conformant"]
    assert ("metrics/new.md", "broken link: missing.md (accepted: maybe not written yet)") in \
        messages(report["warnings"])


def test_a_v02_bundle_reads_with_the_fallback_and_says_so(tmp_path):
    files = {"metrics/margin.md": "---\ntype: Metric\n---\nUses [orders](tables/orders.md).\n",
             "tables/orders.md": "---\ntype: Table\n---\n"}
    report = validate_bundle(write(tmp_path, files))

    assert report["conformant"] and report["summary"]["links"] == 1
    warnings = messages(report["warnings"])
    assert ("index.md", "no okf_version: read as v0.2 or earlier (add okf_version: \"0.3\" to "
            "the root index.md frontmatter)") in warnings
    assert any("resolves only from the bundle root (v0.2 style)" in text for _, text in warnings)


def test_an_unknown_version_is_read_best_effort(tmp_path):
    files = {**GOOD, "index.md": '---\nokf_version: "9.0"\n---\n'}
    report = validate_bundle(write(tmp_path, files))

    assert report["conformant"]
    assert ("index.md", "okf_version '9.0' is not known here: read as v0.3, best effort") in \
        messages(report["warnings"])


# --- the CLI ------------------------------------------------------------------------------

def run(capsys, *args):
    """Run `rag-titan okf ...` and return (exit code, output)."""
    try:
        cli.main(["okf", *args])
        code = 0
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else 2
        if not isinstance(stop.code, int):
            print(stop.code)
    return code, capsys.readouterr().out


def test_validate_exit_codes(tmp_path, capsys):
    good = write(tmp_path / "good", GOOD)
    bad = write(tmp_path / "bad", {**GOOD, "log.md": "# Log\n\n## yesterday\n"})

    code, out = run(capsys, "validate", str(good))
    assert code == 0 and "conformant with OKF v0.3: 0 errors, 1 warning" in out
    assert run(capsys, "validate", str(good), "--strict")[0] == 1       # warnings fail in CI mode
    code, out = run(capsys, "validate", str(bad))
    assert code == 1 and "ERROR   log.md: a log date heading must be YYYY-MM-DD" in out


def test_validate_and_info_as_json_and_text(tmp_path, capsys):
    root = write(tmp_path / "acme", GOOD)

    report = json.loads(run(capsys, "validate", str(root), "--json")[1])
    assert report["conformant"] and report["metadata"]["title"] == "Acme"
    summary = json.loads(run(capsys, "info", str(root), "--json")[1])
    assert summary["concepts"] == 2 and summary["conformant"]
    text = run(capsys, "info", str(root))[1]
    assert text.startswith("Acme (OKF 0.3, CC-BY-4.0), 2 concepts")
    assert "trust: human-reviewed 1, unverified 1" in text


def test_a_zip_and_a_wrong_path(tmp_path, capsys):
    root = write(tmp_path / "acme", GOOD)
    archive = tmp_path / "acme.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in root.rglob("*"):
            zipped.write(path, "acme/" + path.relative_to(root).as_posix())

    assert run(capsys, "validate", str(archive))[0] == 0
    code, out = run(capsys, "info", str(tmp_path / "nope"))
    assert code == 2 and "is a folder or a .zip file" in out
