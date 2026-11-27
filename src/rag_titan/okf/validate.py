"""
Validate an OKF bundle against OKF v0.3 (section 11), and summarize it.

    report = validate_bundle("examples/okf_bundle")
    report["conformant"]       # True when there are no errors
    report["errors"]           # break conformance: bad frontmatter, reserved files (11.1)
    report["warnings"]         # worth fixing, never a reason to reject (11.2)
    report["summary"]          # concepts by type, trust tier, status; stale; broken links

Errors are only what 11.1 names: a frontmatter block that is not a YAML
mapping, and reserved files that break sections 8 and 9. Everything else
(files that are not concepts, broken links, unknown keys or types, missing
optional fields) is a warning at most, as consumers must not reject it.
"""

import re
from collections import Counter
from datetime import datetime
from pathlib import PurePosixPath

from rag_titan.okf.bundle import read_bundle
from rag_titan.okf.concept_checks import concept_warnings
from rag_titan.okf.files import problem
from rag_titan.okf.frontmatter import split_frontmatter
from rag_titan.okf.links import all_files, body_links, bundle_graph, path_fields, resolve
from rag_titan.okf.trust import TIERS, lifecycle, staleness, trust_tier

KNOWN_VERSIONS = ("0.1", "0.2", "0.3")
LOG_HEADING = re.compile(r"^##\s+(.*?)\s*$", re.MULTILINE)


def validate_bundle(source, now=None):
    """
    Check a bundle (a folder, a .zip, or a dict from read_bundle()).

    Returns:
        {"conformant", "errors", "warnings", "summary", "metadata"}.
    """
    bundle = source if isinstance(source, dict) else read_bundle(source)
    found = list(bundle["problems"]) + reserved_problems(bundle) + version_problems(bundle)
    for concept in bundle["concepts"]:
        found += concept_warnings(concept)
    graph = bundle_graph(bundle)
    found += [problem("warning", target_path(bundle, b["from"]),
                      f"broken {b['via']}: {b['target']} (accepted: maybe not written yet)")
              for b in graph["broken"]]
    found += fallback_warnings(bundle)
    errors = [item for item in found if item["level"] == "error"]
    return {"conformant": not errors, "errors": errors,
            "warnings": [item for item in found if item["level"] == "warning"],
            "summary": summarize(bundle, graph, now), "metadata": bundle["metadata"]}


def reserved_problems(bundle):
    """index.md in a subfolder and log.md have no frontmatter; log dates are YYYY-MM-DD (8, 9)."""
    found = []
    for path, text in sorted(bundle["reserved"].items()):
        name = PurePosixPath(path).name
        try:
            has_frontmatter = split_frontmatter(text)[0] is not None
        except ValueError:
            has_frontmatter = True      # the reader already reported the unclosed block
        if has_frontmatter and (name == "log.md" or path != "index.md"):
            kind = "log" if name == "log.md" else "subfolder index"
            found.append(problem("error", path, f"a {kind} file has no frontmatter "
                                 "(only the root index.md may)"))
        if name == "log.md":
            found += [problem("error", path, f"a log date heading must be YYYY-MM-DD: {h!r}")
                      for h in LOG_HEADING.findall(text) if not is_date(h)]
    return found


def is_date(text):
    """True for a real YYYY-MM-DD date."""
    try:
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text)) and bool(
            datetime.strptime(text, "%Y-%m-%d"))
    except ValueError:
        return False


def version_problems(bundle):
    """A missing or unknown okf_version (8.1, 12): read anyway, best effort."""
    version = bundle["metadata"].get("okf_version")
    if version is None:
        return [problem("warning", "index.md", "no okf_version: read as v0.2 or earlier "
                        "(add okf_version: \"0.3\" to the root index.md frontmatter)")]
    if version not in KNOWN_VERSIONS:
        return [problem("warning", "index.md", f"okf_version {version!r} is not known "
                        "here: read as v0.3, best effort")]
    return []


def fallback_warnings(bundle):
    """Relative paths that only resolved from the bundle root (the v0.2 fallback, 6.2)."""
    files, version = all_files(bundle), bundle["metadata"].get("okf_version")
    found = []
    for concept in bundle["concepts"]:
        values = body_links(concept["body"]) + [v for _, v in path_fields(concept["frontmatter"])]
        for value in values:
            if resolve(value, concept["path"], files, version).get("fallback"):
                found.append(problem("warning", concept["path"], f"{value} resolves only from "
                                     "the bundle root (v0.2 style): write /" + str(value)))
    return found


def target_path(bundle, concept_id):
    """The file path of a concept id."""
    return next(c["path"] for c in bundle["concepts"] if c["id"] == concept_id)


def summarize(bundle, graph, now=None):
    """Counts for `rag-titan okf info`."""
    concepts = bundle["concepts"]
    stale = [staleness(c["frontmatter"], now) for c in concepts]
    tiers = Counter(trust_tier(c["frontmatter"]) for c in concepts)
    return {
        "concepts": len(concepts),
        "types": dict(Counter(c["type"] for c in concepts).most_common()),
        # highest trust first, always in the same order
        "trust_tiers": {tier: tiers[tier] for tier in reversed(TIERS) if tiers[tier]},
        "statuses": dict(Counter(lifecycle(c["frontmatter"])["status"] for c in concepts)),
        "stale": sum(s["state"] == "stale" for s in stale),
        "malformed_stale_after": sum(s["state"] == "malformed" for s in stale),
        "links": len(graph["edges"]), "broken_links": len(graph["broken"]),
        "skipped": len(bundle["skipped"]), "assets": len(bundle["assets"]),
    }
