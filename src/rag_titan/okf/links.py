"""
Links and path fields between OKF concepts (OKF v0.3, section 6).

How a link target or a path field resolves:
    "https://..."      an absolute URL: external
    "/metrics/x.md"    from the bundle root
    "x.md", "../y.md"  from the folder of the file that contains it
A value that leaves the bundle root is external: it is never read from disk.
Bundles of v0.2 or with no version may fall back to the bundle root (6.2).

Only `[text](target)` and reference-style links are links (6.1): not images,
not `[[wikilinks]]`, not footnotes, and not anything inside code.
Broken links are accepted: they can be knowledge nobody has written yet.
"""

import posixpath
import re
from urllib.parse import unquote

from rag_titan.okf.bundle import is_absent
from rag_titan.okf.trust import lifecycle

SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
FENCED_CODE = re.compile(r"^ {0,3}(```|~~~).*?^ {0,3}\1[^\n]*$", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
INLINE_LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE = re.compile(r"^ {0,3}\[([^\]^\n][^\]\n]*)\]:\s*<?(\S+?)>?(?:\s+.*)?$", re.MULTILINE)


def body_links(body):
    """The link targets in a Markdown body, in order (code removed first)."""
    text = INLINE_CODE.sub("", FENCED_CODE.sub("", body or ""))
    targets = [match.group(1) for match in INLINE_LINK.finditer(text)]
    return targets + [match.group(2) for match in REFERENCE.finditer(text)]


def is_v02(version):
    """True for a bundle of OKF 0.2 or earlier, or with no version (6.2, 12.1)."""
    if version is None:
        return True
    try:
        major, minor = (int(part) for part in str(version).split(".")[:2])
    except ValueError:
        return False
    return (major, minor) <= (0, 2)


def resolve(value, from_path, files, version=None):
    """
    Where a link target or path field value points.

    Args:
        value: the target, for example "/metrics/x.md" or "../tables/t.md#id".
        from_path: the bundle path of the file that contains it.
        files: every file path in the bundle (concepts, assets, reserved, ...).
        version: the bundle's okf_version, for the v0.2 fallback.

    Returns:
        {"kind": "external" (URL or out of the bundle), "file", "folder" or
        "missing", "path": the bundle path when inside}.
    """
    target = str(value).strip()
    if SCHEME.match(target):
        return {"kind": "external", "url": target}
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target:
        return {"kind": "file", "path": from_path}       # "#section": the same file
    start = "" if target.startswith("/") else posixpath.dirname(from_path)
    found = locate(posixpath.join(start, target.lstrip("/")), files)
    if found["kind"] == "missing" and not target.startswith("/") and is_v02(version):
        fallback = locate(target, files)                  # v0.2: try the bundle root
        if fallback["kind"] in ("file", "folder"):
            return {**fallback, "fallback": "v0.2 root"}
    return found


def locate(joined, files):
    """Normalize a bundle path and look it up among the files and their folders."""
    path = posixpath.normpath(joined)
    if path == ".." or path.startswith("../"):
        return {"kind": "external", "outside": True}     # never read from disk
    path = "" if path == "." else path
    if path in files:
        return {"kind": "file", "path": path}
    if path == "" or any(name.startswith(path.rstrip("/") + "/") for name in files):
        return {"kind": "folder", "path": path}
    return {"kind": "missing", "path": path}


def path_fields(frontmatter):
    """(field name, value) for every path field of a concept (6.2)."""
    fields = [(key, frontmatter.get(key)) for key in ("resource", "computation",
                                                      "superseded_by")]
    for key in ("executor", "attester"):
        if isinstance(frontmatter.get(key), dict):
            fields.append((f"{key}.resource", frontmatter[key].get("resource")))
    sources = frontmatter.get("sources")
    for source in sources if isinstance(sources, list) else [sources]:
        if isinstance(source, dict):
            fields.append(("sources.resource", source.get("resource")))
    return [(key, value) for key, value in fields if not is_absent(value)]


def bundle_graph(bundle):
    """
    The bundle graph (6.4): directed edges between concepts, and broken links.

    Returns:
        {"edges": [{"from": id, "to": id, "via": "link" or a field name}],
         "broken": [{"from": id, "target": value, "via": ...}]}.
        A sources.resource that is a description ("all queries in project X")
        is not a path, so it is never broken.
    """
    files = all_files(bundle)
    concepts = {concept["path"]: concept["id"] for concept in bundle["concepts"]}
    version = bundle["metadata"].get("okf_version")
    edges, broken = [], []
    for concept in bundle["concepts"]:
        targets = [("link", target) for target in body_links(concept["body"])]
        for via, target in targets + path_fields(concept["frontmatter"]):
            found = resolve(target, concept["path"], files, version)
            if found["kind"] == "file" and found["path"] in concepts:
                edges.append({"from": concept["id"], "to": concepts[found["path"]], "via": via})
            elif found["kind"] == "missing" and not is_scope(via, target):
                broken.append({"from": concept["id"], "target": str(target), "via": via})
    return {"edges": edges, "broken": broken}


def is_scope(via, target):
    """A sources.resource with spaces is a scope description, not a path (5.1)."""
    return via == "sources.resource" and " " in str(target).strip()


def all_files(bundle):
    """Every file path of a bundle that was read: concepts, skipped, reserved and assets."""
    return ({concept["path"] for concept in bundle["concepts"]} | set(bundle["skipped"])
            | set(bundle["reserved"]) | set(bundle["assets"]))


def replacement(bundle, concept):
    """
    The id of the concept that replaces a deprecated one, following
    `superseded_by` (5.4) until a concept that is not deprecated; None when
    there is none, the path is broken, or the chain loops.
    """
    by_path = {item["path"]: item for item in bundle["concepts"]}
    files, version = all_files(bundle), bundle["metadata"].get("okf_version")
    seen, current = {concept["path"]}, concept
    while lifecycle(current["frontmatter"])["status"] == "deprecated":
        target = current["frontmatter"].get("superseded_by")
        if is_absent(target):
            return None
        found = resolve(target, current["path"], files, version)
        following = by_path.get(found.get("path")) if found["kind"] == "file" else None
        if following is None or following["path"] in seen:
            return None
        seen.add(following["path"])
        current = following
    return current["id"] if current is not concept else None
