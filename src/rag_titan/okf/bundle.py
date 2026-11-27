"""
Read an OKF bundle: a folder (or .zip) of Markdown files with YAML frontmatter
(Open Knowledge Format v0.3, sections 2, 3, 4 and 8).

    bundle = read_bundle("examples/okf_bundle")
    bundle["metadata"]          # {"okf_version": "0.3", "title": ..., "license": ...}
    bundle["concepts"][0]       # {"id": "metrics/gross-margin", "type": "Metric", ...}
    bundle["problems"]          # [{"level": "warning", "path": ..., "message": ...}]

What counts as what (3.2): a .md file that is not reserved and has a non-empty
`type` in its frontmatter is a concept; a .md file without one is skipped
(README.md, AGENTS.md, ...); any other file is an asset. `index.md` and
`log.md` are reserved and never concepts, whatever they contain.
"""

from pathlib import PurePosixPath

from rag_titan.okf.files import bundle_files, problem
from rag_titan.okf.frontmatter import parse_frontmatter

RESERVED = ("index.md", "log.md")
METADATA_KEYS = ("okf_version", "title", "description", "license")


def read_bundle(source):
    """
    Read every file of a bundle and sort it into concepts, skipped files,
    assets and reserved files.

    Args:
        source: a folder, or a .zip file of one.

    Returns:
        A dict with "metadata" (the root index.md frontmatter), "concepts",
        "skipped" (.md files that are not concepts), "assets", "reserved"
        ({path: text} of index.md and log.md files) and "problems".
    """
    files, problems = bundle_files(source)
    bundle = {"source": str(source), "metadata": {}, "concepts": [], "skipped": [],
              "assets": [], "reserved": {}, "problems": problems}

    for path, data in files:
        if not path.endswith(".md"):
            bundle["assets"].append(path)
            continue
        text = decode(data, path, problems)
        if text is None:
            continue
        if PurePosixPath(path).name in RESERVED:
            bundle["reserved"][path] = text
        else:
            add_document(bundle, path, text)

    bundle["metadata"] = read_metadata(bundle["reserved"].get("index.md"), problems)
    return bundle


def decode(data, path, problems):
    """The file's text as UTF-8, or None (with a problem) when it is not UTF-8."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        problems.append(problem("error", path, "not UTF-8 text: not read"))
        return None


def add_document(bundle, path, text):
    """Add a .md file as a concept, or note why it is not one."""
    try:
        frontmatter, body = parse_frontmatter(text)
    except ValueError as error:
        bundle["problems"].append(problem("error", path, str(error)))
        return
    concept_type = frontmatter.get("type") if frontmatter else None
    if is_absent(concept_type):
        bundle["skipped"].append(path)
        why = "no frontmatter" if frontmatter is None else "no `type` in its frontmatter"
        bundle["problems"].append(problem("warning", path, f"not a concept: {why}"))
        return
    bundle["concepts"].append(make_concept(path, frontmatter, body))


def make_concept(path, frontmatter, body):
    """One concept: its id (the path without .md), the main fields and the raw frontmatter."""
    concept_id = path[:-len(".md")]
    title = frontmatter.get("title")
    return {
        "id": concept_id,
        "path": path,
        "type": str(frontmatter["type"]).strip(),
        # a missing title may be made from the file name (4.1): "gross-margin" -> "Gross margin"
        "title": str(title).strip() if not is_absent(title) else title_from_name(concept_id),
        "description": text_or_none(frontmatter.get("description")),
        "tags": as_list(frontmatter.get("tags")),
        "frontmatter": frontmatter,
        "body": body,
    }


def read_metadata(index_text, problems):
    """The bundle metadata (8.1) from the frontmatter of the root index.md, if any."""
    if index_text is None:
        return {}
    try:
        frontmatter, _ = parse_frontmatter(index_text)
    except ValueError as error:
        problems.append(problem("error", "index.md", str(error)))
        return {}
    frontmatter = frontmatter or {}
    metadata = {key: frontmatter[key] for key in METADATA_KEYS
                if not is_absent(frontmatter.get(key))}
    if "okf_version" in metadata and not isinstance(metadata["okf_version"], str):
        # 0.3 written without quotes is read as the number 0.3: keep it as text
        problems.append(problem("warning", "index.md", "okf_version should be a quoted "
                                f"string, like \"0.3\"; got {metadata['okf_version']!r}"))
        metadata["okf_version"] = str(metadata["okf_version"])
    return metadata


def is_absent(value):
    """OKF 11.2: a null value, an empty string and an empty list are the same as no key."""
    return value is None or (isinstance(value, (str, list, dict)) and not
                             (value.strip() if isinstance(value, str) else value))


def as_list(value):
    """OKF 11.2: one item is a list with one item; a scalar is never split at commas."""
    if is_absent(value):
        return []
    items = value if isinstance(value, list) else [value]
    return [str(item) for item in items if not is_absent(item)]


def text_or_none(value):
    """A field as text, or None when it is absent."""
    return None if is_absent(value) else str(value).strip()


def title_from_name(concept_id):
    """A display title from a concept id: "metrics/gross-margin" -> "Gross margin"."""
    words = PurePosixPath(concept_id).name.replace("-", " ").replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else concept_id
