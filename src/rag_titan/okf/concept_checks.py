"""
Warnings about one OKF concept's frontmatter (OKF v0.3, sections 5 and 10).

None of these make a bundle non-conformant (11.2): consumers keep the value,
ignore an entry without its required key, and may warn. These checks are
those warnings, so a producer can fix them.
"""

from rag_titan.okf.bundle import is_absent
from rag_titan.okf.files import problem
from rag_titan.okf.trust import STATUSES, parse_timestamp


def concept_warnings(concept):
    """Every warning about one concept's frontmatter."""
    frontmatter, path = concept["frontmatter"], concept["path"]
    found = []
    found += timestamp_warnings(frontmatter)
    found += entry_warnings(frontmatter)
    found += lifecycle_warnings(frontmatter)
    found += computation_warnings(frontmatter)
    return [problem("warning", path, message) for message in found]


def as_items(value):
    """A list field: one mapping counts as a list of one (11.2)."""
    if is_absent(value):
        return []
    return value if isinstance(value, list) else [value]


def timestamp_warnings(frontmatter):
    """Timestamps that are not RFC 3339 (5): consumers cannot use them."""
    values = []
    if isinstance(frontmatter.get("generated"), dict):
        values.append(("generated.at", frontmatter["generated"].get("at")))
    values += [("verified.at", e.get("at")) for e in as_items(frontmatter.get("verified"))
               if isinstance(e, dict)]
    values += [("sources.last_modified", s.get("last_modified"))
               for s in as_items(frontmatter.get("sources")) if isinstance(s, dict)]
    window = frontmatter.get("usage_window")
    if isinstance(window, dict):
        values += [(f"usage_window.{key}", window.get(key)) for key in ("from", "to")]
    values.append(("stale_after", frontmatter.get("stale_after")))
    return [f"{key} is not an RFC 3339 date or date-time with an offset: {value!r}"
            for key, value in values if not is_absent(value) and parse_timestamp(value) is None]


def entry_warnings(frontmatter):
    """Entries without their REQUIRED key are ignored by consumers (11.2)."""
    found = []
    generated = frontmatter.get("generated")
    if not is_absent(generated) and (not isinstance(generated, dict)
                                     or is_absent(generated.get("by"))):
        found.append("generated has no `by`: who or what made the content is unknown")
    for event in as_items(frontmatter.get("verified")):
        if not isinstance(event, dict) or is_absent(event.get("by")) or is_absent(event.get("at")):
            found.append(f"a verified event needs both `by` and `at`, it is ignored: {event!r}")
    for source in as_items(frontmatter.get("sources")):
        if not isinstance(source, dict) or is_absent(source.get("resource")):
            found.append(f"a sources entry has no `resource`, it is ignored: {source!r}")
    return found


def lifecycle_warnings(frontmatter):
    """An unknown status, or superseded_by without status: deprecated (5.4)."""
    found = []
    status = frontmatter.get("status")
    if not is_absent(status) and str(status).strip() not in STATUSES:
        found.append(f"status {status!r} is not draft, stable or deprecated: "
                     "it is not treated as current")
    if not is_absent(frontmatter.get("superseded_by")) and str(status).strip() != "deprecated":
        found.append("superseded_by has a meaning only with status: deprecated")
    return found


def computation_warnings(frontmatter):
    """An Attested Computation needs `runtime`; parameters need name and type (10.2)."""
    if str(frontmatter.get("type", "")).strip() != "Attested Computation":
        return []
    found = []
    if is_absent(frontmatter.get("runtime")):
        found.append("an Attested Computation needs `runtime`; without it (and an attester) "
                     "no value may be shown as attested")
    for parameter in as_items(frontmatter.get("parameters")):
        if not isinstance(parameter, dict) or is_absent(parameter.get("name")) or \
                is_absent(parameter.get("type")):
            found.append(f"a parameter needs `name` and `type`, it is ignored: {parameter!r}")
    return found
