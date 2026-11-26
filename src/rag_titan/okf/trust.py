"""
Trust, freshness and lifecycle of an OKF concept (OKF v0.3, section 5).

    trust_tier(frontmatter)       # "unverified", "machine-confirmed" or "human-reviewed"
    staleness(frontmatter, now)   # {"state": "fresh" | "stale" | "malformed" | None, ...}
    lifecycle(frontmatter)        # {"status": "stable", "current": True, ...}

A trust tier counts only *current* verification events: an event at or after
`generated.at` (5.3). When an agent rewrites a concept after a human review,
the human did not read the new text, so the tier goes down. Only an actor
written `human:<id>` gives "human-reviewed" (7). Trust tiers are advisory
signals, not access control.
"""

import re
from datetime import date, datetime, timezone

from rag_titan.okf.bundle import is_absent

TIERS = ["unverified", "machine-confirmed", "human-reviewed"]   # lowest first
STATUSES = ("draft", "stable", "deprecated")

# RFC 3339: a date-time with an offset, or a date only (5)
DATE_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$")
FULL_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_timestamp(value):
    """
    An OKF timestamp as a UTC datetime, or None when it is absent or not valid.

    Accepts a quoted RFC 3339 string or a YAML-native date/datetime. A date
    means the start of that day in UTC. A date-time without an offset is not
    valid (5).
    """
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else None
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        if FULL_DATE.match(text):
            return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if DATE_TIME.match(text):
            text = re.sub(r"[Zz]$", "+00:00", text)
            # Python 3.10 reads only 3 or 6 digits of a fraction: pad or cut to 6
            text = re.sub(r"\.(\d+)", lambda m: "." + (m.group(1) + "000000")[:6], text)
            return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None   # for example month 13
    return None


def verification_events(frontmatter):
    """
    The usable `verified` events: [{"by", "at" (UTC datetime)}].

    One bare mapping is a list of one event (5.2). An event without both `by`
    and a valid `at` is ignored.
    """
    raw = frontmatter.get("verified")
    if is_absent(raw):
        return []
    events = []
    for item in raw if isinstance(raw, list) else [raw]:
        if not isinstance(item, dict) or is_absent(item.get("by")):
            continue
        at = parse_timestamp(item.get("at"))
        if at is not None:
            events.append({"by": str(item["by"]).strip(), "at": at})
    return events


def generated_at(frontmatter):
    """`generated.at` as a UTC datetime, or None."""
    generated = frontmatter.get("generated")
    return parse_timestamp(generated.get("at")) if isinstance(generated, dict) else None


def trust_tier(frontmatter):
    """The trust tier from the current verification events only (5.3)."""
    changed = generated_at(frontmatter)
    current = [event for event in verification_events(frontmatter)
               if changed is None or event["at"] >= changed]
    if any(event["by"].startswith("human:") for event in current):
        return "human-reviewed"
    return "machine-confirmed" if current else "unverified"


def staleness(frontmatter, now=None):
    """
    Is the concept due for a check (5.5)?

    Returns:
        {"state": None (no review date), "fresh", "stale" or "malformed",
         "stale": bool}. A value that does not parse is never fresh.
    """
    raw = frontmatter.get("stale_after")
    if is_absent(raw):
        return {"state": None, "stale": False}
    when = parse_timestamp(raw)
    if when is None:
        return {"state": "malformed", "stale": True, "value": str(raw)}
    now = now or datetime.now(timezone.utc)
    stale = now >= when
    return {"state": "stale" if stale else "fresh", "stale": stale, "value": when.isoformat()}


def lifecycle(frontmatter):
    """
    The concept's status (5.4).

    Returns:
        {"status": the raw value ("stable" when absent), "current": bool,
         "known": bool}. Deprecated and unknown values are not current;
        a draft is current but not yet reviewed.
    """
    raw = frontmatter.get("status")
    status = "stable" if is_absent(raw) else str(raw).strip()
    return {"status": status, "known": status in STATUSES,
            "current": status in ("stable", "draft")}


def tier_at_least(tier, minimum):
    """True when `tier` is `minimum` or higher (for OKF_MIN_TRUST)."""
    if minimum not in TIERS:
        raise ValueError(f"Unknown trust tier {minimum!r}: use one of {', '.join(TIERS)}")
    return TIERS.index(tier) >= TIERS.index(minimum)
