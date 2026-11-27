"""
Turn an OKF bundle into documents rag-titan can index.

    documents = okf_documents("examples/okf_bundle")              # current knowledge only
    documents = okf_documents("bundle.zip", include="all", min_trust="machine-confirmed")

Each concept becomes one document: its title as a heading, its description,
then its body. Every chunk of it carries the concept's facts in "metadata"
(concept id, type, tags, status, trust tier, staleness), so answers can say how
far to trust a passage. The values are plain text, lists and booleans, so a
saved index (SQLite, Chroma, Qdrant) keeps them.

By default only current knowledge is indexed (OKF 5.4): deprecated concepts
and unknown statuses are left out, so a question finds the concept that
replaced an old one. Drafts are kept and marked as drafts.
"""

from rag_titan.okf.bundle import read_bundle
from rag_titan.okf.links import replacement
from rag_titan.okf.trust import TIERS, lifecycle, staleness, tier_at_least, trust_tier
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

INCLUDE = ("current", "all")


def check_okf_options(include, min_trust):
    """Raise a clear ValueError for an unknown OKF_INCLUDE or OKF_MIN_TRUST."""
    if include not in INCLUDE:
        raise ValueError(f"OKF_INCLUDE must be current or all, got {include!r}")
    if min_trust not in TIERS:
        raise ValueError(f"OKF_MIN_TRUST must be one of {', '.join(TIERS)}, got {min_trust!r}")


def okf_documents(source, include="current", min_trust="unverified", now=None):
    """
    The concepts of a bundle as documents ({"source", "file_type", "text", "metadata"}).

    Args:
        source: a bundle folder or a .zip of one.
        include: "current" leaves out deprecated concepts and unknown statuses;
            "all" keeps every concept (its status is in the metadata).
        min_trust: leave out concepts below this trust tier ("unverified" keeps all).
        now: the time to judge staleness against (default: now).
    """
    check_okf_options(include, min_trust)
    bundle = read_bundle(source)
    documents, left_out = [], {"not current": 0, "below trust": 0}
    for concept in bundle["concepts"]:
        meta = concept_metadata(bundle, concept, now)
        if include == "current" and not lifecycle(concept["frontmatter"])["current"]:
            left_out["not current"] += 1
        elif not tier_at_least(meta["trust_tier"], min_trust):
            left_out["below trust"] += 1
        else:
            documents.append({"source": concept["path"], "file_type": ".md",
                              "text": concept_text(concept), "metadata": meta})
    logger.info("OKF bundle %s: %d concepts indexed, %d not current, %d below %s",
                source, len(documents), left_out["not current"], left_out["below trust"],
                min_trust)
    return documents


def concept_text(concept):
    """The text that is chunked and searched: title, description, body."""
    parts = [f"# {concept['title']}"]
    if concept["description"]:
        parts.append(concept["description"])
    if concept["body"].strip():
        parts.append(concept["body"].strip())
    return "\n\n".join(parts) + "\n"


def concept_metadata(bundle, concept, now=None):
    """The facts every chunk of the concept carries (JSON-safe values only)."""
    frontmatter = concept["frontmatter"]
    status = lifecycle(frontmatter)["status"]
    meta = {
        "okf_concept": concept["id"],
        "okf_type": concept["type"],
        "tags": concept["tags"],
        "status": status,
        "trust_tier": trust_tier(frontmatter),
        "stale": staleness(frontmatter, now)["stale"],
    }
    if bundle["metadata"].get("title"):
        meta["okf_bundle"] = str(bundle["metadata"]["title"])
    if status == "deprecated":
        meta["replaced_by"] = replacement(bundle, concept)
    return meta
