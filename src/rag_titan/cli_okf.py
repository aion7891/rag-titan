"""
`rag-titan okf`: check and describe Open Knowledge Format bundles.

    rag-titan okf validate path/to/bundle        # errors and warnings; exit 1 on errors
    rag-titan okf validate bundle.zip --strict   # warnings fail too (for CI)
    rag-titan okf info path/to/bundle            # concepts by type, trust tier, status
    rag-titan okf info path/to/bundle --json     # the same as JSON

Needs PyYAML: pip install "rag-titan[okf]".
"""

import json

from rag_titan.okf.validate import validate_bundle


def add_okf_command(commands):
    """Add the `okf` command and its `validate` and `info` sub-commands."""
    okf = commands.add_parser("okf", help="check or describe an OKF knowledge bundle")
    actions = okf.add_subparsers(dest="okf_command", required=True)
    for name, text in [("validate", "check a bundle against OKF v0.3"),
                       ("info", "count concepts by type, trust tier and status")]:
        action = actions.add_parser(name, help=text)
        action.add_argument("bundle", help="a bundle folder or a .zip of one")
        action.add_argument("--json", action="store_true", help="print JSON instead of text")
        action.add_argument("--log-level", default="WARNING", help="DEBUG, INFO or WARNING")
    actions.choices["validate"].add_argument(
        "--strict", action="store_true", help="exit 1 on warnings too, not only on errors")


def run_okf(args):
    """Run `okf validate` or `okf info` and set the exit code."""
    try:
        report = validate_bundle(args.bundle)
    except (ValueError, ImportError) as error:
        raise SystemExit(f"rag-titan okf: {error}")

    if args.okf_command == "info":
        print(json.dumps(info(report), indent=2) if args.json else info_text(report))
        return
    print(json.dumps(report, indent=2, default=str) if args.json else validate_text(report))
    if report["errors"] or (args.strict and report["warnings"]):
        raise SystemExit(1)


def heading(report):
    """"Acme analytics (OKF 0.3, CC-BY-4.0), 12 concepts"."""
    metadata = report["metadata"]
    details = [f"OKF {metadata.get('okf_version', 'version not stated')}"]
    if metadata.get("license"):
        details.append(str(metadata["license"]))
    return (f"{metadata.get('title', 'Bundle')} ({', '.join(details)}), "
            f"{report['summary']['concepts']} concepts")


def validate_text(report):
    """Errors first, then warnings, then the verdict."""
    lines = [heading(report)]
    for item in report["errors"] + report["warnings"]:
        lines.append(f"{item['level'].upper():8}{item['path']}: {item['message']}")
    verdict = "conformant" if report["conformant"] else "NOT conformant"
    lines.append(f"{verdict} with OKF v0.3: {plural(len(report['errors']), 'error')}, "
                 f"{plural(len(report['warnings']), 'warning')}")
    return "\n".join(lines)


def plural(count, word):
    """"1 error", "2 errors"."""
    return f"{count} {word}{'s' * (count != 1)}"


def info(report):
    """The summary with the bundle metadata."""
    return {"metadata": report["metadata"], **report["summary"],
            "conformant": report["conformant"]}


def counts(table):
    """{"Metric": 5, "Table": 2} -> "Metric 5, Table 2"."""
    return ", ".join(f"{name} {count}" for name, count in table.items()) or "none"


def info_text(report):
    """A few readable lines about the bundle."""
    summary = report["summary"]
    return "\n".join([
        heading(report),
        f"types: {counts(summary['types'])}",
        f"trust: {counts(summary['trust_tiers'])}",
        f"status: {counts(summary['statuses'])}",
        f"stale: {summary['stale']} (stale_after not readable: {summary['malformed_stale_after']})",
        f"links between concepts: {summary['links']}, broken: {summary['broken_links']}",
        f"other files: {summary['skipped']} skipped .md (no type), {summary['assets']} assets",
        f"{'conformant' if report['conformant'] else 'NOT conformant'} with OKF v0.3 "
        f"({plural(len(report['errors']), 'error')}, "
        f"{plural(len(report['warnings']), 'warning')}: see rag-titan okf validate)",
    ])
