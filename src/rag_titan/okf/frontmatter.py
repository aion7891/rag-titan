"""
Read the YAML frontmatter of an OKF document (OKF v0.3, section 4).

    ---
    type: Metric
    title: Gross margin
    ---
    The body, in Markdown.

The rules: the file starts with a line that is exactly `---`, the block ends at
the next line that is exactly `---`, and the block parses as a YAML mapping.

OKF uses the YAML 1.2 core schema; PyYAML follows YAML 1.1, where `yes`, `no`,
`on` and `off` are booleans. The loader here keeps only `true` and `false` as
booleans (so `status: on` stays the text "on") and refuses duplicate keys.
PyYAML is the optional `okf` extra: pip install "rag-titan[okf]".
"""

import re

DELIMITER = "---"
YAML_12_BOOL = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
_LOADER = None   # built once, the first time a frontmatter is read


def yaml_loader():
    """A safe PyYAML loader with YAML 1.2 booleans that refuses duplicate keys."""
    global _LOADER
    if _LOADER is not None:
        return _LOADER
    try:
        import yaml
    except ImportError:
        raise ImportError(
            'Reading OKF bundles needs PyYAML: pip install "rag-titan[okf]"') from None

    class OKFLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            seen = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=True)
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        None, None, f"duplicate key {key!r}", key_node.start_mark)
                seen.add(key)
            return super().construct_mapping(node, deep)

    # copy the resolvers, so the change never touches PyYAML's own SafeLoader
    bool_tag = "tag:yaml.org,2002:bool"
    OKFLoader.yaml_implicit_resolvers = {
        first: [(tag, regexp) for tag, regexp in resolvers if tag != bool_tag]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()}
    OKFLoader.add_implicit_resolver(bool_tag, YAML_12_BOOL, list("tTfF"))
    _LOADER = OKFLoader
    return _LOADER


def split_frontmatter(text):
    """
    Cut a document into (frontmatter text, body).

    Returns:
        (None, text) when the file does not start with a `---` line;
        (block, body) when it does; raises ValueError when the block is never
        closed by a second `---` line.
    """
    text = text.lstrip("﻿")   # a UTF-8 byte order mark is not part of the text
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != DELIMITER:
        return None, text
    for number in range(1, len(lines)):
        if lines[number].rstrip("\r\n") == DELIMITER:
            return "".join(lines[1:number]), "".join(lines[number + 1:])
    raise ValueError("the frontmatter starts with --- but no line closes it with ---")


def parse_frontmatter(text):
    """
    Read a document's frontmatter.

    Returns:
        (data, body): data is a dict, or None when the document has no
        frontmatter. An empty block gives {}.

    Raises:
        ValueError: the block is not closed, is not valid YAML, has a duplicate
            key, or is not a mapping (for example a list).
    """
    block, body = split_frontmatter(text)
    if block is None:
        return None, body

    loader = yaml_loader()   # first: it explains how to install PyYAML when it is missing
    import yaml
    try:
        data = yaml.load(block, Loader=loader)
    except yaml.YAMLError as error:
        raise ValueError(f"the frontmatter is not valid YAML: {error}") from None
    if data is None:
        return {}, body
    if not isinstance(data, dict):
        raise ValueError(f"the frontmatter must be a YAML mapping, got a {type(data).__name__}")
    return data, body
