"""
Open Knowledge Format (OKF) bundles: folders of Markdown files with YAML
frontmatter that carry provenance, trust and lifecycle (OKF v0.3, v0.2 readable).
Needs PyYAML: pip install "rag-titan[okf]".
"""

from rag_titan.okf.bundle import read_bundle
from rag_titan.okf.frontmatter import parse_frontmatter

__all__ = ["parse_frontmatter", "read_bundle"]
