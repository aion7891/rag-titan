"""Reports: save evaluation reports as JSON or Markdown and compare them over time."""

from rag_titan.evaluation.reports.compare import find_regressions, flatten, trends
from rag_titan.evaluation.reports.failures import failed_queries
from rag_titan.evaluation.reports.manager import ReportManager
from rag_titan.evaluation.reports.markdown import to_markdown

__all__ = [
    "ReportManager",
    "failed_queries",
    "find_regressions",
    "flatten",
    "to_markdown",
    "trends",
]
