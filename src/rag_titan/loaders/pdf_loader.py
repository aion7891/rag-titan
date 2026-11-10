"""Loader for PDF files, using the pypdf library."""

from pypdf import PdfReader

from rag_titan.loaders.base_loader import BaseLoader


class PdfLoader(BaseLoader):
    """Reads the text of every page in a .pdf file."""

    extensions = [".pdf"]

    def extract_text(self, path):
        """Return the text of all pages joined by spaces."""
        reader = PdfReader(path)

        # some pages (for example scanned images) have no text, so use ""
        pages = [page.extract_text() or "" for page in reader.pages]
        return " ".join(pages)
