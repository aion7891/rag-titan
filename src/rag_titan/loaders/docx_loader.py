"""Loader for Word files, using the python-docx library."""

from docx import Document

from rag_titan.loaders.base_loader import BaseLoader


class DocxLoader(BaseLoader):
    """Reads the text of every paragraph in a .docx file."""

    extensions = [".docx"]

    def extract_text(self, path):
        """Return all non-empty paragraphs joined by spaces."""
        document = Document(path)
        paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
        return " ".join(paragraphs)
