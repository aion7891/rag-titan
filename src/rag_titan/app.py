"""
RAGTitan: the easiest way to use the package.

It builds every part for you, so you only need two calls:

    from rag_titan import RAGTitan

    rag = RAGTitan(rag_type="simple")
    rag.index(["handbook.pdf", "policy.docx"])
    print(rag.ask("How many leave days do I get?"))
"""

from rag_titan.config import get_settings
from rag_titan.factory import build_rag, build_rag_parts


class RAGTitan:
    """
    One object that can index files and answer questions about them.

    Args:
        rag_type: which RAG to use, "simple" or "adaptive".
        settings: optional dict of settings to change, for example
            {"llm_model": "llama3.2"}. Anything missing uses the
            value from the environment or the default.
    """

    def __init__(self, rag_type="simple", settings=None):
        self.settings = get_settings(settings)
        self.rag_type = rag_type

        # build all the parts once, they share one embedder and one store
        self.parts = build_rag_parts(self.settings)
        self.pipeline = self.parts["pipeline"]
        self.rag = build_rag(rag_type, self.parts)

    def index(self, file_paths, reset=True):
        """
        Read, chunk, embed and store the given files.

        Args:
            file_paths: list of paths to .pdf or .docx files.
            reset: when True, remove previously indexed chunks first.

        Returns:
            The number of chunks stored.
        """
        return self.pipeline.run(file_paths, reset=reset)

    def ask(self, question):
        """
        Answer a question using the indexed files.

        Args:
            question: the user's question as text.

        Returns:
            The answer text.
        """
        return self.rag.answer(question)
