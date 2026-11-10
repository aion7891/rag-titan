"""
RAGTitan: the easiest way to use the package.

It builds every part for you, so you only need two calls:

    from rag_titan import RAGTitan

    rag = RAGTitan(rag_type="simple")
    rag.index(["handbook.pdf", "policy.docx"])
    print(rag.ask("How many leave days do I get?"))

    print(rag.run("How many leave days do I get?")["log"])   # every step
    rag.evaluate("questions.json")                           # end to end, uses the LLM
    rag.evaluate_retrieval("questions.json")                 # search only, fast
"""

from rag_titan.config import get_settings
from rag_titan.evaluation.benchmark import evaluate_retriever, format_report
from rag_titan.evaluation.end_to_end import evaluate_rag, format_rag_report
from rag_titan.evaluation.questions import as_questions
from rag_titan.factory import build_rag, build_rag_parts
from rag_titan.rag_types.trace import describe_steps


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

    def run(self, question):
        """
        Answer a question and show every step taken.

        Returns:
            The dict from the RAG type's run() ("answer", "sources", "steps"),
            plus "log": the steps as readable text (see describe_steps).
        """
        result = self.rag.run(question)
        result["log"] = describe_steps(result["steps"])
        return result

    def evaluate(self, questions, show=True):
        """
        Answer every test question with this RAG type and score the runs (uses the LLM).

        Args:
            questions: path to a questions JSON file, or a list of question dicts.
            show: print the report table.

        Returns:
            The dict from evaluate_rag(): evidence scores on the chunks given to
            the LLM, answered share, steps, time, and one row per question.
        """
        result = evaluate_rag(self.rag, questions, name=self.rag_type)
        if show:
            print(format_rag_report([result]))
        return result

    def evaluate_retrieval(self, questions, top_k=5, show=True):
        """
        Check how well search finds the evidence for test questions (no LLM used).

        Args:
            questions: path to a questions JSON file, or a list of question dicts.
            top_k: how many chunks to retrieve per question.
            show: print the report table.

        Returns:
            The dict from evaluate_retriever(): means, latency and one row per question.
        """
        result = evaluate_retriever(self.parts["retriever"], as_questions(questions), top_k)
        if show:
            print(format_report([result]))
        return result

