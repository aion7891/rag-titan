"""
Command line tool, installed as `rag-titan`.

Examples:
    rag-titan ask --files handbook.pdf --question "How many leave days?"
    rag-titan ask --files handbook.pdf --question "How many leave days?" --trace
    rag-titan chat --files handbook.pdf policy.docx --rag-type adaptive
    rag-titan evaluate --files *.md --questions questions.json \\
        --rag-type simple adaptive --chunker word paragraph
    rag-titan --version
"""

import argparse
import itertools

from rag_titan import __version__
from rag_titan.app import RAGTitan
from rag_titan.evaluation import (
    compare_setups,
    evaluate_retriever,
    format_rag_report,
    format_report,
    load_questions,
)
from rag_titan.factory import CHUNKERS, RAG_TYPES
from rag_titan.utils.logger import setup_logging


def build_parser():
    """Describe every command and option the CLI accepts."""
    parser = argparse.ArgumentParser(
        prog="rag-titan",
        description="Ask questions about your documents with different RAG types.",
    )
    parser.add_argument("--version", action="version", version=f"rag-titan {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="answer one question and exit")
    chat = commands.add_parser("chat", help="ask many questions in a loop")
    evaluate = commands.add_parser("evaluate", help="score RAG types on test questions")

    for command in (ask, chat, evaluate):
        command.add_argument("--files", nargs="+", required=True, help="PDF, DOCX, MD or TXT files")
        command.add_argument("--log-level", default="WARNING", help="DEBUG, INFO or WARNING")
    for command in (ask, chat):
        command.add_argument("--rag-type", default="simple", choices=RAG_TYPES)

    ask.add_argument("--question", required=True, help="the question to answer")
    ask.add_argument("--trace", action="store_true", help="also print every step taken")

    evaluate.add_argument("--questions", required=True, help="questions JSON file")
    evaluate.add_argument("--rag-type", nargs="+", default=["simple"], choices=RAG_TYPES,
                          help="one or more RAG types to compare")
    evaluate.add_argument("--chunker", nargs="+", choices=CHUNKERS,
                          help="one or more chunkers to compare (default: from settings)")
    evaluate.add_argument("--chunk-size", type=int, help="chunk size for every setup")
    evaluate.add_argument("--chunk-overlap", type=int, help="chunk overlap for every setup")
    evaluate.add_argument("--retrieval-only", action="store_true",
                          help="score search only (fast, no LLM); --rag-type is ignored")
    evaluate.add_argument("--top-k", type=int, default=5, help="chunks per search (retrieval only)")
    return parser


def run_chat(rag):
    """Read questions from the keyboard until the user types 'exit'."""
    print("Type your question, or 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ("exit", "quit"):
            break

        print(f"\nRAG: {rag.ask(question)}")


def ingestion_settings(args, chunker):
    """The settings one setup changes: the chunker and the chunk size/overlap given."""
    settings = {"chunker": chunker} if chunker else {}
    if args.chunk_size is not None:
        settings["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        settings["chunk_overlap"] = args.chunk_overlap
    return settings


def build_setups(args):
    """One setup for every RAG type x chunker combination, e.g. "adaptive + paragraph"."""
    setups = []
    for chunker, rag_type in itertools.product(args.chunker or [None], args.rag_type):
        name = f"{rag_type} + {chunker}" if chunker else rag_type
        setups.append({"name": name, "rag_type": rag_type,
                       "settings": ingestion_settings(args, chunker)})
    return setups


def run_evaluate(args):
    """Print the report for `rag-titan evaluate`, then the questions that went wrong."""
    questions = load_questions(args.questions)

    if args.retrieval_only:
        results = []
        for chunker in args.chunker or [None]:
            rag = RAGTitan(settings=ingestion_settings(args, chunker))
            rag.index(args.files)
            results.append(evaluate_retriever(rag.parts["retriever"], questions, args.top_k,
                                              name=chunker or "dense"))
        print(format_report(results))
    else:
        results = compare_setups(args.files, questions, build_setups(args))
        print(format_rag_report(results))

    print_weak_rows(results)


def print_weak_rows(results):
    """List every question that missed evidence or was not answered."""
    weak = [(result["name"], row) for result in results for row in result["rows"]
            if row["recall"] < 1 or not row.get("answered", True)]
    if not weak:
        print("\nEvery question found all its evidence.")
        return

    print("\nQuestions to look at:")
    for name, row in weak:
        answered = "" if row.get("answered", True) else "  not answered"
        print(f"  {name}: {row['id']}  recall {row['recall']:.2f}{answered}  "
              f"found {sorted(set(row['sources'])) or 'nothing'}")


def main(argv=None):
    """Entry point used by the `rag-titan` command."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.command == "evaluate":
        run_evaluate(args)
        return

    rag = RAGTitan(rag_type=args.rag_type)
    chunk_count = rag.index(args.files)
    print(f"Indexed {chunk_count} chunks from {len(args.files)} file(s).")

    if args.command == "chat":
        run_chat(rag)
    elif args.trace:
        result = rag.run(args.question)
        print(result["log"] + "\n\n" + result["answer"])
    else:
        print(rag.ask(args.question))


if __name__ == "__main__":
    main()
