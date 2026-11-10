"""
Command line tool, installed as `rag-titan`.

Examples:
    rag-titan ask --files handbook.pdf --question "How many leave days?"
    rag-titan chat --files handbook.pdf policy.docx --rag-type adaptive
    rag-titan --version
"""

import argparse

from rag_titan import __version__
from rag_titan.app import RAGTitan
from rag_titan.factory import RAG_TYPES
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

    # both commands share the same options
    for command in (ask, chat):
        command.add_argument("--files", nargs="+", required=True, help="PDF or DOCX files")
        command.add_argument("--rag-type", default="simple", choices=RAG_TYPES)
        command.add_argument("--log-level", default="WARNING", help="DEBUG, INFO or WARNING")

    ask.add_argument("--question", required=True, help="the question to answer")
    return parser


def run_chat(rag):
    """Read questions from the keyboard until the user types 'exit'."""
    print("Type your question, or 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ("exit", "quit"):
            break

        print(f"\nRAG: {rag.ask(question)}")


def main(argv=None):
    """Entry point used by the `rag-titan` command."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    rag = RAGTitan(rag_type=args.rag_type)
    chunk_count = rag.index(args.files)
    print(f"Indexed {chunk_count} chunks from {len(args.files)} file(s).")

    if args.command == "ask":
        print(rag.ask(args.question))
    else:
        run_chat(rag)


if __name__ == "__main__":
    main()
