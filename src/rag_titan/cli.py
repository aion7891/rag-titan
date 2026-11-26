"""
Command line tool, installed as `rag-titan`.

Examples:
    rag-titan ask --files handbook.pdf --question "How many leave days?"
    rag-titan index --files handbook.pdf --vector-store sqlite      # index once ...
    rag-titan ask --question "How many leave days?" --vector-store sqlite   # ... ask later
    rag-titan ask --files handbook.pdf --question "How many leave days?" --trace
    rag-titan chat --files handbook.pdf policy.docx --rag-type adaptive
    rag-titan ask --files handbook.pdf --question "..." --rag-type parent_child \\
        --top-k 10 --top-n 4 --parent-budget 2000 --reranker llm
    rag-titan evaluate --files *.md --questions questions.json \\
        --rag-type simple adaptive --chunker word paragraph --retriever dense hybrid
    rag-titan evaluate --input data.json --output report.json --baseline last_report.json
    rag-titan golden --questions questions.txt --files policy.md --output golden.json
    rag-titan okf validate path/to/bundle
    rag-titan --version
"""

import argparse

from rag_titan import __version__
from rag_titan.cli_evaluate import add_evaluate_command, build_setups, run_evaluate  # noqa: F401
from rag_titan.cli_golden import add_golden_command, run_golden
from rag_titan.cli_index import add_index_command, add_store_options, prepare_rag, run_index
from rag_titan.cli_okf import add_okf_command, run_okf
from rag_titan.cli_options import add_tuning_options
from rag_titan.factory import RAG_TYPES, RERANKERS, RETRIEVERS
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

    for command in (ask, chat):
        # optional: without --files the saved index is used
        command.add_argument("--files", nargs="+", help="PDF, DOCX, MD or TXT files")
        command.add_argument("--log-level", default="WARNING", help="DEBUG, INFO or WARNING")
        command.add_argument("--rag-type", default="simple", choices=RAG_TYPES)
        command.add_argument("--retriever", choices=RETRIEVERS,
                             help="dense, bm25 or hybrid (default: RETRIEVER)")
        command.add_argument("--reranker", choices=RERANKERS,
                             help="keyword, none, mmr, llm or cross_encoder (default: RERANKER)")
        add_store_options(command)
        add_tuning_options(command)

    ask.add_argument("--question", required=True, help="the question to answer")
    ask.add_argument("--trace", action="store_true", help="also print every step taken")

    add_evaluate_command(commands)
    add_golden_command(commands)
    add_index_command(commands)
    add_okf_command(commands)
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

    runners = {"evaluate": run_evaluate, "golden": run_golden, "index": run_index,
               "okf": run_okf}
    if args.command in runners:
        runners[args.command](args)
        return

    rag = prepare_rag(args)

    if args.command == "chat":
        run_chat(rag)
    elif args.trace:
        result = rag.run(args.question)
        print(result["log"] + "\n\n" + result["answer"])
    else:
        print(rag.ask(args.question))


if __name__ == "__main__":
    main()
