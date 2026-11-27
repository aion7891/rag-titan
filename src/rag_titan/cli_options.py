"""
Tuning options shared by `rag-titan ask`, `chat`, `index` and `evaluate`.
Each one wins over its setting in .env, for that command only:

    --top-k 8          TOP_K                  chunks searched
    --top-n 3          TOP_N                  chunks kept after reranking
    --max-attempts 3   ADAPTIVE_MAX_ATTEMPTS  adaptive: searches before giving up
    --parent-size 800  PARENT_SIZE            parent_child: largest section (indexing)
    --texttiling-window 4  TEXTTILING_WINDOW  texttiling: sentences compared (indexing)
    --okf-include all      OKF_INCLUDE        OKF bundles: current concepts only, or all
    --okf-min-trust ...    OKF_MIN_TRUST      OKF bundles: lowest trust tier to index
    --parent-budget    PARENT_BUDGET          parent_child: characters given to the model
    --no-hyde-with-question  HYDE_WITH_QUESTION   hyde: search the made-up answer only
    --fusion-queries 4 FUSION_QUERIES         rag_fusion: rewordings to search with
    --crag-upper 8     CRAG_UPPER             corrective: grade that makes evidence good
    --crag-lower 2     CRAG_LOWER             corrective: below it for all = bad evidence
    --no-crag-refine   CRAG_REFINE            corrective: give the model whole chunks
    --crag-max-retries CRAG_MAX_RETRIES       corrective: times to search again
    --iterative-max-steps ITERATIVE_MAX_STEPS iterative: think-then-search rounds
    --hybrid-dense-weight 2  HYBRID_DENSE_WEIGHT  hybrid: how much vector search counts
    --hybrid-bm25-weight 2   HYBRID_BM25_WEIGHT   hybrid: how much keyword search counts
    --question-weight 2      QUESTION_WEIGHT      hyde, rag_fusion: the question's search
    --rrf-keep-top           RRF_KEEP_TOP         keep every search's #1 when fusing
"""

import argparse

# option -> the setting it changes
TUNING = {"top_k": "top_k", "top_n": "top_n", "max_attempts": "adaptive_max_attempts",
          "parent_size": "parent_size", "parent_budget": "parent_budget",
          "texttiling_window": "texttiling_window", "okf_include": "okf_include",
          "okf_min_trust": "okf_min_trust",
          "hyde_with_question": "hyde_with_question", "fusion_queries": "fusion_queries",
          "crag_upper": "crag_upper", "crag_lower": "crag_lower", "crag_refine": "crag_refine",
          "crag_max_retries": "crag_max_retries", "iterative_max_steps": "iterative_max_steps",
          "hybrid_dense_weight": "hybrid_dense_weight", "hybrid_bm25_weight": "hybrid_bm25_weight",
          "question_weight": "question_weight", "rrf_keep_top": "rrf_keep_top"}


def positive_int(text):
    """argparse type: a whole number above 0, with a short message otherwise."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {text!r}")
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be above 0, got {value}")
    return value


def whole_number(text):
    """argparse type: a whole number of 0 or more (0 turns a loop off)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {text!r}")
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def positive_number(text):
    """argparse type: a number above 0 like 1.5, as an RRF weight."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number, got {text!r}")
    # value > 0 is False for nan too; inf would outweigh every other list
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError(f"must be a number above 0, got {text}")
    return value


def grade(text):
    """argparse type: a grade from 0 to 10, as Corrective RAG gives its chunks."""
    value = whole_number(text)
    if value > 10:
        raise argparse.ArgumentTypeError(f"must be from 0 to 10, got {value}")
    return value


def add_tuning_options(command, search=True):
    """
    Add the tuning options to one command.

    Args:
        command: the argparse parser of the command.
        search: False for `index`, which only builds the index and so only
            takes the indexing options (--parent-size, --texttiling-window).
    """
    command.add_argument("--parent-size", type=positive_int,
                         help="parent_child: largest parent section in characters, used when "
                              "indexing (default: PARENT_SIZE, 1500)")
    command.add_argument("--texttiling-window", type=positive_int,
                         help="texttiling chunker: sentences compared on each side of a gap, "
                              "used when indexing (default: TEXTTILING_WINDOW, 3)")
    command.add_argument("--okf-include", choices=["current", "all"],
                         help="OKF bundles: index current concepts only (leave out deprecated "
                              "ones) or all (default: OKF_INCLUDE, current)")
    command.add_argument("--okf-min-trust",
                         choices=["unverified", "machine-confirmed", "human-reviewed"],
                         help="OKF bundles: lowest trust tier to index "
                              "(default: OKF_MIN_TRUST, unverified)")
    if not search:
        return
    command.add_argument("--top-k", type=positive_int,
                         help="chunks to search (default: TOP_K, else the type's own: "
                              "simple 5, adaptive 4, parent_child 8, iterative 3, others 5)")
    command.add_argument("--top-n", type=positive_int,
                         help="chunks to keep after reranking (default: TOP_N, else the "
                              "type's own: parent_child 4, others 2)")
    command.add_argument("--max-attempts", type=positive_int,
                         help="adaptive: searches before giving up (default: "
                              "ADAPTIVE_MAX_ATTEMPTS, 2)")
    command.add_argument("--parent-budget", type=positive_int,
                         help="parent_child: most characters of parent text given to the model "
                              "(default: PARENT_BUDGET, 3000)")
    command.add_argument("--hyde-with-question", action=argparse.BooleanOptionalAction,
                         help="hyde: also search with the question and fuse the two "
                              "searches (default: HYDE_WITH_QUESTION, on)")
    command.add_argument("--fusion-queries", type=positive_int,
                         help="rag_fusion: rewordings of the question to search with, "
                              "1 to 10 (default: FUSION_QUERIES, 3)")
    add_correction_options(command)
    add_fusion_options(command)


def add_correction_options(command):
    """The options of the RAG types that correct themselves (corrective, iterative)."""
    command.add_argument("--crag-upper", type=grade,
                         help="corrective: a chunk graded this or more (0-10) makes the "
                              "evidence good (default: CRAG_UPPER, 7)")
    command.add_argument("--crag-lower", type=grade,
                         help="corrective: every chunk graded below this (0-10) makes the "
                              "evidence bad, so it is searched again (default: CRAG_LOWER, 3)")
    command.add_argument("--crag-refine", action=argparse.BooleanOptionalAction,
                         help="corrective: keep only the useful sentences of each chunk "
                              "(default: CRAG_REFINE, on)")
    command.add_argument("--crag-max-retries", type=whole_number,
                         help="corrective: times to search again with a rewritten question, "
                              "0 grades only (default: CRAG_MAX_RETRIES, 1)")
    command.add_argument("--iterative-max-steps", type=whole_number,
                         help="iterative: think-then-search rounds after the first search, "
                              "0 searches once (default: ITERATIVE_MAX_STEPS, 3)")


def add_fusion_options(command):
    """The options of fusing ranked lists (RRF): hybrid search, hyde and rag_fusion."""
    command.add_argument("--hybrid-dense-weight", type=positive_number,
                         help="hybrid: how much the vector search's ranking counts, above 0 "
                              "(default: HYBRID_DENSE_WEIGHT, 1.0)")
    command.add_argument("--hybrid-bm25-weight", type=positive_number,
                         help="hybrid: how much the keyword search's ranking counts, above 0 "
                              "(default: HYBRID_BM25_WEIGHT, 1.0)")
    command.add_argument("--question-weight", type=positive_number,
                         help="hyde, rag_fusion: how much the question's own search counts "
                              "against the made-up queries' (default: QUESTION_WEIGHT, 1.0)")
    command.add_argument("--rrf-keep-top", action=argparse.BooleanOptionalAction,
                         help="keep the #1 chunk of every search when the lists are fused "
                              "(default: RRF_KEEP_TOP, off)")


def tuning_settings(args):
    """The settings the tuning options given on the command line change."""
    return {setting: getattr(args, option) for option, setting in TUNING.items()
            if getattr(args, option, None) is not None}
