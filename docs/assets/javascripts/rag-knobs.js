// The knobs of each RAG type's own playground page (rag-type-playground.js draws them).
// The names are the setting names, so a knob here means the same as its .env setting.

const RagKnobs = (() => {
  // [setting, label, min, max, step, what it does], or [setting, label, "on/off", what it does]
  const KNOBS = {
    all: [["top_k", "top_k", 1, 12, 1, "chunks the search finds"],
          ["top_n", "top_n", 1, 12, 1, "chunks kept after reranking (at most top_k)"]],
    adaptive: [["adaptive_max_attempts", "max attempts", 1, 4, 1,
                "searches (with a rewritten question) before giving up"]],
    parent_child: [["parent_budget", "parent budget", 200, 4000, 100,
                    "most characters of parent sections given to the model"]],
    hyde: [["hyde_with_question", "search the question too", "on/off",
            "fuse the question's search with the made-up answer's"]],
    rag_fusion: [["fusion_queries", "queries", 1, 5, 1,
                  "rewordings of the question, each one searched next to it"]],
    corrective: [["crag_upper", "upper grade", 0, 10, 1,
                  "a chunk graded this or more (0-10) makes the evidence correct"],
                 ["crag_lower", "lower grade", 0, 10, 1,
                  "every chunk graded below this makes it incorrect: thrown away, searched again"],
                 ["crag_refine", "keep useful sentences", "on/off",
                  "the model reads only the sentences of each chunk that help"],
                 ["crag_max_retries", "max retries", 0, 3, 1,
                  "times to search again with a rewritten question (0: grade only)"]],
    iterative: [["iterative_max_steps", "max steps", 0, 4, 1,
                 "think-then-search rounds after the first search (0: search once)"]],
  };

  // [bigger, smaller]: the smaller knob can never be above the bigger one
  const PAIRS = [["top_k", "top_n"], ["crag_upper", "crag_lower"]];

  return { KNOBS, PAIRS };
})();
