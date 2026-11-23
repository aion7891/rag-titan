// Readable details of RAG steps (the {"step", "input", "output", "ms"} dicts of run()).
// Like describe_steps() in Python: chunks are shown by id, file and score.
// Everything is set with textContent, so text from documents is never run as HTML.

const RagSteps = (() => {
  const LONG = 220;   // characters of a text to show

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function isChunks(value) {
    return Array.isArray(value) && value.every((item) => item && typeof item === "object" && "id" in item);
  }

  function short(text) {
    const flat = String(text).replace(/\s+/g, " ").trim();
    return flat.length > LONG ? `${flat.slice(0, LONG - 1)}…` : flat;
  }

  // How a fused score was made: "vector #2 · keyword #1 · rrf 0.0325" (hybrid search) or
  // "query 1 #3 · query 2 ×2 #1 · rrf 0.0325" (HyDE, RAG-Fusion: one list per search query;
  // ×2 is a list's weight when it is not 1), and "kept as #1 of query 2" (keep_top).
  const LIST_NAMES = { dense: "vector", bm25: "keyword" };
  const listName = (list) => LIST_NAMES[list] || list.replace("_", " ");
  function ranks(trace) {
    if (!trace || !("rrf" in trace)) return "";
    const parts = Object.keys(trace).filter((key) => key.endsWith("_rank")).map((key) => {
      const list = key.slice(0, -"_rank".length);
      const weight = trace[`${list}_weight`];
      return `${listName(list)}${weight === undefined ? "" : ` ×${weight}`} #${trace[key]}`;
    });
    parts.push(`rrf ${trace.rrf.toFixed(4)}`);
    if (trace.kept_as_top_of) parts.push(`kept as #1 of ${trace.kept_as_top_of.map(listName).join(", ")}`);
    return `  [${parts.join(" · ")}]`;
  }

  // Several result lists by name (the input of "fuse lists"): "query_1: 8 chunks, query_2: 3".
  function isLists(value) {
    return value && typeof value === "object" && !Array.isArray(value)
      && Object.values(value).length > 0 && Object.values(value).every(isChunks);
  }

  function chunkLine(chunk) {
    if (chunk.children) return parentLine(chunk);
    const score = "rerank_score" in chunk ? `rerank ${chunk.rerank_score.toFixed(3)}`
      : "score" in chunk ? chunk.score.toFixed(3) : "";
    // the LLM reranker's own 0-10 rating, and why a score was kept or set to 0
    const rating = chunk.llm_score === undefined || chunk.llm_score === null ? ""
      : `  [model rated ${chunk.llm_score}/10]`;
    // Corrective RAG: the grade the model gave, and how many sentences were kept
    const grade = chunk.grade === undefined ? "" : `  [graded ${chunk.grade}/10]`;
    const notes = [chunk.rerank_note, chunk.grade_note, chunk.refine_note].filter(Boolean)
      .map((text) => `\n   note: ${text}`).join("");
    return `${chunk.id} ${chunk.source || ""} ${score ? `(${score})` : ""}${ranks(chunk.score_trace)}`
      + `${rating}${grade}\n   “${short(chunk.text || "")}”${notes}`;
  }

  // A parent section of Parent-Child RAG: which children led to it, then its text.
  function parentLine(parent) {
    const count = parent.children.length;
    return `${parent.parent_id} · ${count} child${count === 1 ? "" : "ren"} `
      + `(${parent.children.join(", ")}) · ${parent.text.length} chars\n`
      + `   matched: “${short(parent.child_text || "")}”\n`
      + `   section given to the model: “${short(parent.text)}”`;
  }

  // The output of Corrective RAG's grade step, Iterative RAG's think step, or a loop's summary.
  function correction(value, brief) {
    if ("verdict" in value && "chunks" in value) {
      const head = `${value.verdict}${value.note ? ` (${value.note})` : ""}`;
      return brief ? head : [head, ...value.chunks.map(chunkLine)].join("\n");
    }
    if ("thought" in value && "search" in value) {
      const next = value.search ? `next search: “${short(value.search)}”`
        : `done${value.note ? ` (${value.note})` : ""}`;
      return value.thought ? `thought: “${short(value.thought)}”\n${next}` : next;
    }
    if ("stop_reason" in value) {
      return `${value.steps} of ${value.max_steps} rounds, stopped: ${value.stop_reason}`;
    }
    return null;
  }

  // One value: chunks one per line, yes/no for booleans, quoted text otherwise.
  function describe(value, brief) {
    const corrected = value && typeof value === "object" && !Array.isArray(value)
      && correction(value, brief);
    if (corrected) return corrected;
    if (value && typeof value === "object" && !Array.isArray(value) && "error" in value) {
      return value.error.startsWith("NotRecorded")
        ? "not recorded: no model reply was recorded for these settings"
        : `failed: ${short(value.error)}`;
    }
    if (isLists(value)) {
      return Object.entries(value).map(([name, chunks]) =>
        `${name.replace("_", " ")}: ${chunks.length} chunk${chunks.length === 1 ? "" : "s"}`).join(", ");
    }
    if (isChunks(value)) {
      const word = value.length && value[0].children ? "parent section" : "chunk";
      const count = `${value.length} ${word}${value.length === 1 ? "" : "s"}`;
      return brief ? count : [count, ...value.map(chunkLine)].join("\n");
    }
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (typeof value === "string") return `“${short(value)}”`;
    return short(JSON.stringify(value));
  }

  function io(label, text) {
    const line = el("p", "rf-io");
    line.append(el("b", "", `${label} `), document.createTextNode(text));
    return line;
  }

  // One step: "#6 retrieve · 0.01 ms", its inputs (chunk lists only counted) and its output.
  function step(visit) {
    const box = el("div", "rf-step");
    const head = el("div", "rf-step-head", `#${visit.number} ${visit.step.step} `);
    head.append(el("span", "", `· ${Number(visit.step.ms).toFixed(2)} ms`));
    const inputs = Object.entries(visit.step.input || {})
      .map(([key, value]) => `${key}=${describe(value, true)}`).join(", ");
    box.append(head, io("in", inputs || "–"), io("out", describe(visit.step.output, false)));
    // a step stopped for lack of a recording: show in full what the model would have read
    const output = visit.step.output;
    const stopped = output && typeof output === "object" && "error" in output;
    if (stopped && isChunks(visit.step.input.chunks)) {
      box.append(io("the model would read", describe(visit.step.input.chunks, false)));
    }
    return box;
  }

  // All the visits of one node, e.g. retrieve was steps #2 and #6 after a rewrite.
  function details(label, visits) {
    const box = el("div", "rf-details");
    if (!visits.length) {
      box.append(el("p", "rf-io", `${label}: not used in this run.`));
      return box;
    }
    visits.forEach((visit) => box.append(step(visit)));
    return box;
  }

  return { details, describe };
})();
