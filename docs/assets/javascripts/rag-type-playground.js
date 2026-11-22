// One RAG type's own playground (docs/rag-playground/<type>.md): its diagram, its knobs.
//   <div id="rag-type-app" data-rag-type="parent_child"></div>
// Move a knob and the question runs again: search, reranking and parents run live in your
// browser; the model's replies are replayed when they were recorded for those settings.

(() => {
  const app = document.getElementById("rag-type-app");
  if (!app) return;
  const ragType = app.dataset.ragType;
  const { el, documentsBox, card } = RagCards;
  const status = el("p", "rp-status", "Starting…");
  app.replaceChildren(status);
  const python = RagPython.start(status);

  const { KNOBS, PAIRS } = RagKnobs;   // each type's knobs: rag-knobs.js

  function select(label, options, value) {
    const input = el("select", "rp-input");
    options.forEach(([name, text]) => input.append(Object.assign(el("option", "", text),
      { value: name, selected: String(name) === String(value) })));
    const box = el("label", "rt-control");
    box.append(el("span", "rt-name", label), input);
    return { box, input, value: () => input.value };
  }

  // a slider with its value, its default and the values whose answers were recorded
  function slider([key, label, min, max, step, help], value, recorded) {
    const input = Object.assign(el("input"), { type: "range", min, max, step, value });
    const shown = el("b", "rt-value");
    const box = el("label", "rt-control");
    const update = () => { shown.textContent = input.value; };
    input.addEventListener("input", update);
    update();
    box.append(el("span", "rt-name", label), input, shown,
      el("span", "rt-help", `${help}. Default ${value}`
        + (recorded ? `; answers recorded for ${recorded.join(" · ")}` : "")));
    return { key, box, input, update, value: () => Number(input.value),
             reset: () => { input.value = value; update(); } };
  }

  // an on/off knob, with the same parts as a slider
  function toggle([key, label, , help], value, recorded) {
    const input = el("select", "rp-input");
    [["on", "on"], ["off", "off"]].forEach(([name, text]) => input.append(
      Object.assign(el("option", "", text), { value: name, selected: (name === "on") === value })));
    const box = el("label", "rt-control");
    box.append(el("span", "rt-name", label), input,
      el("span", "rt-help", `${help}. Default ${value ? "on" : "off"}`
        + (recorded ? "; answers recorded for on · off" : "")));
    return { key, box, input, update: () => {}, value: () => input.value === "on",
             reset: () => { input.value = value ? "on" : "off"; } };
  }

  function build(info) {
    const flows = info.flows[ragType];
    if (!info.rag_types.includes(ragType)) {
      status.textContent = "This RAG type is not in the recorded runs yet: here is its diagram.";
      const holder = el("div");
      app.append(holder);
      RagFlow.render(holder, flows.dense, { label: ragType });
      return;
    }
    const defaults = info.defaults[ragType];
    const grid = (info.grid || {})[ragType] || {};
    const question = select("Question", info.questions.map((q) => [q.id, q.question]),
                            info.questions[0].id);
    const chunker = select(ragType === "parent_child" ? "Child chunker" : "Chunker",
      Object.entries(info.chunkings).map(([name, s]) =>
        [name, `${name} (${s.chunk_size} ${name === "word" ? "words" : "characters"})`]),
      "paragraph" in info.chunkings ? "paragraph" : Object.keys(info.chunkings)[0]);
    const parentSize = ragType === "parent_child" ? select("Parent size",
      info.parent_sizes.map((n) => [n, `${n} characters`]), info.parent_sizes[0]) : null;
    const retriever = select("Retriever", info.retrievers.map((r) => [r, r]), "dense");
    const reranker = select("Reranker", info.rerankers.map((r) => [r, r]), info.reranker);
    const sliders = [...KNOBS.all, ...(KNOBS[ragType] || [])].map((knob) =>
      (knob[2] === "on/off" ? toggle : slider)(knob, defaults[knob[0]], grid[knob[0]]));

    const tuning = () => {
      const values = { reranker: reranker.value() };
      sliders.forEach((s) => { values[s.key] = s.value(); });
      if (parentSize) values.parent_size = Number(parentSize.value());
      return values;
    };
    const setup = () => [chunker.value(), ragType, question.value(), retriever.value(), tuning()];
    const result = el("div", "rt-result"), code = el("pre", "rt-code"), report = el("div");
    const indicator = el("p", "rt-indicator");
    const knobDefaults = { ...defaults, reranker: info.reranker };
    // a knob changed: the old report no longer fits, so it goes
    const rerun = debounce(() => {
      report.replaceChildren();
      indicator.textContent = recordedText(tuning(), knobDefaults, grid);
      runOnce(info, setup(), result, code);
    }, 300);

    // top_n can never be more than top_k (nor the lower grade above the upper one):
    // moving one knob of a pair past the other moves the other along
    PAIRS.forEach(([big, small]) => {
      const a = sliders.find((s) => s.key === big), b = sliders.find((s) => s.key === small);
      if (!a || !b) return;
      const follow = (moved, other) => () => {
        if (b.value() > a.value()) { other.input.value = moved.value(); other.update(); }
      };
      a.input.addEventListener("input", follow(a, b));
      b.input.addEventListener("input", follow(b, a));
    });
    [question, chunker, parentSize, retriever, reranker, ...sliders].filter(Boolean)
      .forEach((c) => c.input.addEventListener(c.input.type === "range" ? "input" : "change", rerun));

    const reset = el("button", "md-button", "Reset to defaults");
    reset.onclick = () => {
      sliders.forEach((s) => s.reset());
      reranker.input.value = info.reranker;
      retriever.input.value = "dense";
      if (parentSize) parentSize.input.value = info.parent_sizes[0];
      rerun();
    };
    const evaluate = el("button", "md-button", "Evaluate all questions with these settings");
    evaluate.onclick = () => evaluateAll(info, setup(), report);

    const panel = el("div", "rt-panel");
    panel.append(...[question, chunker, parentSize, retriever, reranker].filter(Boolean).map((c) => c.box),
                 ...sliders.map((s) => s.box), indicator);
    app.append(documentsBox(info.documents), panel, reset, result, evaluate, report,
               el("p", "rp-label", "The same setup in Python"), code);
    status.textContent = `Ready: real ${info.model} replies and ${info.embed_model} vectors, `
      + "replayed by the real rag-titan code in your browser. Move any knob.";
    rerun();
  }

  async function runOnce(info, [chunking, type, questionId, retriever, tuning], result, code) {
    code.textContent = pythonCode(info, chunking, type, retriever, tuning);
    try {
      const run = await python("run", chunking, type, questionId, retriever, tuning);
      const view = card(info.flows[type][retriever], `${type} + ${chunking} + ${retriever}`, run);
      result.replaceChildren(view.box);
      view.play(300);
    } catch (error) {
      result.replaceChildren(el("p", "rp-status", `Could not run: ${error.message}`));
    }
  }

  async function evaluateAll(info, [chunking, type, , retriever, tuning], report) {
    report.replaceChildren(el("p", "rp-status", "Running every question…"));
    const summary = await python("evaluate_tuned", chunking, type, retriever, tuning);
    const pct = (x) => (x === null ? "–" : `${Math.round(x * 100)}%`);
    const table = el("table", "rp-report");
    const row = (cells, tag) => { const tr = el("tr");
      cells.forEach((c) => tr.append(el(tag, "", c))); table.append(tr); };
    row(["question", "evidence found", "answer", "files given to the model"], "th");
    summary.rows.forEach((r) => row([r.id, pct(r.recall),
      r.recorded ? (r.answered ? "answered" : "no answer") : `not recorded (${r.stopped_at})`,
      [...new Set(r.sources)].join(", ") || "–"], "td"));
    const answers = summary.recorded
      ? `answers recorded for ${summary.recorded}, answered ${pct(summary.answered)} of those`
      : "no answers recorded for these settings (evidence scored on what the model would read)";
    report.replaceChildren(el("p", "rp-stats", `${summary.questions} questions · evidence `
      + `found ${pct(summary.recall)} · ${answers}`), table);
  }

  // the Python that builds this exact setup, so a good setting can be copied into a project
  function pythonCode(info, chunking, type, retriever, tuning) {
    const s = info.chunkings[chunking];
    const settings = { chunker: s.chunker, chunk_size: s.chunk_size, chunk_overlap: s.chunk_overlap,
                       retriever, ...tuning };
    const body = Object.entries(settings).map(([k, v]) => `        "${k}": ${JSON.stringify(v)},`);
    return ["from rag_titan import RAGTitan", "", `rag = RAGTitan(rag_type="${type}", settings={`,
            ...body, "})", "rag.index([\"leave_policy.md\", \"remote_work.md\", \"expenses.md\"])",
            "print(rag.run(\"your question\")[\"log\"])"].join("\n");
  }

  // Answers were recorded with the defaults, and with one knob at a time on a listed value.
  function recordedText(values, defaults, grid) {
    const changed = Object.keys(values).filter((k) => String(values[k]) !== String(defaults[k]));
    const one = changed.length === 1 && (grid[changed[0]] || []).map(String)
      .includes(String(values[changed[0]]));
    if (!changed.length || one) {
      return "● Answers recorded for these settings: the real model reply replays.";
    }
    return `○ ${changed.join(" and ")} changed: search and reranking run live, the answer `
      + "was recorded only with one knob at a time on a listed value.";
  }

  function debounce(fn, ms) {
    let timer = null;
    return () => { clearTimeout(timer); timer = setTimeout(fn, ms); };
  }

  python("info").then(build).catch((error) => {
    status.textContent = `The playground could not start: ${error.message}`;
  });
})();
