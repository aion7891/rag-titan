// RAG playground (docs/rag-playground.md): pick chunkings, RAG types and a question,
// then watch each setup run on its own live flow diagram. Python runs in rag-worker.js.

(() => {
  const app = document.getElementById("rag-playground-app");
  if (!app) return;

  const here = document.currentScript.src;   // .../assets/javascripts/rag-playground.js?v=...
  const assets = new URL("../", here).href;
  const worker = new Worker(new URL(`rag-worker.js${new URL(here).search}`, here), { type: "module" });
  const waiting = {};
  let calls = 0;

  const status = el("p", "rp-status", "Starting…");
  app.replaceChildren(status);

  worker.onmessage = ({ data }) => {
    if (data.status) { status.textContent = data.status; return; }
    const { resolve, reject } = waiting[data.id];
    delete waiting[data.id];
    if (data.error) reject(new Error(data.error)); else resolve(data.result);
  };
  // a worker that fails to load (no network, old browser) must not leave "Starting…" forever
  worker.onerror = (event) => {
    status.textContent = `The playground could not start: ${event.message || "the Python worker failed to load"}`;
  };
  const python = (call, ...args) => new Promise((resolve, reject) => {
    waiting[++calls] = { resolve, reject };
    worker.postMessage({ id: calls, call, args, assets });
  });

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function choices(title, names, describe, checked) {
    const box = el("fieldset", "rp-choices");
    box.append(el("legend", "", title));
    const inputs = names.map((name) => {
      const input = Object.assign(el("input"), { type: "checkbox", value: name,
                                                 checked: checked.includes(name) });
      const label = el("label", "rp-choice");
      label.append(input, ` ${name}`, el("span", "", describe(name)));
      box.append(label);
      return input;
    });
    return { box, picked: () => inputs.filter((i) => i.checked).map((i) => i.value) };
  }

  function documentsBox(documents) {
    const box = el("details", "rp-documents");
    box.append(el("summary", "", `The ${documents.length} documents every setup searches`));
    documents.forEach((doc) => box.append(el("h4", "", doc.source), el("pre", "", doc.text)));
    return box;
  }

  function card(info, chunking, ragType, run) {
    const box = el("div", "rp-card");
    const holder = el("div"), details = el("div", "rp-details");
    const flow = info.flows[ragType];
    const label = (id) => flow.nodes.find((node) => node.id === id).label;
    box.append(el("h3", "", `${ragType} + ${chunking}`), holder,
      el("p", run.answered ? "rp-answer" : "rp-answer rp-none", run.answer),
      el("p", "rp-stats", `found ${run.recall * 100}% of the evidence · ${run.steps.length} steps · `
        + `files given to the LLM: ${[...new Set(run.sources)].join(", ") || "none"}`), details);
    const diagram = RagFlow.render(holder, flow, {
      label: `${ragType} + ${chunking}`,
      onSelect: (id, visits) => details.replaceChildren(RagSteps.details(label(id), visits)),
    });
    return { box, play: () => diagram.play(run.steps, 650).then(() => diagram.select("retrieve")) };
  }

  function reportTable(report) {
    const table = el("table", "rp-report");
    const head = ["setup", "Recall", "MRR", "nDCG", "Answered", "Steps"];
    const row = (cells, tag) => {
      const tr = el("tr");
      cells.forEach((cell) => tr.append(el(tag, "", cell)));
      table.append(tr);
    };
    row(head, "th");
    report.results.forEach((r) => row([r.name, r.recall.toFixed(3), r.mrr.toFixed(3),
      r.ndcg.toFixed(3), `${Math.round(r.answered * 100)}%`, r.steps.toFixed(1)], "td"));
    const weak = report.results.flatMap((r) => r.rows.filter((q) => q.recall < 1 || !q.answered)
      .map((q) => `${r.name}: ${q.id} · recall ${q.recall.toFixed(2)}`
        + `${q.answered ? "" : " · not answered"} · found ${[...new Set(q.sources)].join(", ") || "nothing"}`));
    const list = el("ul", "rp-weak");
    (weak.length ? weak : ["Every question found all its evidence."])
      .forEach((line) => list.append(el("li", "", line)));
    return [table, el("p", "rp-stats", "Questions to look at:"), list];
  }

  async function build(info) {
    const question = el("select", "rp-question");
    info.questions.forEach((q) => question.append(Object.assign(el("option", "", q.question),
                                                                { value: q.id })));
    const chunkings = choices("Ingestion (chunker)", Object.keys(info.chunkings), (name) => {
      const s = info.chunkings[name];
      return ` ${s.chunk_size} ${name === "word" ? "words" : "characters"}, overlap ${s.chunk_overlap}`;
    }, ["word", "paragraph"]);
    const ragTypes = choices("RAG type", info.rag_types, () => "", info.rag_types);
    const runButton = el("button", "md-button md-button--primary", "Run");
    const evalButton = el("button", "md-button", "Evaluate all questions");
    const results = el("div", "rp-results");

    runButton.onclick = async () => {
      results.replaceChildren(el("p", "rp-status", "Running…"));
      const cards = [];
      for (const chunking of chunkings.picked()) {
        for (const ragType of ragTypes.picked()) {
          cards.push(card(info, chunking, ragType, await python("run", chunking, ragType, question.value)));
        }
      }
      results.replaceChildren(...cards.map((c) => c.box));
      cards.forEach((c) => c.play());
    };
    evalButton.onclick = async () => {
      results.replaceChildren(el("p", "rp-status", "Running every question on every setup…"));
      results.replaceChildren(...reportTable(await python("evaluate", chunkings.picked(), ragTypes.picked())));
    };

    status.textContent = `Ready: real ${info.model} replies and ${info.embed_model} vectors, `
      + "replayed by the real rag-titan code running in your browser.";
    const controls = el("div", "rp-controls");
    controls.append(chunkings.box, ragTypes.box);
    app.append(documentsBox(info.documents), el("label", "rp-label", "Question"), question,
               controls, runButton, evalButton, results);
    runButton.onclick();
  }

  python("info").then(build).catch((error) => {
    status.textContent = `The playground could not start: ${error.message}`;
  });
})();
