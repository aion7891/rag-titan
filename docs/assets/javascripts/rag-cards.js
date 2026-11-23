// Building blocks of the RAG playground page: controls, one card per run, the evaluate report.
// Text from documents and models is always set with textContent, never as HTML.

const RagCards = (() => {
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // a group of checkboxes; picked() returns the ticked values
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
    box.append(el("summary", "", `The ${documents.length} sample documents`));
    documents.forEach((doc) => box.append(el("h4", "", doc.source), el("pre", "", doc.text)));
    return box;
  }

  function stats(run) {
    const share = run.recall === null ? null : `${Math.round(run.recall * 100)}%`;
    let evidence = share === null ? "evidence not scored (your own question or text)"
      : `found ${share} of the evidence`;
    if (run.recorded === false) {
      evidence = share === null ? "evidence not scored (the run stopped before the answer)"
        : `the context the model would read holds ${share} of the evidence`;
    }
    const files = [...new Set(run.sources)].join(", ") || "none";
    const given = run.recorded === false ? "files the model would read" : "files given to the LLM";
    // Adaptive, Corrective and Iterative RAG: how their loop went, e.g. "1 of 2 rounds, stopped: evidence is good"
    const loop = run.loop ? ` · loop: ${RagSteps.describe(run.loop)}` : "";
    return `${evidence} · ${run.steps.length} steps · ${Math.round(run.ms)} ms · ${given}: ${files}`
      + loop;
  }

  // The answer, or why there is none: a run past the recording stops at a model step.
  function answerText(run, label) {
    if (run.recorded !== false) return run.answer;
    if (!run.stopped_at) return "Not recorded for these settings.";
    const where = `Not recorded for these settings: the run stopped at “${label(run.stopped_at)}”.`;
    return run.stopped_at === "generate"
      ? `${where} Click it to see exactly what the model would read.`
      : `${where} That model reply was never recorded; try the values marked as recorded.`;
  }

  // one setup's run: title, live diagram, answer, numbers, and the details of a clicked step
  function card(flow, title, run) {
    const box = el("div", "rp-card");
    const holder = el("div"), details = el("div", "rp-details");
    const label = (id) => (flow.nodes.find((node) => node.id === id) || { label: id }).label;
    box.append(el("h3", "", title), holder,
      el("p", run.answered ? "rp-answer" : "rp-answer rp-none", answerText(run, label)),
      el("p", "rp-stats", stats(run)), details);
    const diagram = RagFlow.render(holder, flow, {
      label: title,
      onSelect: (id, visits) => details.replaceChildren(RagSteps.details(label(id), visits)),
    });
    // after the run, open the search step: the fused ranks for hybrid, else the one search
    // (or the step where a run past the recording stopped)
    const has = (id) => flow.nodes.some((node) => node.id === id);
    const search = has("fuse") ? "fuse" : "retrieve";
    // a type's own step says the most: Parent-Child's expansion, HyDE's and RAG-Fusion's
    // fusing, Corrective's grades, Iterative's thinking (when this run used it)
    const used = (id) => has(id) && run.steps.some((step) => step.step === id);
    const open = run.stopped_at || ["expand", "fuse_queries", "grade", "think"].find(used)
      || search;
    return { box, play: (delay) => diagram.play(run.steps, delay || 650)
      .then(() => diagram.select(open)) };
  }

  function reportTable(report) {
    const table = el("table", "rp-report");
    const row = (cells, tag) => {
      const tr = el("tr");
      cells.forEach((cell) => tr.append(el(tag, "", cell)));
      table.append(tr);
    };
    row(["setup", "Recall", "MRR", "nDCG", "Answered", "Steps"], "th");
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

  return { el, choices, documentsBox, card, reportTable };
})();
