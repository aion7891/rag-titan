// Light up a run on a RAG flow diagram drawn by rag-flow.js: step numbers on the boxes,
// the path taken in orange, a failed step dashed. rag-flow.js calls RagFlowPlay.api().

const RagFlowPlay = (() => {
  // step numbers of a node's visits, runs shortened: [2, 3, 4, 5] -> "2–5", [2, 5] -> "2 · 5"
  function badgeText(numbers) {
    const parts = [];
    numbers.forEach((n, i) => {
      const last = parts[parts.length - 1];
      if (i > 0 && n === numbers[i - 1] + 1) last[1] = n; else parts.push([n, n]);
    });
    return parts.map(([a, b]) => (a === b ? `${a}` : b === a + 1 ? `${a} · ${b}` : `${a}–${b}`))
      .join(" · ");
  }

  function diagramApi(nodes, edges, onSelect) {
    let timer = null;
    const reset = () => {
      clearTimeout(timer);
      Object.values(nodes).forEach((n) => { n.visits = []; n.badge.textContent = "";
        n.group.classList.remove("rf-visited", "rf-current", "rf-failed"); });
      Object.values(edges).forEach((edge) => {
        edge.classList.remove("rf-taken");
        if (edge.dataset.plain) edge.setAttribute("marker-end", edge.dataset.plain);
      });
    };
    const light = (steps, i) => {
      const step = steps[i];
      const node = nodes[step.step];
      if (!node) throw new Error(`Step "${step.step}" is not in this FLOW`);
      Object.values(nodes).forEach((n) => n.group.classList.remove("rf-current"));
      node.visits.push({ number: i + 1, step });
      node.badge.textContent = badgeText(node.visits.map((v) => v.number));
      node.group.classList.add("rf-visited", "rf-current");
      // a step that failed (a model reply not recorded) is drawn dashed
      if (step.output && typeof step.output === "object" && "error" in step.output) {
        node.group.classList.add("rf-failed");
      }
      const edge = i > 0 && edges[`${steps[i - 1].step}>${step.step}`];
      if (edge) {
        edge.dataset.plain = edge.dataset.plain || edge.getAttribute("marker-end");
        edge.classList.add("rf-taken");
        edge.setAttribute("marker-end", edge.dataset.taken);
      }
    };
    const still = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    return {
      reset,
      // open a node's details from code, as if it was clicked
      select(id) { if (onSelect && nodes[id]) onSelect(id, nodes[id].visits); },
      show(steps) { reset(); steps.forEach((_, i) => light(steps, i)); },
      play(steps, delay) {
        reset();
        if (still()) { steps.forEach((_, i) => light(steps, i)); return Promise.resolve(); }
        return new Promise((done) => {
          const next = (i) => {
            if (i === steps.length) { done(); return; }
            light(steps, i);
            timer = setTimeout(() => next(i + 1), delay || 650);
          };
          next(0);
        });
      },
    };
  }

  return { api: diagramApi, badgeText };
})();
