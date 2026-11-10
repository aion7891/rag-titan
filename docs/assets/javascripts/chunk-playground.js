// Chunk playground for docs/chunking.md: the controls and the live output.
// The chunking rules live in chunk-core.js (ChunkCore), which mirrors the Python package.

const PLAYGROUND = {
  text: "Leave policy\n\nStaff get twenty days. Unused days carry over. "
    + "Ask HR for the form. Sick leave is separate.",
  // per mode: [default size, default overlap, max size]
  modes: {
    recursive: [50, 25, 400],
    sentence: [50, 25, 400],
    paragraph: [60, 0, 400],
    character: [18, 4, 400],
    word: [4, 1, 60],
  },
};

function el(tag, props, children) {
  const node = Object.assign(document.createElement(tag), props || {});
  (children || []).forEach((child) => node.append(child));
  return node;
}

function slider(label, value, max) {
  const input = el("input", { type: "range", min: 0, max, value });
  const shown = el("output", { textContent: value });
  input.addEventListener("input", () => { shown.textContent = input.value; });
  return { input, shown, row: el("label", { className: "cp-field" }, [label, " ", shown, input]) };
}

function chunkBlock(chunk, index, unit) {
  const units = Array.from(chunk.text);
  const shared = el("mark", { textContent: units.slice(0, chunk.overlap).join("") });
  const rest = document.createTextNode(units.slice(chunk.overlap).join(""));
  const head = el("div", { className: "cp-head", textContent: `C${index} · ${unit.count(chunk.text)} ${unit.name}` });
  const body = el("div", { className: "cp-text" }, chunk.overlap ? [shared, rest] : [rest]);
  return el("div", { className: `cp-chunk cp-c${index % 6}` }, [head, body]);
}

function statsText(mode, text, size, overlap, chunks) {
  const unit = ChunkCore.UNITS[mode];
  const n = unit.count(text);
  const k = chunks.length;
  const h = size - overlap;
  const lines = [`N = ${n} ${unit.name} · S = ${size} · O = ${overlap} · H = S − O = ${h} · K = ${k} chunks`];
  if (unit.packed) {
    const stored = chunks.reduce((sum, c) => sum + Array.from(c.text).length, 0);
    lines.push(`stored ${stored} characters for ${n} (overlap repeats whole ${unit.packed} only)`);
  } else if (n > size) {
    const plain = ChunkCore.countWindows(n, size, 0);
    lines.push(`K = 1 + ⌈(N − S) / H⌉ = 1 + ⌈(${n} − ${size}) / ${h}⌉ = ${k}`);
    lines.push(`without overlap: ${plain} chunks, so overlap adds ${Math.round((k / plain - 1) * 100)}%`);
  }
  return lines.join("\n");
}

function buildPlayground(root) {
  root.dataset.ready = "1";
  const mode = el("select", {}, Object.keys(PLAYGROUND.modes).map(
    (name) => el("option", { value: name, textContent: name })));
  const size = slider("size", 50, 400);
  const overlap = slider("overlap", 25, 49);
  const text = el("textarea", { rows: 5, value: PLAYGROUND.text, spellcheck: false });
  const stats = el("pre", { className: "cp-stats" });
  stats.setAttribute("aria-live", "polite");
  const output = el("div", { className: "cp-chunks" });

  function render() {
    size.input.min = 1;
    overlap.input.max = Number(size.input.value) - 1;
    overlap.shown.textContent = overlap.input.value;
    const s = Number(size.input.value);
    const o = Number(overlap.input.value);
    const chunks = ChunkCore.UNITS[mode.value].run(text.value, s, o);
    stats.textContent = statsText(mode.value, text.value, s, o, chunks);
    const unit = ChunkCore.UNITS[mode.value];
    output.replaceChildren(...chunks.map((chunk, i) => chunkBlock(chunk, i, unit)));
  }

  mode.addEventListener("change", () => {
    const [s, o, max] = PLAYGROUND.modes[mode.value];
    size.input.max = max;
    size.input.value = size.shown.textContent = s;
    overlap.input.max = s - 1;
    overlap.input.value = overlap.shown.textContent = o;
    render();
  });
  [size.input, overlap.input, text].forEach((input) => input.addEventListener("input", render));

  const controls = el("div", { className: "cp-controls" },
    [el("label", { className: "cp-field" }, ["chunker ", mode]), size.row, overlap.row]);
  root.replaceChildren(controls, text, stats, output);
  render();
}

if (typeof document !== "undefined") {
  const start = () => {
    const root = document.getElementById("chunk-playground");
    if (root && !root.dataset.ready) buildPlayground(root);
  };
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
}
