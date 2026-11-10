// Chunk playground for docs/chunking.md: try the chunkers in the browser.
// The core mirrors rag_titan.chunking.fixed_size (same rules, same output).
// Text is handled as an array of characters (code points) to match Python.

const ChunkCore = (() => {
  const SEPARATORS = ["\n\n", "\n", ". ", " ", ""];
  const chars = (text) => Array.from(text);

  function fixedWindows(units, size, overlap) {
    const step = size - overlap;
    const windows = [];
    for (let start = 0; start < units.length; start += step) {
      windows.push(units.slice(start, start + size));
      if (start + size >= units.length) break;   // tail is not repeated
    }
    return windows;
  }

  function countWindows(n, size, overlap) {
    if (n <= size) return Math.min(n, 1);
    return 1 + Math.ceil((n - size) / (size - overlap));
  }

  function byCharacter(text, size, overlap) {
    if (!text.trim()) return [];
    return fixedWindows(chars(text), size, overlap).map((w, i) => ({
      text: w.join(""), overlap: i === 0 ? 0 : overlap,
    }));
  }

  function byWord(text, size, overlap) {
    const words = text.split(/\s+/).filter(Boolean);
    return fixedWindows(words, size, overlap).map((w, i) => ({
      text: w.join(" "),
      // shared words plus the space after them, in characters
      overlap: i === 0 ? 0 : chars(w.slice(0, overlap).join(" ")).length + 1,
    }));
  }

  function splitKeepSeparator(text, sep) {
    const parts = text.split(sep);
    const pieces = parts.slice(0, -1).map((p) => p + sep).concat(parts[parts.length - 1]);
    return pieces.filter(Boolean);
  }

  function recursiveSplit(text, separators, size) {
    if (chars(text).length <= size) return [text];
    for (let i = 0; i < separators.length; i++) {
      const sep = separators[i];
      if (sep === "") break;
      if (text.includes(sep)) {
        return splitKeepSeparator(text, sep)
          .flatMap((part) => recursiveSplit(part, separators.slice(i + 1), size));
      }
    }
    return fixedWindows(chars(text), size, 0).map((w) => w.join(""));   // hard cut
  }

  function recursiveMerge(pieces, size, overlap) {
    const sizes = pieces.map((p) => chars(p).length);
    const total = (ids) => ids.reduce((sum, i) => sum + sizes[i], 0);
    const groups = [];
    let current = [];
    let carried = 0;
    sizes.forEach((s, index) => {
      if (current.length && total(current) + s > size) {
        groups.push({ ids: current, carried });
        let tail = [];
        for (const i of [...current].reverse()) {   // last whole pieces <= overlap
          if (total(tail) + sizes[i] > overlap) break;
          tail.unshift(i);
        }
        while (tail.length && total(tail) + s > size) tail.shift();
        current = tail;
        carried = tail.length;
      }
      current.push(index);
    });
    if (current.length) groups.push({ ids: current, carried });
    return groups;
  }

  function byRecursive(text, size, overlap) {
    if (!text.trim()) return [];
    const pieces = recursiveSplit(text, SEPARATORS, size);
    return recursiveMerge(pieces, size, overlap).map(({ ids, carried }) => {
      const raw = ids.map((i) => pieces[i]).join("");
      const lead = chars(raw).length - chars(raw.trimStart()).length;
      const repeated = chars(ids.slice(0, carried).map((i) => pieces[i]).join("")).length;
      const text = raw.trim();
      // trimming can remove a whitespace-only new piece, so keep the highlight inside the chunk
      return { text, overlap: Math.min(Math.max(0, repeated - lead), chars(text).length) };
    }).filter((chunk) => chunk.text);
  }

  const UNITS = {
    character: { run: byCharacter, count: (t) => chars(t).length, name: "characters" },
    word: { run: byWord, count: (t) => t.split(/\s+/).filter(Boolean).length, name: "words" },
    recursive: { run: byRecursive, count: (t) => chars(t).length, name: "characters" },
  };

  return { fixedWindows, countWindows, UNITS };
})();

// ---------- the page: controls and live output ----------

const PLAYGROUND = {
  text: "Leave policy\n\nStaff get twenty days. Unused days carry over. "
    + "Ask HR for the form. Sick leave is separate.",
  // per mode: [default size, default overlap, max size]
  modes: { recursive: [50, 25, 400], character: [18, 4, 400], word: [4, 1, 60] },
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
  if (mode === "recursive") {
    const stored = chunks.reduce((sum, c) => sum + Array.from(c.text).length, 0);
    lines.push(`stored ${stored} characters for ${n} (overlap repeats whole pieces only)`);
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

if (typeof module !== "undefined") {
  module.exports = ChunkCore;   // for tests in Node
} else {
  const start = () => {
    const root = document.getElementById("chunk-playground");
    if (root && !root.dataset.ready) buildPlayground(root);
  };
  if (typeof document$ !== "undefined") document$.subscribe(start);
  else document.addEventListener("DOMContentLoaded", start);
}
