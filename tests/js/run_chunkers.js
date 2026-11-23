// Run the chunk playground's JS chunkers on cases from a JSON file (see test_chunk_js_parity.py).
//   node tests/js/run_chunkers.js cases.json  ->  {"chunks": [[text, ...], ...], "stopwords": [...]}

const fs = require("fs");
const path = require("path");

const scripts = path.join(__dirname, "..", "..", "docs", "assets", "javascripts");
const core = require(path.join(scripts, "chunk-core.js"));
require(path.join(scripts, "chunk-markdown.js"));
const tiling = require(path.join(scripts, "chunk-tiling.js"));

const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const chunks = cases.map(({ mode, text, size, overlap, window }) =>
  core.UNITS[mode].run(text, size, overlap, window).map((chunk) => chunk.text));
process.stdout.write(JSON.stringify({ chunks, stopwords: [...tiling.STOPWORDS].sort() }));
