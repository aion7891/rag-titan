// Web Worker (a module worker) for the RAG playground: runs the real rag_titan package with Pyodide,
// away from the page, so the page stays responsive while Python loads and runs.
// Messages in:  {id, call: "info" | "run" | "evaluate", args: [...]}
// Messages out: {id, result} or {id, error}, plus {status: "..."} while loading.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

let ready = null;   // a promise for the Python Playground object

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  return response;
}

async function start(assets) {
  postMessage({ status: "Loading Python (Pyodide, about 10 MB the first time)…" });
  const pyodide = await loadPyodide({ indexURL: PYODIDE });

  postMessage({ status: "Loading numpy…" });
  await pyodide.loadPackage("numpy");

  postMessage({ status: "Loading rag-titan and the recorded runs…" });
  const [bundle, presets] = await Promise.all([
    fetchOk(`${assets}playground/rag_titan.zip`).then((r) => r.arrayBuffer()),
    fetchOk(`${assets}playground/presets.json`).then((r) => r.text()),
  ]);
  pyodide.unpackArchive(bundle, "zip", { extractDir: "/home/pyodide/lib" });
  pyodide.globals.set("PRESETS_TEXT", presets);
  pyodide.runPython(`
import json, sys
sys.path.insert(0, "/home/pyodide/lib")
from rag_titan.playground import Playground
PLAYGROUND = Playground(json.loads(PRESETS_TEXT))
`);
  // every call goes through JSON, so the page gets plain objects
  return (call, args) => JSON.parse(pyodide.runPython(
    `json.dumps(getattr(PLAYGROUND, ${JSON.stringify(call)})(*json.loads(${JSON.stringify(JSON.stringify(args))})))`));
}

onmessage = async (event) => {
  const { id, call, args, assets } = event.data;
  try {
    if (!ready) ready = start(assets);
    const python = await ready;
    if (!["info", "run", "evaluate"].includes(call)) throw new Error(`Unknown call ${call}`);
    postMessage({ id, result: python(call, args || []) });
  } catch (error) {
    postMessage({ id, error: String(error.message || error) });
  }
};
