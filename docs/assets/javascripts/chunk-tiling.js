// The playground's texttiling chunker, mirroring rag_titan.chunking.statistical
// (cohesion.py, texttiling.py): sentences, word overlap of `window` sentences on each
// side of every gap, cuts at the deepest valleys, then each topic packed on its own.

const ChunkTiling = (() => {
  const core = typeof ChunkCore !== "undefined" ? ChunkCore : require("./chunk-core.js");
  const count = (text) => core.chars(text).length;
  // retrieval/bm25.py STOPWORDS (a test checks the two lists are the same)
  const STOPWORDS = new Set(`a about after again all am an and any are as at be because been
    before being both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its just me more most my
    no nor not now of off on once only or other our ours out over own same she should so some
    such than that the their them then there these they this those through to too under until
    up very was we were what when where which while who whom why will with would you your
    yours`.split(/\s+/));
  const WORD = /[\p{L}\p{N}_]+/gu;
  const ENDS_PARAGRAPH = /\n[ \t]*\n\s*$/;

  const contentWords = (sentence) =>
    (sentence.toLowerCase().match(WORD) || []).filter((word) => !STOPWORDS.has(word));

  function counts(bags) {
    const result = new Map();
    bags.forEach((bag) => bag.forEach((word) => result.set(word, (result.get(word) || 0) + 1)));
    return result;
  }

  function cosine(left, right) {
    if (!left.size || !right.size) return null;
    let dot = 0, a = 0, b = 0;
    left.forEach((c, word) => { dot += c * (right.get(word) || 0); });
    left.forEach((c) => { a += c * c; });
    right.forEach((c) => { b += c * c; });
    return dot / (Math.sqrt(a) * Math.sqrt(b));
  }

  function gapScores(bags, window) {
    const scores = [];
    for (let gap = 1; gap < bags.length; gap++) {
      if (gap < window || bags.length - gap < window) { scores.push(null); continue; }
      scores.push(cosine(counts(bags.slice(gap - window, gap)), counts(bags.slice(gap, gap + window))));
    }
    return scores;
  }

  function peak(scores, index, step) {
    let best = scores[index];
    index += step;
    while (index >= 0 && index < scores.length && scores[index] !== null && scores[index] >= best) {
      best = scores[index];
      index += step;
    }
    return best;
  }

  const depth = (s, i) => peak(s, i, -1) - s[i] + peak(s, i, 1) - s[i];
  const isValley = (s, i) =>
    s[i] <= Math.min(...s.slice(Math.max(0, i - 1), i + 2).filter((x) => x !== null));

  // cohesion.boundaries: the gaps where a new topic starts
  function boundaries(bags, window) {
    const scores = gapScores(bags, window);
    const deep = scores.map((s, i) => (s === null ? null : depth(scores, i)));
    const scored = deep.filter((d) => d !== null);
    if (!scored.length) return { cuts: [], scores, deep };
    const mean = scored.reduce((sum, d) => sum + d, 0) / scored.length;
    const spread = Math.sqrt(scored.reduce((sum, d) => sum + (d - mean) ** 2, 0) / scored.length);
    const cuts = [];
    deep.forEach((d, i) => {
      if (d !== null && d > 0 && d > mean - spread / 2 && isValley(scores, i)) cuts.push(i + 1);
    });
    return { cuts, scores, deep };
  }

  function snapToParagraphs(cuts, pieces) {
    const breaks = new Set();
    for (let i = 1; i < pieces.length; i++) if (ENDS_PARAGRAPH.test(pieces[i - 1])) breaks.add(i);
    const snapped = [];
    for (let cut of cuts) {
      if (!breaks.has(cut)) cut = [cut - 1, cut + 1].find((gap) => breaks.has(gap)) ?? cut;
      if (!snapped.includes(cut)) snapped.push(cut);
    }
    return snapped.sort((a, b) => a - b);
  }

  // TextTilingChunker.tiling: the sentences, the scores and the cuts
  // topics are found between whole sentences (a table or list has no full stops)
  function tiling(text, size, window) {
    const sentences = text.trim() ? core.splitSentences(text) : [];
    const found = boundaries(sentences.map(contentWords), window);
    return { sentences, ...found, cuts: snapToParagraphs(found.cuts, sentences) };
  }

  function byTiling(text, size, overlap, window) {
    if (!text.trim()) return [];
    const { sentences, cuts } = tiling(text, size, window || 3);
    const words = (p) => core.recursiveSplit(p, [" ", ""], size);
    const edges = [0, ...cuts, sentences.length];
    // a too-long sentence is cut at words only now, inside its topic
    return edges.slice(1).flatMap((end, i) =>
      core.packPieces(core.fit(sentences.slice(edges[i], end), size, words), size, overlap));
  }

  core.UNITS.texttiling = { run: byTiling, count, name: "characters", packed: "sentences", tiling };
  return { byTiling, tiling, STOPWORDS };
})();

if (typeof module !== "undefined") module.exports = ChunkTiling;   // for tests in Node
