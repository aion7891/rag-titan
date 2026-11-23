// The playground's markdown chunker, mirroring rag_titan.chunking.markdown.MarkdownChunker
// (with its heading path, include_path=True): whole blocks packed per heading section;
// a too-big table gets chunks of its own that repeat its lead-in and header row.
// A chunk is {text, context, overlap}: `context` characters at the start were added
// (heading path, repeated table header), then `overlap` characters repeat the chunk before.

const ChunkMarkdown = (() => {
  const core = typeof ChunkCore !== "undefined" ? ChunkCore : require("./chunk-core.js");
  const blocks = typeof ChunkBlocks !== "undefined" ? ChunkBlocks : require("./chunk-blocks.js");
  const count = (text) => core.chars(text).length;
  const isSpace = (c) => /\s/.test(c);

  // offsets.trim_span
  function trimSpan(text, start, end) {
    while (start < end && isSpace(text[start])) start += 1;
    while (end > start && isSpace(text[end - 1])) end -= 1;
    return [start, end];
  }

  // BlockPacker for one section: `size` is the room left after the heading path
  function packer(size, overlap) {
    const words = (piece, room) => core.recursiveSplit(piece, [" ", ""], room || size);
    const fit = (pieces, room) => pieces.flatMap((p) => (count(p) > room ? words(p, room) : [p]));
    const located = (pieces, start) => pieces.map((piece) => {
      const item = [piece, start];
      start += piece.length;
      return item;
    });

    function unitPieces(kind, lead, body) {
      if (count(lead + body) <= size) return [lead + body];
      return [...fit(blocks.blockPieces("paragraph", lead), size),
              ...fit(blocks.blockPieces(kind, body), size)];
    }

    // pack_run: chunks of whole located pieces, `context` in front of each
    function packRun(text, run, room, context) {
      if (!run.length) return [];
      const limit = room || size;
      const groups = core.pack(run.map(([piece]) => count(piece)), limit,
                               Math.min(overlap, limit - 1));
      return groups.flatMap(({ ids, carried }) => {
        const last = run[ids[ids.length - 1]];
        const [start, end] = trimSpan(text, run[ids[0]][1], last[1] + last[0].length);
        if (start >= end) return [];
        const fresh = carried < ids.length ? run[ids[carried]][1] : end;
        const repeated = carried ? count(text.slice(start, Math.max(start, fresh))) : 0;
        const body = text.slice(start, end);
        return [{ text: (context || "") + body, context: count(context || ""),
                  overlap: Math.min(repeated, count(body)), start, end }];
      });
    }

    function tableChunks(text, lead, body, position) {
      const [header, rows] = blocks.tableParts(body);
      const context = (lead + header).trimStart();
      const room = size - count(context);
      if (room < size / 2) return packRun(text, located(unitPieces("table", lead, body), position));
      const rowStart = position + (lead + header).length;
      const chunks = packRun(text, located(fit(rows, room), rowStart), room, context);
      if (chunks.length) {   // the first chunk is the table's own start, header included
        const [start, end] = trimSpan(text, position, chunks[0].end);
        chunks[0] = { ...chunks[0], text: text.slice(start, end), context: 0, start };
      }
      return chunks;
    }

    return (text) => {
      if (!text.trim()) return [];
      let chunks = [], run = [], position = 0;
      for (const [kind, lead, body] of blocks.glueLeadIns(blocks.splitBlocks(text))) {
        if (kind === "table" && count(lead + body) > size) {
          chunks = chunks.concat(packRun(text, run), tableChunks(text, lead, body, position));
          run = [];
        } else {
          run = run.concat(located(unitPieces(kind, lead, body), position));
        }
        position += (lead + body).length;
      }
      return chunks.concat(packRun(text, run));
    };
  }

  // MarkdownChunker: every heading section on its own, its path in front of each chunk
  function byMarkdown(text, size, overlap) {
    if (!text.trim()) return [];
    return blocks.sectionSpans(text).flatMap(([path, body]) => {
      let prefix = path.length ? `${path.join(" > ")}\n\n` : "";
      let room = size - count(prefix);
      if (room < size / 2) { prefix = ""; room = size; }
      return packer(room, Math.min(overlap, room - 1))(body).map((chunk) => ({
        text: prefix + chunk.text, context: count(prefix) + chunk.context, overlap: chunk.overlap,
      }));
    });
  }

  core.UNITS.markdown = { run: byMarkdown, count, name: "characters", packed: "blocks" };
  return { byMarkdown };
})();

if (typeof module !== "undefined") module.exports = ChunkMarkdown;   // for tests in Node
