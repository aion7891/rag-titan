// Markdown blocks for the playground's markdown chunker, mirroring the Python package:
// rag_titan.chunking.markdown (blocks.py, block_pieces.py) and the heading sections of
// structure_aware/heading_chunker.py. Positions are JS string indexes; sizes are counted
// in characters (code points) by the caller, like Python's len().

const ChunkBlocks = (() => {
  const core = typeof ChunkCore !== "undefined" ? ChunkCore : require("./chunk-core.js");
  const FENCE = /^\s*(```|~~~)/;
  const HEADING = /^(#{1,6})[ \t]+(.+?)[ \t#]*$/;
  const LIST_ITEM = /^[ \t]*(?:[-*+]|\p{Nd}{1,3}[.)])[ \t]+\S/u;
  const TABLE_ROW = /^[ \t]*\|/;
  const QUOTE = /^[ \t]*>/;
  const TABLE_RULE = /^[ \t]*\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$/;
  const LED_KINDS = ["list", "table", "code"];

  // Python's str.splitlines(keepends=True)
  function splitLines(text) {
    const lines = [];
    const end = new RegExp("\\r\\n|[\\n\\r\\v\\f\\x1c\\x1d\\x1e\\x85\\u2028\\u2029]", "g");
    let start = 0;
    for (const match of text.matchAll(end)) {
      lines.push(text.slice(start, match.index + match[0].length));
      start = match.index + match[0].length;
    }
    if (start < text.length) lines.push(text.slice(start));
    return lines;
  }

  const isBlank = (line) => !line.trim();
  const isIndented = (line) => (line[0] === " " || line[0] === "\t") && !isBlank(line);
  const headingOf = (line) => HEADING.exec(line.replace(/[\r\n]+$/, ""));

  // heading_chunker.section_spans (heading lines left out): [path, body, start]
  function sectionSpans(text) {
    const sections = [];
    let levels = [], body = [], inCode = false, start = 0, position = 0;
    for (const line of splitLines(text)) {
      if (FENCE.test(line)) inCode = !inCode;
      const match = inCode ? null : headingOf(line);
      if (match) {
        sections.push([levels.map((l) => l[1]), body.join(""), start]);
        start = position + line.length;
        const level = match[1].length;
        levels = levels.filter((l) => l[0] < level);
        levels.push([level, match[2].trim()]);
        body = [];
      } else {
        body.push(line);
      }
      position += line.length;
    }
    sections.push([levels.map((l) => l[1]), body.join(""), start]);
    return sections.filter((s) => s[1].trim());
  }

  function blockKind(line) {
    if (FENCE.test(line)) return "code";
    if (headingOf(line)) return "heading";
    if (TABLE_ROW.test(line)) return "table";
    if (LIST_ITEM.test(line)) return "list";
    if (QUOTE.test(line)) return "quote";
    return "paragraph";
  }

  function fenceEnd(lines, start) {
    for (let end = start + 1; end < lines.length; end++) if (FENCE.test(lines[end])) return end + 1;
    return lines.length;
  }

  function listEnd(lines, start) {
    let end = start + 1;
    while (end < lines.length) {
      const line = lines[end];
      if (isBlank(line)) {
        const following = lines.slice(end).find((other) => !isBlank(other)) || "";
        if (!(LIST_ITEM.test(following) || isIndented(following))) break;
      } else if (!(LIST_ITEM.test(line) || isIndented(line) || blockKind(line) === "paragraph")) {
        break;
      }
      end += 1;
    }
    return end;
  }

  function blockEnd(lines, start, kind) {
    if (kind === "heading") return start + 1;
    if (kind === "code") return fenceEnd(lines, start);
    if (kind === "list") return listEnd(lines, start);
    let end = start + 1;
    while (end < lines.length && !isBlank(lines[end]) && blockKind(lines[end]) === kind) end += 1;
    return end;
  }

  // blocks.split_blocks: [[kind, text]], joining back into text
  function splitBlocks(text) {
    const lines = splitLines(text);
    const blocks = [];
    let start = 0, leading = "";
    while (start < lines.length) {
      if (isBlank(lines[start])) {
        if (blocks.length) blocks[blocks.length - 1][1] += lines[start];
        else leading += lines[start];
        start += 1;
        continue;
      }
      const kind = blockKind(lines[start]);
      const end = blockEnd(lines, start, kind);
      blocks.push([kind, leading + lines.slice(start, end).join("")]);
      start = end;
      leading = "";
    }
    return blocks;
  }

  // blocks.glue_lead_ins: [[kind, lead, body]]
  function glueLeadIns(blocks) {
    const units = [];
    for (const [kind, text] of blocks) {
      const previous = units[units.length - 1];
      const leads = previous && previous[0] === "paragraph" && !previous[1]
        && previous[2].trimEnd().endsWith(":");
      if (leads && LED_KINDS.includes(kind)) units[units.length - 1] = [kind, previous[2], text];
      else units.push([kind, "", text]);
    }
    return units;
  }

  // block_pieces.py
  // a blank line stays with the line before it; blank lines at the start, with the first line
  function linesWithBlanks(text) {
    const pieces = [];
    let leading = "";
    for (const line of splitLines(text)) {
      if (!isBlank(line)) { pieces.push(leading + line); leading = ""; }
      else if (pieces.length) pieces[pieces.length - 1] += line;
      else leading += line;
    }
    return leading ? [...pieces, leading] : pieces;
  }

  function listItems(text) {
    const ends = [];
    let position = 0;
    for (const line of splitLines(text)) {
      if (position && LIST_ITEM.test(line)) ends.push(position);
      position += line.length;
    }
    return core.cutAfter(text, ends);
  }

  function codeLines(text) {
    let lines = linesWithBlanks(text);
    if (lines.length >= 2) lines = [...lines.slice(0, -2), lines[lines.length - 2] + lines[lines.length - 1]];
    if (lines.length >= 2) lines = [lines[0] + lines[1], ...lines.slice(2)];
    return lines;
  }

  function tableParts(text) {
    const lines = linesWithBlanks(text);
    const size = lines.length > 1 && TABLE_RULE.test(lines[1].trim()) ? 2 : 1;
    return [lines.slice(0, size).join(""), lines.slice(size)];
  }

  function blockPieces(kind, text) {
    if (kind === "list") return listItems(text);
    if (kind === "code") return codeLines(text);
    if (kind === "quote" || kind === "table") return linesWithBlanks(text);
    return core.splitSentences(text);
  }

  return { splitLines, sectionSpans, splitBlocks, glueLeadIns, blockPieces, tableParts };
})();

if (typeof module !== "undefined") module.exports = ChunkBlocks;   // for tests in Node
