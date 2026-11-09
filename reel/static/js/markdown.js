// A small, safe Markdown renderer for chat replies. It builds DOM nodes (never HTML strings), so nothing a model
// writes can inject markup or script. Supports what chat models actually produce: paragraphs, headings, bold,
// italic, inline code, fenced code (with a copy button), lists, quotes, links, tables and rules.
import { h, icon, copyText } from "./ui.js";

const SAFE_URL = /^(https?:\/\/|mailto:)/i;

function inline(text) {
  const out = [];
  const re = /(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)|\*\*([^*][\s\S]*?)\*\*|__([^_][\s\S]*?)__|\*([^*\s][^*]*?)\*|(?<![\w])_([^_\s][^_]*?)_(?![\w])|\[([^\]]+)\]\(([^)\s]+)\)|(https?:\/\/[^\s<>)\]]+[^\s<>)\].,;:!?])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[2] !== undefined) out.push(h("code", {}, m[2].trim()));
    else if (m[3] !== undefined || m[4] !== undefined) out.push(h("strong", {}, ...inline(m[3] ?? m[4])));
    else if (m[5] !== undefined || m[6] !== undefined) out.push(h("em", {}, ...inline(m[5] ?? m[6])));
    else if (m[7] !== undefined) out.push(SAFE_URL.test(m[8]) ? h("a", { href: m[8], target: "_blank", rel: "noopener noreferrer" }, ...inline(m[7])) : m[0]);
    else out.push(h("a", { href: m[9], target: "_blank", rel: "noopener noreferrer" }, m[9]));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function codeBlock(lang, code) {
  const copy = h("button", { type: "button", class: "btn btn-ghost code-copy", title: "Copy code", "aria-label": "Copy code", onclick: () => copyText(code) }, icon("copy"));
  return h("div", { class: "code-block" },
    h("div", { class: "code-head" }, h("span", { class: "mono" }, lang || "code"), copy),
    h("pre", { class: "mono" }, h("code", {}, code)));
}

const cells = (line) => line.replace(/^\s*\||\|\s*$/g, "").split("|").map((c) => c.trim());
const LI = /^(\s*)([-*+]|\d+[.)])\s+(.*)/;
const indentOf = (l) => l.length - l.trimStart().length;
const isRule = (l) => /^\s*([-*_])(\s*\1){2,}\s*$/.test(l);
const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l) && l.includes("-");

export function renderMarkdown(src) {
  const root = h("div", { class: "md" });
  const lines = String(src).replace(/\r\n/g, "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    const fence = line.match(/^\s*(`{3,}|~{3,})\s*([\w+#.-]*)/);
    if (fence) {
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence[1])) buf.push(lines[i++]);
      i++;
      root.append(codeBlock(fence[2], buf.join("\n")));
      continue;
    }
    const head = line.match(/^(#{1,6})\s+(.*?)\s*#*$/);
    if (head) { root.append(h(`h${Math.min(head[1].length + 2, 6)}`, {}, ...inline(head[2]))); i++; continue; }
    if (isRule(line)) { root.append(h("hr")); i++; continue; }

    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const header = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
      root.append(h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, header.map((c) => h("th", {}, ...inline(c))))),
        h("tbody", {}, rows.map((r) => h("tr", {}, header.map((_, k) => h("td", {}, ...inline(r[k] ?? ""))))))
      )));
      continue;
    }
    if (/^\s*>/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
      const q = h("blockquote");
      q.append(renderMarkdown(buf.join("\n")));
      root.append(q);
      continue;
    }
    const li = line.match(LI);
    if (li) {
      const ordered = /\d/.test(li[2]);
      const base = li[1].length;
      const list = h(ordered ? "ol" : "ul");
      if (ordered && Number(li[2].match(/\d+/)[0]) !== 1) list.start = Number(li[2].match(/\d+/)[0]);
      while (i < lines.length) {
        const m = lines[i].match(LI);
        if (!m || m[1].length !== base || /\d/.test(m[2]) !== ordered) break;
        const item = h("li", {}, ...inline(m[3]));
        i++;
        const nested = [];
        while (i < lines.length) {
          const l = lines[i];
          if (!l.trim()) {                      // a blank line belongs to the item only if more indented text follows
            if (i + 1 < lines.length && indentOf(lines[i + 1]) > base) { nested.push(""); i++; continue; }
            break;
          }
          if (indentOf(l) <= base) break;
          nested.push(l.replace(new RegExp(`^ {0,${base + 2}}`), ""));
          i++;
        }
        if (nested.length) item.append(renderMarkdown(nested.join("\n")));
        list.append(item);
        if (!lines[i]?.trim() && i + 1 < lines.length) { const n = lines[i + 1].match(LI); if (n && n[1].length === base && /\d/.test(n[2]) === ordered) i++; }
      }
      root.append(list);
      continue;
    }
    const buf = [];
    while (i < lines.length && lines[i].trim() && !/^\s*(`{3,}|~{3,}|#{1,6}\s|>|([-*+]|\d+[.)])\s)/.test(lines[i]) && !isRule(lines[i])
           && !(lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]))) buf.push(lines[i++]);
    if (!buf.length) { buf.push(lines[i++]); }
    const p = h("p");
    buf.forEach((l, k) => { if (k) p.append(h("br")); p.append(...inline(l)); });
    root.append(p);
  }
  return root;
}

// Reasoning models write their thinking in <think>…</think> (or the server splits it out already).
export function splitThinking(text) {
  const m = text.match(/^\s*<think>([\s\S]*?)(<\/think>|$)/);
  if (!m) return { reasoning: "", answer: text, open: false };
  return { reasoning: m[1].trim(), answer: text.slice(m[0].length).trimStart(), open: m[2] === "" };
}
