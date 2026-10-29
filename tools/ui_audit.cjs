#!/usr/bin/env node
/*
 * Reel UI audit (dev tool, not needed to run the app).
 * Drives a real Chromium against a running Reel and measures, on the rendered DOM:
 *   - text contrast (target >= 4.5:1), including placeholders
 *   - icon contrast (>= 3:1)
 *   - control boundary contrast (>= 3:1): border, or fill when there is no border
 *   - selected vs hover distinguishability on nav items
 *   - OS-follow, manual toggle and persistence
 *   - raw colour values outside tokens.css
 * It reports; it never changes styles.
 *
 * Usage: node tools/ui_audit.cjs [--url http://127.0.0.1:8765] [--routes generate,library,settings,styleguide]
 *                                [--shots DIR] [--json FILE]
 * Needs: npm i playwright (and a Chromium it can launch).
 */
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith("--")) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith("--") ? arr[i + 1] : true]);
  return acc;
}, []));
const URL = args.url || "http://127.0.0.1:8765";
const ROUTES = (args.routes || "generate,library,settings,styleguide").split(",");
const SHOTS = args.shots || null;
if (SHOTS) fs.mkdirSync(SHOTS, { recursive: true });

// ---- runs inside the page ------------------------------------------------
function pageAudit() {
  const parse = (s) => {
    if (!s || s === "transparent") return { r: 0, g: 0, b: 0, a: 0 };
    let m = s.match(/^rgba?\(([^)]+)\)$/);
    if (m) {
      const p = m[1].split(/[,\s/]+/).filter(Boolean).map(Number);
      return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 };
    }
    m = s.match(/^color\(srgb ([^)]+)\)$/);
    if (m) {
      const p = m[1].split(/[\s/]+/).filter(Boolean).map(Number);
      return { r: p[0] * 255, g: p[1] * 255, b: p[2] * 255, a: p.length > 3 ? p[3] : 1 };
    }
    throw new Error("unparsed colour: " + s);
  };
  const over = (top, bottom) => {
    const a = top.a + bottom.a * (1 - top.a);
    if (a === 0) return { r: 0, g: 0, b: 0, a: 0 };
    const f = (t, b) => (t * top.a + b * bottom.a * (1 - top.a)) / a;
    return { r: f(top.r, bottom.r), g: f(top.g, bottom.g), b: f(top.b, bottom.b), a };
  };
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };
  const ratio = (a, b) => {
    const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  };
  const canvas = parse(getComputedStyle(document.body).backgroundColor);
  const effBg = (el) => {
    const chain = [];
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) chain.push(parse(getComputedStyle(n).backgroundColor));
    let acc = { ...canvas, a: 1 };
    for (let i = chain.length - 1; i >= 0; i--) acc = over(chain[i], acc);
    return acc;
  };
  const visible = (el) => {
    if (!el.getClientRects().length) return false;
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      const cs = getComputedStyle(n);
      if (cs.visibility === "hidden" || cs.display === "none" || parseFloat(cs.opacity) === 0) return false;
    }
    return true;
  };
  const disabled = (el) => !!el.closest(":disabled, [aria-disabled='true'], fieldset:disabled");
  const hex = (c) => "#" + [c.r, c.g, c.b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
  const label = (el) => {
    const cls = typeof el.className === "string" ? el.className.trim().split(/\s+/).slice(0, 3).join(".") : "";
    const txt = (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 32);
    return `${el.tagName.toLowerCase()}${cls ? "." + cls : ""}${txt ? ` "${txt}"` : ""}`;
  };

  const text = [], icons = [], controls = [];

  // Text
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const seen = new Set();
  for (let t = walker.nextNode(); t; t = walker.nextNode()) {
    if (!t.nodeValue.trim()) continue;
    const el = t.parentElement;
    if (!el || seen.has(el) || ["SCRIPT", "STYLE", "OPTION"].includes(el.tagName)) continue;
    seen.add(el);
    if (!visible(el) || disabled(el)) continue;
    const bg = effBg(el);
    const fg = over(parse(getComputedStyle(el).color), bg);
    text.push({ el: label(el), ratio: ratio(fg, bg), fg: hex(fg), bg: hex(bg), size: getComputedStyle(el).fontSize });
  }
  // Placeholders
  for (const el of document.querySelectorAll("input[placeholder], textarea[placeholder]")) {
    if (!visible(el) || disabled(el) || el.value) continue;
    const bg = effBg(el);
    const fg = over(parse(getComputedStyle(el, "::placeholder").color), bg);
    text.push({ el: label(el) + " [placeholder]", ratio: ratio(fg, bg), fg: hex(fg), bg: hex(bg), size: getComputedStyle(el).fontSize });
  }
  // Icons (non-text graphics, 3:1)
  for (const svg of document.querySelectorAll("svg")) {
    if (!visible(svg) || disabled(svg)) continue;
    const bg = effBg(svg);
    const fg = over(parse(getComputedStyle(svg).color), bg);
    icons.push({ el: label(svg.parentElement), ratio: ratio(fg, bg), fg: hex(fg), bg: hex(bg) });
  }
  // Control boundaries
  const sel = ".btn, .input, .select, .textarea, .seg";
  for (const el of document.querySelectorAll(sel)) {
    if (!visible(el) || disabled(el)) continue;
    const cs = getComputedStyle(el);
    const outside = effBg(el.parentElement);
    const bw = parseFloat(cs.borderTopWidth);
    const border = parse(cs.borderTopColor);
    let kind, boundary;
    if (bw > 0 && border.a > 0) { kind = "border"; boundary = over(border, outside); }
    else { kind = "fill"; boundary = over(parse(cs.backgroundColor), outside); }
    const cls = (el.className || "").split(/\s+/).filter((c) => /^(btn|input|select|textarea|seg)/.test(c)).join(".");
    controls.push({ el: label(el), cls, kind, ratio: ratio(boundary, outside), boundary: hex(boundary), outside: hex(outside) });
  }
  return { text, icons, controls, ground: hex(canvas) };
}

// ---- driver ---------------------------------------------------------------
const THR = { text: 4.5, icon: 3, control: 3 };
const results = { runs: [], toggle: {}, selected_vs_hover: {}, failures: [] };

async function gotoRoute(page, route) {
  await page.goto(`${URL}/#/${route}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(400);
}

(async () => {
  const browser = await chromium.launch();
  for (const scheme of ["light", "dark"]) {
    const ctx = await browser.newContext({ colorScheme: scheme, viewport: { width: 1360, height: 860 } });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => results.failures.push(`[${scheme}] page error: ${e.message}`));
    for (const route of ROUTES) {
      await gotoRoute(page, route);
      const a = await page.evaluate(pageAudit);
      const textFails = a.text.filter((t) => t.ratio < THR.text);
      const iconFails = a.icons.filter((t) => t.ratio < THR.icon);
      const byCls = {};
      for (const c of a.controls) {
        const k = `${c.cls || c.el.split(" ")[0]} (${c.kind})`;
        if (!byCls[k] || c.ratio < byCls[k].ratio) byCls[k] = c;
      }
      const controlSummary = Object.entries(byCls).map(([k, c]) => ({
        control: k, min_ratio: +c.ratio.toFixed(2), boundary: c.boundary, outside: c.outside,
        // Borderless controls (ghost, soft-danger) have no border to measure; they are identified by their
        // label, whose text contrast is checked above. Reported as notes, not failures.
        borderless: c.kind === "fill" && c.ratio < THR.control, pass: c.ratio >= THR.control }));
      results.runs.push({
        scheme, route, ground: a.ground,
        text_checked: a.text.length, text_min: +Math.min(...a.text.map((t) => t.ratio), 99).toFixed(2),
        text_lowest: a.text.slice().sort((x,y)=>x.ratio-y.ratio).slice(0,4).map((t)=>({el:t.el,ratio:+t.ratio.toFixed(2),fg:t.fg,bg:t.bg})),
        text_fails: textFails.map((t) => ({ ...t, ratio: +t.ratio.toFixed(2) })),
        icons_checked: a.icons.length, icon_fails: iconFails.map((t) => ({ ...t, ratio: +t.ratio.toFixed(2) })),
        controls: controlSummary,
      });
      if (SHOTS) {
        if (route === "styleguide") await page.setViewportSize({ width: 1360, height: 2300 });
        await page.screenshot({ path: path.join(SHOTS, `${route}-${scheme}.png`) });
        await page.setViewportSize({ width: 1360, height: 860 });
      }
    }

    // selected vs hover on nav
    await gotoRoute(page, ROUTES[0]);
    const sv = await page.evaluate(() => {
      const nav = document.querySelector("#nav");
      const cur = nav.querySelector("[aria-current='page']");
      const other = [...nav.querySelectorAll("a")].find((a) => a !== cur);
      return { cur: cur && cur.textContent.trim(), other: other && other.textContent.trim() };
    });
    const hoverEl = page.locator("#nav a:not([aria-current])").first();
    await hoverEl.hover();
    await page.waitForTimeout(250);
    const state = await page.evaluate(() => {
      const nav = document.querySelector("#nav");
      const cur = nav.querySelector("[aria-current='page']");
      const hov = [...nav.querySelectorAll("a:hover")][0];
      const cs = (e) => getComputedStyle(e);
      const bar = getComputedStyle(cur, "::before");
      return {
        selected_bg: cs(cur).backgroundColor, hover_bg: hov && cs(hov).backgroundColor,
        selected_weight: cs(cur).fontWeight, hover_weight: hov && cs(hov).fontWeight,
        selected_bar_width: bar.width, hover_bar: hov ? getComputedStyle(hov, "::before").content : null,
        selected_color: cs(cur).color, hover_color: hov && cs(hov).color,
      };
    });
    results.selected_vs_hover[scheme] = state;
    await ctx.close();
  }

  // Toggle + persistence + OS-follow: OS dark, no saved choice => dark; click Light => light; reload => still light.
  const ctx = await browser.newContext({ colorScheme: "dark", viewport: { width: 1360, height: 860 } });
  const page = await ctx.newPage();
  await gotoRoute(page, ROUTES[0]);
  const bodyBg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  const attr = () => page.evaluate(() => document.documentElement.getAttribute("data-theme"));
  results.toggle.os_dark = { data_theme: await attr(), body_bg: await bodyBg() };
  await page.click("#theme-toggle button:has-text('Light')");
  results.toggle.after_click_light = { data_theme: await attr(), body_bg: await bodyBg() };
  await page.reload({ waitUntil: "networkidle" });
  results.toggle.after_reload = { data_theme: await attr(), body_bg: await bodyBg(), stored: await page.evaluate(() => localStorage.getItem("reel.theme")) };
  await page.click("#theme-toggle button:has-text('Auto')");
  results.toggle.back_to_auto = { data_theme: await attr(), body_bg: await bodyBg(), stored: await page.evaluate(() => localStorage.getItem("reel.theme")) };
  await ctx.close();
  await browser.close();

  // Raw colours outside tokens.css
  const root = path.join(__dirname, "..", "reel", "static");
  const offenders = [];
  const re = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|\boklch\(|\boklab\(/;
  (function walk(dir) {
    for (const f of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, f.name);
      if (f.isDirectory()) { if (f.name !== "icons") walk(p); continue; }
      if (!/\.(css|js|html)$/.test(f.name) || f.name === "tokens.css") continue;
      fs.readFileSync(p, "utf8").split("\n").forEach((line, i) => { if (re.test(line.replace(/&#\d+;/g, ""))) offenders.push(`${path.relative(root, p)}:${i + 1}: ${line.trim().slice(0, 90)}`); });
    }
  })(root);
  results.raw_colour_offenders = offenders;

  if (args.json) fs.writeFileSync(args.json, JSON.stringify(results, null, 2));

  // ---- report ----
  const line = (s = "") => console.log(s);
  for (const r of results.runs) {
    line(`\n[${r.scheme}] #/${r.route}   (page ground ${r.ground})`);
    line(`  text: ${r.text_checked} checked, min ${r.text_min}:1, ${r.text_fails.length} below ${THR.text}:1`);
    for (const f of r.text_fails.slice(0, 12)) line(`    FAIL ${f.ratio}:1  ${f.el}  fg ${f.fg} on ${f.bg}`);
    line(`  icons: ${r.icons_checked} checked, ${r.icon_fails.length} below ${THR.icon}:1`);
    for (const f of r.icon_fails.slice(0, 8)) line(`    FAIL ${f.ratio}:1  ${f.el}`);
    for (const c of r.controls) line(`  control ${c.pass ? "ok  " : c.borderless ? "note" : "FAIL"} ${String(c.min_ratio).padStart(5)}:1  ${c.control}  ${c.boundary} on ${c.outside}${c.borderless ? "  (no border; label-identified)" : ""}`);
  }
  line("\nSelected vs hover (nav):"); line(JSON.stringify(results.selected_vs_hover, null, 2));
  line("\nTheme toggle:"); line(JSON.stringify(results.toggle, null, 2));
  line(`\nRaw colour values outside tokens.css: ${offenders.length}`); offenders.forEach((o) => line("  " + o));
  if (results.failures.length) { line("\nPage errors:"); results.failures.forEach((f) => line("  " + f)); }
})().catch((e) => { console.error(e); process.exit(1); });
