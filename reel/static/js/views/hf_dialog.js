// "Add model": search Hugging Face, inspect a repo, and either add an image/video recipe or download a GGUF language model.
import { h, icon, clear, toast, fmtCount, fmtBytes, kindBadge, statusBadge } from "../ui.js";
import { api } from "../api.js";

// Null-safe append: native Element.append(null) would write the text "null".
const put = (el, ...items) => el.append(...items.flat(Infinity).filter((x) => x !== null && x !== undefined && x !== false));

const looksLikeRepo = (t) => /^https?:\/\//.test(t) || /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/.test(t);

const stat = (label, value, sub) => h("div", { class: "stat" },
  h("span", { class: "stat-label" }, label), h("span", { class: "stat-value" }, value), sub ? h("span", { class: "stat-label" }, sub) : null);

export function openAddFromHF({ onAdded, mode: startMode = "language" }) {
  let mode = startMode;                      // "language" (GGUF, runs on llama.cpp) | "diffusers" (image and video)
  const input = h("input", { id: "hf-q", class: "input", type: "text", spellcheck: "false", autocomplete: "off" });
  const go = h("button", { class: "btn btn-primary", type: "submit" }, icon("search"), "Find");
  let kind = "all";
  const kindSeg = h("div", { class: "seg", role: "group", "aria-label": "Model type" });
  const modeSeg = h("div", { class: "seg", role: "group", "aria-label": "What kind of model" });
  const intro = h("p", { class: "muted" });
  const body = h("div", { class: "hf-body stack-12", "aria-live": "polite" });
  const dlg = h("dialog", { class: "dialog dialog-wide", "aria-labelledby": "hf-title" },
    h("div", { class: "stack-12" },
      h("h3", { id: "hf-title" }, "Add a model from Hugging Face"),
      modeSeg, intro,
      h("form", { class: "row-8 hf-search", onsubmit: (e) => { e.preventDefault(); submit(); } },
        h("div", { class: "grow" }, h("label", { class: "sr-only", for: "hf-q" }, "Link, name or search"), input), go),
      h("div", {}, kindSeg), body,
      h("div", { class: "dialog-actions" }, h("button", { class: "btn", type: "button", onclick: () => dlg.close() }, "Close"))));
  dlg.addEventListener("close", () => dlg.remove());

  function drawMode() {
    clear(modeSeg);
    for (const [v, label, ic] of [["language", "Language", "chat"], ["diffusers", "Image & video", "image"]]) {
      put(modeSeg, h("button", { type: "button", "aria-pressed": String(mode === v), onclick: () => { mode = v; drawMode(); drawKindSeg(); input.value = ""; search(); } }, icon(ic), label));
    }
    input.placeholder = mode === "language" ? "Search models (e.g. qwen, llama, gemma) or paste a link or org/name" : "Paste a link or org/name, or search (e.g. flux, sdxl, wan)";
    intro.textContent = mode === "language"
      ? "Chat models in GGUF format, run by llama.cpp. Pick a file size that fits your graphics card: 4-bit (Q4) is the usual sweet spot."
      : "Reel runs diffusers-format text-to-image and text-to-video models. It picks one weight format per part, so the download is far smaller than the repo.";
  }

  function drawKindSeg() {
    kindSeg.hidden = mode === "language";
    clear(kindSeg);
    for (const [v, label, ic] of [["all", "All", null], ["image", "Image", "image"], ["video", "Video", "generate"]]) {
      put(kindSeg, h("button", { type: "button", "aria-pressed": String(kind === v), onclick: () => { kind = v; drawKindSeg(); if (!looksLikeRepo(input.value.trim())) search(); } },
        ic ? icon(ic) : null, label));
    }
  }

  const busy = (msg) => { clear(body); put(body, h("div", { class: "muted icon-line" }, icon("clock"), msg)); };
  const fail = (msg) => { clear(body); put(body, h("div", { class: "notice notice-danger", role: "alert" }, icon("error"), msg)); };

  async function search() {
    busy("Searching Hugging Face…");
    try {
      const url = mode === "language" ? `/api/llm/discover?q=${encodeURIComponent(input.value.trim())}` : `/api/hf/search?kind=${kind}&q=${encodeURIComponent(input.value.trim())}`;
      const { results } = await api.get(url);
      clear(body);
      if (!results.length) { put(body, h("div", { class: "empty" }, "Nothing found. Try another search, or paste a model link.")); return; }
      put(body, h("div", { class: "muted", style: { fontSize: "var(--text-sm)" } }, input.value.trim() ? "Results, most downloaded first" : "Most downloaded"),
        h("ul", { class: "hf-results" }, results.map((r) => h("li", { class: "hf-row" },
          h("div", { class: "grow" },
            h("div", { class: "mono hf-name" }, r.id),
            h("div", { class: "muted hf-sub" }, `${fmtCount(r.downloads)} downloads · ${fmtCount(r.likes)} likes`)),
          mode === "language" ? null : kindBadge(r.kind), r.gated ? statusBadge("warning", "Gated", "alert") : null,
          h("button", { class: "btn", type: "button", onclick: () => inspect(r.id) }, "Inspect")))));
    } catch (e) { fail(e.message); }
  }

  async function inspectLanguage(repo) {
    busy(`Reading ${repo}…`);
    try {
      const r = await api.post("/api/llm/inspect", { repo });
      clear(body);
      const back = h("button", { class: "btn btn-ghost", type: "button", onclick: () => { clear(body); search(); } }, icon("back"), "Back to results");
      if (r.error) { put(body, h("div", { class: "mono hf-name" }, r.repo), h("div", { class: "notice notice-danger" }, icon("error"), r.error), back); return; }
      let chosen = (r.files.find((f) => f.name === r.recommended && !f.downloaded) || r.files.find((f) => !f.downloaded) || r.files[0]).name;
      const list = h("div", { class: "hf-files", role: "radiogroup", "aria-label": "File to download" });
      const dl = h("button", { class: "btn btn-primary", type: "button" });
      const draw = () => {
        clear(list);
        for (const f of r.files) {
          const on = f.name === chosen;
          list.append(h("button", { type: "button", role: "radio", "aria-checked": String(on), class: "item hf-file", "aria-current": on ? "true" : undefined, disabled: f.downloaded || undefined, onclick: () => { chosen = f.name; draw(); } },
            h("span", { class: "grow hf-file-main" }, h("span", { class: "mono hf-name" }, f.name), f.split ? h("span", { class: "muted hf-sub" }, `${f.files.length} parts`) : null),
            f.name === r.recommended ? statusBadge("info", "Recommended", "check") : null,
            f.downloaded ? statusBadge("success", "Downloaded", "check") : f.fit.state === "gpu" ? statusBadge("success", "Fits GPU", "check", f.fit.label)
              : f.fit.state === "partial" ? statusBadge("warning", "Partial GPU", "alert", f.fit.label) : statusBadge("neutral", "CPU speed", "chip", f.fit.label),
            h("strong", { class: "hf-size" }, fmtBytes(f.size))));
        }
        const f = r.files.find((x) => x.name === chosen);
        dl.disabled = !f || f.downloaded;
        clear(dl); dl.append(icon("download"), f && !f.downloaded ? `Download ${fmtBytes(f.size)}` : "Downloaded");
      };
      dl.onclick = async () => {
        dl.disabled = true;
        try {
          await api.post("/api/llm/models/download", { repo: r.repo, name: chosen });
          toast("Download started. Progress is in the Library.", "success", 5000);
          dlg.close(); onAdded();
        } catch (e) { toast(e.message, "danger", 9000); dl.disabled = false; }
      };
      put(body,
        h("div", { class: "row-between" }, h("h4", { class: "mono hf-name" }, r.repo), h("div", { class: "row-8" }, kindBadge("language"), r.gated ? statusBadge("warning", "Gated", "alert") : null)),
        h("div", { class: "muted text-sm" }, `${fmtCount(r.downloads)} downloads · ${fmtCount(r.likes)} likes${r.license ? ` · ${r.license}` : ""}`),
        r.gated ? h("div", { class: "notice notice-warning" }, icon("alert"), h("div", {}, "Gated model: accept its terms on ",
          h("a", { href: `https://huggingface.co/${r.repo}`, target: "_blank", rel: "noopener" }, "its Hugging Face page"), " and save an access token in Settings first.")) : null,
        r.vision_projectors ? h("div", { class: "notice notice-info" }, icon("info"), "This repo also has vision files. Reel chats with text only for now, so they are skipped.") : null,
        list, h("div", { class: "row-8" }, dl, back));
      draw();
    } catch (e) { fail(e.message); }
  }

  async function inspect(repo) {
    if (mode === "language") return inspectLanguage(repo);
    busy(`Reading ${repo}…`);
    try {
      const r = await api.post("/api/hf/inspect", { repo });
      clear(body);
      const back = h("button", { class: "btn btn-ghost", type: "button", onclick: () => { clear(body); search(); } }, icon("back"), "Back to results");
      if (r.unsupported || !r.recipe) {
        put(body, h("div", { class: "mono hf-name" }, r.repo),
          h("div", { class: "notice notice-danger" }, icon("error"), h("div", {}, h("strong", {}, "Reel can't run this one. "), r.unsupported)), back);
        return;
      }
      const rc = r.recipe;
      const saved = r.raw_gb > rc.download_gb + 0.5 ? `of ${r.raw_gb} GB in the repo` : null;
      const add = h("button", { class: "btn btn-primary", type: "button", disabled: !!r.already_in_library, onclick: async () => {
        add.disabled = true;
        try {
          const res = await api.post("/api/hf/add", { repo: r.repo });
          toast(`${res.name} added to your Library. Press Install to download it.`, "success", 6000);
          dlg.close(); onAdded(res.id);
        } catch (e) { toast(e.message, "danger", 9000); add.disabled = false; }
      } }, icon("plus"), "Add to Library");
      put(body, 
        h("div", { class: "row-between" }, h("h4", { class: "mono hf-name" }, r.repo), h("div", { class: "row-8" }, kindBadge(rc.kind),
          r.gated ? statusBadge("warning", "Gated", "alert") : null)),
        h("div", { class: "muted", style: { fontSize: "var(--text-sm)" } }, `${rc.version} · pipeline ${r.pipeline}`),
        h("div", { class: "stats" }, stat("Download", `${rc.download_gb} GB`, saved), stat("VRAM (est.)", `~${rc.vram_gb} GB`),
          stat("Default size", `${rc.defaults.width} × ${rc.defaults.height}`), stat("Steps", String(rc.defaults.steps)), stat("License", rc.license)),
        r.already_in_library ? h("div", { class: "notice notice-info" }, icon("info"), `Already in your Library as ${r.existing_name}.`) : null,
        ...r.warnings.filter((w) => !w.startsWith("Gated model:") && !(r.gated && w.startsWith("Could not read model_index.json"))).map((w) => h("div", { class: "notice notice-warning" }, icon("alert"), w)),
        r.gated ? h("div", { class: "notice notice-warning" }, icon("alert"),
          h("div", {}, "This is a gated model. Accept its terms on ", h("a", { href: `https://huggingface.co/${r.repo}`, target: "_blank", rel: "noopener" }, "its Hugging Face page"),
            ", and save an access token in Settings before installing.",
            r.needs_token ? " No token is saved yet, so the size and VRAM figures here are rougher estimates." : null)) : null,
        h("div", { class: "row-8" }, add, back));
    } catch (e) { fail(e.message); }
  }

  function submit() {
    const t = input.value.trim();
    if (!t) { search(); return; }
    if (looksLikeRepo(t)) inspect(t); else search();
  }

  drawMode();
  drawKindSeg();
  document.body.append(dlg);
  dlg.showModal();
  input.focus();
  search();
}
