// "Add from Hugging Face": search, inspect a repo, add it to the Library as a recipe.
import { h, icon, clear, toast, fmtCount, kindBadge, statusBadge } from "../ui.js";
import { api } from "../api.js";

// Null-safe append: native Element.append(null) would write the text "null".
const put = (el, ...items) => el.append(...items.flat(Infinity).filter((x) => x !== null && x !== undefined && x !== false));

const looksLikeRepo = (t) => /^https?:\/\//.test(t) || /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/.test(t);

const stat = (label, value, sub) => h("div", { class: "stat" },
  h("span", { class: "stat-label" }, label), h("span", { class: "stat-value" }, value), sub ? h("span", { class: "stat-label" }, sub) : null);

export function openAddFromHF({ onAdded }) {
  const input = h("input", { id: "hf-q", class: "input", type: "text", spellcheck: "false", autocomplete: "off",
    placeholder: "Paste a link or org/name, or search (e.g. flux, sdxl, wan)" });
  const go = h("button", { class: "btn btn-primary", type: "submit" }, icon("search"), "Find");
  let kind = "all";
  const kindSeg = h("div", { class: "seg", role: "group", "aria-label": "Model type" });
  const body = h("div", { class: "hf-body stack-12", "aria-live": "polite" });
  const dlg = h("dialog", { class: "dialog dialog-wide", "aria-labelledby": "hf-title" },
    h("div", { class: "stack-12" },
      h("h3", { id: "hf-title" }, "Add a model from Hugging Face"),
      h("p", { class: "muted" }, "Reel runs diffusers-format text-to-image and text-to-video models. It picks one weight format per part, so the download is far smaller than the repo."),
      h("form", { class: "row-8 hf-search", onsubmit: (e) => { e.preventDefault(); submit(); } },
        h("div", { class: "grow" }, h("label", { class: "sr-only", for: "hf-q" }, "Link, name or search"), input), go),
      h("div", {}, kindSeg), body,
      h("div", { class: "dialog-actions" }, h("button", { class: "btn", type: "button", onclick: () => dlg.close() }, "Close"))));
  dlg.addEventListener("close", () => dlg.remove());

  function drawKindSeg() {
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
      const { results } = await api.get(`/api/hf/search?kind=${kind}&q=${encodeURIComponent(input.value.trim())}`);
      clear(body);
      if (!results.length) { put(body, h("div", { class: "empty" }, "Nothing found. Try another search, or paste a model link.")); return; }
      put(body, h("div", { class: "muted", style: { fontSize: "var(--text-sm)" } }, input.value.trim() ? "Results, most downloaded first" : "Most downloaded"),
        h("ul", { class: "hf-results" }, results.map((r) => h("li", { class: "hf-row" },
          h("div", { class: "grow" },
            h("div", { class: "mono hf-name" }, r.id),
            h("div", { class: "muted hf-sub" }, `${fmtCount(r.downloads)} downloads · ${fmtCount(r.likes)} likes`)),
          kindBadge(r.kind), r.gated ? statusBadge("warning", "Gated", "alert") : null,
          h("button", { class: "btn", type: "button", onclick: () => inspect(r.id) }, "Inspect")))));
    } catch (e) { fail(e.message); }
  }

  async function inspect(repo) {
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

  drawKindSeg();
  document.body.append(dlg);
  dlg.showModal();
  input.focus();
  search();
}
