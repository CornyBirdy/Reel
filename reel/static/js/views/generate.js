import { h, icon, clear, toast, fmtDuration, fmtDate, progressBar, statusBadge } from "../ui.js";
import { api } from "../api.js";

const FORM_KEY = "reel.generate.form";
const PHASES = {
  video: [["loading", "Load model"], ["denoising", "Generate frames"], ["decoding", "Decode video"], ["encoding", "Write mp4"]],
  image: [["loading", "Load model"], ["denoising", "Generate image"], ["decoding", "Decode image"], ["encoding", "Save PNG"]],
};
const KIND_LABEL = { image: "Image", video: "Video" };
const isImageUrl = (url) => /\.png$/i.test(url);

const loadForm = () => { try { return JSON.parse(localStorage.getItem(FORM_KEY)) || {}; } catch { return {}; } };
const saveForm = (v) => { try { localStorage.setItem(FORM_KEY, JSON.stringify(v)); } catch { /* ignore */ } };

const snapFrames = (v, fr) => {
  const n = Math.round((v - fr.offset) / fr.step) * fr.step + fr.offset;
  return Math.max(fr.min, Math.min(fr.max, n));
};

function field(label, control, hint, id) {
  return h("div", { class: "field" }, h("label", { for: id }, label), control, hint ? h("span", { class: "field-hint" }, hint) : null);
}

export async function mount(root) {
  let models = [];       // every installed model
  let kind = "video";    // which kind the form is showing
  let histFilter = "all";
  let alive = true;
  let timer = null;
  let activeJob = null;
  let shownClipId = null;
  let history = { clips: [], total: 0 };
  const saved = loadForm();
  const q = new URLSearchParams(location.hash.split("?")[1] || "");

  // ---- layout ----
  const formCard = h("section", { class: "card stack-16 gen-form" });
  const outputCard = h("section", { class: "card stack-16 gen-output", "aria-live": "polite" });
  const historyEl = h("section", { class: "stack-16" });
  root.append(h("div", { class: "view" },
    h("div", { class: "page-head" }, h("h1", {}, "Generate"), h("p", {}, "Describe what you want, pick an installed model, and render it on this PC.")),
    h("div", { class: "gen-layout" }, formCard, outputCard), historyEl));

  // ---- data ----
  async function loadModels() {
    const d = await api.get("/api/models");
    models = d.models.filter((m) => m.installed);
  }
  const modelById = (id) => models.find((m) => m.id === id);
  const ofKind = () => models.filter((m) => m.kind === kind);
  let wantedModel = q.get("model") || saved.model_id;

  function kindSwitch() {
    const seg = h("div", { class: "seg seg-fill", role: "group", "aria-label": "What to generate" });
    for (const [v, label, ic] of [["image", "Image", "image"], ["video", "Video", "generate"]]) {
      seg.append(h("button", { type: "button", "aria-pressed": String(kind === v), disabled: activeRunning() || undefined,
        onclick: () => { if (kind === v) return; kind = v; wantedModel = null; persist(); buildForm(); renderOutput(); } }, icon(ic), label));
    }
    return seg;
  }

  // ---- form ----
  const f = {}; // element refs
  let model = null;

  function buildForm() {
    clear(formCard);
    formCard.append(kindSwitch());
    const list = ofKind();
    if (!list.length) {
      model = null;
      formCard.append(h("div", { class: "empty" }, icon(kind === "image" ? "image" : "generate"),
        h("div", {}, h("strong", {}, `No ${kind} models installed yet`), h("div", {}, `Install one from the Library, or add one from Hugging Face.`)),
        h("a", { class: "btn btn-primary", href: "#/library" }, icon("library"), "Open Library")));
      return;
    }
    model = list.find((m) => m.id === wantedModel) || list[0];

    f.model = h("select", { id: "g-model", class: "select", onchange: () => { model = modelById(f.model.value); applyModel(model, true); persist(); } },
      list.map((m) => h("option", { value: m.id, selected: m.id === model.id }, `${m.name} ${m.version || ""}`.trim())));
    f.prompt = h("textarea", { id: "g-prompt", class: "textarea", rows: 4, maxlength: 4000, oninput: persist,
      placeholder: kind === "image" ? "A watercolor fox in a misty pine forest, soft morning light, detailed fur…"
                                    : "A slow dolly shot through a rain-soaked neon alley at night, reflections on the wet street…" });
    f.negative = h("textarea", { id: "g-neg", class: "textarea", rows: 2, oninput: persist });
    f.res = h("select", { id: "g-res", class: "select", onchange: () => { persist(); validate(); } });
    f.frames = h("input", { id: "g-frames", type: "range", oninput: () => { syncFrames(); persist(); validate(); } });
    f.framesOut = h("span", { class: "mono" });
    f.seed = h("input", { id: "g-seed", class: "input input-mono", type: "number", min: 0, max: 4294967295, placeholder: "Random", oninput: persist });
    f.steps = h("input", { id: "g-steps", class: "input", type: "number", oninput: persist });
    f.guidance = h("input", { id: "g-guidance", class: "input", type: "number", step: 0.1, min: 0, max: 20, oninput: persist });
    f.offload = h("select", { id: "g-offload", class: "select", onchange: persist });
    f.quant = h("select", { id: "g-quant", class: "select", onchange: persist });
    f.tiling = h("input", { id: "g-tiling", type: "checkbox", onchange: persist });
    f.caps = h("div", { class: "field-hint" });
    f.warn = h("div", { class: "stack-8" });
    f.go = h("button", { class: "btn btn-primary grow", type: "button", onclick: () => start() }, icon("play"), "Generate");
    f.stop = h("button", { class: "btn", type: "button", hidden: true, onclick: () => api.post("/api/generate/cancel").catch((e) => toast(e.message, "danger")) }, icon("stop"), "Cancel");

    formCard.append(
      field("Model", f.model, null, "g-model"),
      field("Prompt", f.prompt, "Describe subject, action, camera and lighting. Longer, concrete prompts work best.", "g-prompt"),
      field("Negative prompt", f.negative, null, "g-neg"),
      h("div", { class: kind === "video" ? "grid-2" : "" },
        field("Resolution", f.res, null, "g-res"),
        kind === "video" ? h("div", { class: "field" },
          h("label", { for: "g-frames" }, "Frames: ", f.framesOut), f.frames,
          h("span", { class: "field-hint" }, `${model.limits.fps} fps`)) : null),
      h("div", { class: "grid-3" },
        field("Seed", h("div", { class: "row-8" },
          h("div", { class: "grow" }, f.seed),
          h("button", { class: "btn btn-icon", type: "button", title: "Random seed", "aria-label": "Random seed", onclick: () => { f.seed.value = Math.floor(Math.random() * 4294967295); persist(); } }, icon("dice"))), null, "g-seed"),
        field("Steps", f.steps, null, "g-steps"),
        field("Guidance", f.guidance, null, "g-guidance")),
      h("details", { class: "memory" },
        h("summary", {}, "Memory options (VRAM saving)"),
        h("div", { class: "stack-12" },
          h("div", { class: "grid-2" }, field("Offloading", f.offload, null, "g-offload"), field("Quantization", f.quant, null, "g-quant")),
          h("label", { class: "check" }, f.tiling, "Tiled VAE decode"),
          h("div", { class: "field-hint", id: "mem-note" }))),
      f.caps, f.warn,
      h("div", { class: "row-8" }, f.go, f.stop));

    applyModel(model, false);
    for (const [k, v] of Object.entries(saved)) {
      if (saved.model_id !== model.id) break;
      if (k === "prompt") f.prompt.value = v;
      else if (k === "negative_prompt") f.negative.value = v;
      else if (k === "seed" && v !== "") f.seed.value = v;
      else if (k === "res" && [...f.res.options].some((o) => o.value === v)) f.res.value = v;
      else if (k === "frames" && kind === "video") { f.frames.value = snapFrames(Number(v), model.limits.frames); }
      else if (k === "steps") f.steps.value = v;
      else if (k === "guidance") f.guidance.value = v;
      else if (k === "offload" && [...f.offload.options].some((o) => o.value === v)) f.offload.value = v;
      else if (k === "quantization" && [...f.quant.options].some((o) => o.value === v)) f.quant.value = v;
      else if (k === "vae_tiling") f.tiling.checked = !!v;
    }
    syncFrames(); validate();
  }

  function applyModel(m, reset) {
    const lim = m.limits, d = m.defaults, mem = m.memory || {};
    clear(f.res);
    for (const p of lim.presets) f.res.append(h("option", { value: `${p.w}x${p.h}` }, `${p.w} × ${p.h}`));
    f.res.value = `${d.width}x${d.height}`;
    if (kind === "video") {
      f.frames.min = lim.frames.min; f.frames.max = lim.frames.max; f.frames.step = lim.frames.step;
      f.frames.value = lim.frames.default;
    }
    f.steps.min = lim.steps?.min ?? 1; f.steps.max = lim.steps?.max ?? 100; f.steps.value = d.steps;
    f.guidance.value = d.guidance;
    f.negative.value = d.negative_prompt || "";
    clear(f.offload);
    for (const o of mem.offload_options || ["model"]) f.offload.append(h("option", { value: o }, { none: "None (fastest, most VRAM)", model: "Model offload", sequential: "Sequential (slowest, least VRAM)" }[o] || o));
    f.offload.value = d.offload;
    clear(f.quant);
    for (const o of mem.quantization_options || ["none"]) f.quant.append(h("option", { value: o }, { none: "Off (bf16)", int8: "8-bit", nf4: "4-bit NF4" }[o] || o));
    f.quant.value = d.quantization;
    f.tiling.checked = d.vae_tiling;
    const note = document.getElementById("mem-note");
    if (note) note.textContent = mem.notes || "";
    if (kind === "video") {
      const budget = lim.max_area_frames ? " The width × height × frames total is also capped." : "";
      f.caps.textContent = `${m.name} is capped at ${lim.max_width} × ${lim.max_height} and ${lim.frames.max} frames to fit a 16 GB GPU.${budget}`;
    } else {
      f.caps.textContent = `${m.name} is capped at ${lim.max_width} × ${lim.max_height} (and ${((lim.max_area || lim.max_width * lim.max_height) / 1e6).toFixed(1)} megapixels) to fit a 16 GB GPU.`;
    }
    syncFrames(); validate();
  }

  function syncFrames() {
    if (!model || kind !== "video") return;
    const fr = model.limits.frames;
    const n = snapFrames(Number(f.frames.value), fr);
    f.frames.value = n;
    f.framesOut.textContent = `${n} · ${(n / model.limits.fps).toFixed(1)} s`;
  }

  function validate() {
    clear(f.warn);
    if (!model) return true;
    const [w, hgt] = f.res.value.split("x").map(Number);
    let ok = true;
    if (kind === "video") {
      const frames = Number(f.frames.value);
      const budget = model.limits.max_area_frames;
      if (budget && w * hgt * frames > budget) {
        ok = false;
        f.warn.append(h("div", { class: "notice notice-warning" }, icon("alert"), `${w} × ${hgt} with ${frames} frames is over the 16 GB budget. Lower the resolution or the frame count.`));
      }
    } else if (model.limits.max_area && w * hgt > model.limits.max_area) {
      ok = false;
      f.warn.append(h("div", { class: "notice notice-warning" }, icon("alert"), `${w} × ${hgt} is over the 16 GB size budget. Lower the resolution.`));
    }
    if (!activeRunning()) f.go.disabled = !ok;
    return ok;
  }

  function readForm() {
    const [w, hgt] = f.res.value.split("x").map(Number);
    return {
      model_id: model.id, prompt: f.prompt.value, negative_prompt: f.negative.value,
      width: w, height: hgt, frames: kind === "video" ? Number(f.frames.value) : undefined, steps: Number(f.steps.value),
      guidance: Number(f.guidance.value), seed: f.seed.value === "" ? null : Number(f.seed.value),
      offload: f.offload.value, quantization: f.quant.value, vae_tiling: f.tiling.checked,
    };
  }

  function persist() {
    if (!model) return;
    saveForm({
      kind, model_id: model.id, prompt: f.prompt.value, negative_prompt: f.negative.value, seed: f.seed.value,
      res: f.res.value, frames: f.frames.value, steps: f.steps.value, guidance: f.guidance.value,
      offload: f.offload.value, quantization: f.quant.value, vae_tiling: f.tiling.checked,
    });
  }

  function loadSettings(clip) {
    const m = modelById(clip.model_id);
    if (!m) { toast(`${clip.model_name} is not installed any more.`, "warning"); return; }
    const p = clip.params;
    if (m.kind !== kind || !model || model.id !== m.id) { kind = m.kind; wantedModel = m.id; buildForm(); }
    f.prompt.value = clip.prompt; f.negative.value = p.negative_prompt || "";
    f.res.value = `${p.width}x${p.height}`; if (kind === "video") f.frames.value = p.frames; f.steps.value = p.steps;
    f.guidance.value = p.guidance; f.seed.value = p.seed;
    f.offload.value = p.offload; f.quant.value = p.quantization; f.tiling.checked = p.vae_tiling;
    syncFrames(); validate(); persist();
    formCard.scrollIntoView({ behavior: "smooth", block: "start" });
    toast("Settings loaded into the form.", "info", 2500);
  }

  async function start(body) {
    const req = body || readForm();
    if (!model) return;
    if (!body && !validate()) return;
    if (!body) { if (!req.prompt.trim()) { toast("Write a prompt first.", "warning"); f.prompt.focus(); return; } }
    try {
      activeJob = await api.post("/api/generate", req);
      shownClipId = null;
      renderOutput();
      poll();
    } catch (e) { toast(e.message, "danger", 9000); }
  }

  async function rerun(clip) {
    if (!modelById(clip.model_id)) { toast(`${clip.model_name} is not installed any more.`, "warning"); return; }
    await start({ model_id: clip.model_id, ...clip.params });
  }

  const activeRunning = () => !!activeJob && activeJob.status === "running";

  // ---- output panel ----
  function renderOutput() {
    clear(outputCard);
    const running = activeRunning();
    if (f.go) { f.go.hidden = running; f.stop.hidden = !running; f.go.disabled = running; }
    const job = activeJob;

    if (running) {
      const phases = PHASES[job.kind || "video"];
      const order = phases.map((p) => p[0]);
      const idx = Math.max(0, order.indexOf(job.phase === "encoding-prompt" ? "loading" : job.phase));
      const frac = job.phase === "denoising" || job.phase === "decoding"
        ? 0.1 + 0.8 * (job.step / Math.max(job.total, 1)) + (job.phase === "decoding" ? 0.05 : 0)
        : job.phase === "encoding" ? 0.97 : 0.05;
      outputCard.append(
        h("div", { class: "row-between" }, h("h3", {}, `Generating with ${job.model_name}`), statusBadge("info", "Running", "play")),
        h("ol", { class: "steps", "aria-label": "Generation steps" },
          phases.map(([id, label], i) => h("li", { class: "step", "data-state": i < idx ? "done" : i === idx ? "active" : "pending", "aria-current": i === idx ? "step" : undefined },
            icon(i < idx ? "check" : i === idx ? "play" : "clock"), h("span", {}, label)))),
        progressBar(frac, { indeterminate: job.phase === "loading" && job.step === 0, label: "Generation progress" }),
        h("div", { class: "progress-meta" },
          h("span", {}, job.phase === "denoising" ? `Step ${job.step} of ${job.total}` : job.message),
          h("span", {}, [`${fmtDuration(job.elapsed)} elapsed`, job.eta ? `~${fmtDuration(job.eta)} left` : null].filter(Boolean).join(" · "))),
        h("p", { class: "muted" }, job.params.prompt));
      return;
    }

    if (job && job.status === "failed") {
      outputCard.append(h("div", { class: "notice notice-danger", role: "alert" }, icon("error"),
        h("div", { class: "stack-8" }, h("strong", {}, "Generation failed"), h("div", { class: "mono" }, job.error || "Unknown error"),
          job.error_kind === "oom" ? h("div", {}, "Out of VRAM. Lower the resolution (or frame count), set offloading to Sequential, or use NF4 quantization.") : null)));
    } else if (job && job.status === "cancelled") {
      outputCard.append(h("div", { class: "notice notice-info" }, icon("info"), "Cancelled."));
    }

    const clip = (job && job.status === "done" && job.clip) || history.clips.find((c) => c.id === shownClipId);
    if (clip) { showClip(clip); return; }
    outputCard.append(h("div", { class: "empty" }, icon(kind === "image" ? "image" : "generate"),
      h("div", {}, h("strong", {}, kind === "image" ? "Your image appears here" : "Your clip appears here"), h("div", {}, "Pick a model, write a prompt and press Generate."))));
  }

  function showClip(clip) {
    shownClipId = clip.id;
    const p = clip.params;
    const image = isImageUrl(clip.url);
    const media = image
      ? h("img", { class: "player", src: clip.url, alt: clip.prompt })
      : h("video", { class: "player", controls: true, loop: true, muted: true, autoplay: true, playsinline: true, src: clip.url });
    outputCard.append(media,
      h("p", {}, clip.prompt),
      h("div", { class: "stats" },
        stat("Model", clip.model_name), stat("Size", `${p.width} × ${p.height}`),
        image ? null : stat("Frames", `${p.frames} @ ${p.fps} fps`),
        stat("Seed", String(p.seed)), stat("Time", fmtDuration(clip.seconds)),
        clip.peak_vram_gb ? stat("Peak VRAM", `${clip.peak_vram_gb} GB`) : null),
      h("div", { class: "row-8" },
        h("a", { class: "btn", href: clip.url, download: `reel-${clip.id}.${image ? "png" : "mp4"}` }, icon("download"), "Download"),
        h("button", { class: "btn", type: "button", onclick: () => loadSettings(clip) }, icon("load"), "Reuse settings"),
        h("button", { class: "btn", type: "button", disabled: activeRunning(), onclick: () => rerun(clip) }, icon("rerun"), "Re-run")));
    renderHistory();
  }

  const stat = (label, value) => h("div", { class: "stat" }, h("span", { class: "stat-label" }, label), h("span", { class: "stat-value" }, value));

  // ---- history ----
  async function loadHistory() {
    history = await api.get("/api/history?limit=60");
    renderHistory();
  }

  function renderHistory() {
    clear(historyEl);
    const seg = h("div", { class: "seg", role: "group", "aria-label": "Filter history" });
    for (const [v, label] of [["all", "All"], ["image", "Images"], ["video", "Videos"]]) {
      seg.append(h("button", { type: "button", "aria-pressed": String(histFilter === v), onclick: () => { histFilter = v; renderHistory(); } }, label));
    }
    const shown = history.clips.filter((c) => histFilter === "all" || c.kind === histFilter);
    historyEl.append(h("div", { class: "row-between" }, h("h2", {}, "History"),
      h("div", { class: "row-12" }, h("span", { class: "muted" }, `${shown.length} item${shown.length === 1 ? "" : "s"}`), seg)));
    if (!shown.length) {
      historyEl.append(h("div", { class: "empty" }, icon("clock"), h("div", {}, "Finished images and clips are saved here with their prompts, so you can re-run them.")));
      return;
    }
    historyEl.append(h("div", { class: "history-grid" }, shown.map(clipCard)));
  }

  function clipCard(clip) {
    const p = clip.params;
    const current = clip.id === shownClipId;
    return h("article", { class: "item clip", "aria-current": current ? "true" : undefined },
      h("button", { class: "clip-thumb", type: "button", "aria-label": `Show: ${clip.prompt.slice(0, 60)}`, onclick: () => { activeJob = null; shownClipId = clip.id; renderOutput(); outputCard.scrollIntoView({ behavior: "smooth", block: "nearest" }); } },
        isImageUrl(clip.url)
          ? h("img", { src: clip.url, alt: "", loading: "lazy" })
          : h("video", { src: `${clip.url}#t=0.1`, preload: "metadata", muted: true, playsinline: true, tabindex: -1 })),
      h("div", { class: "clip-body stack-8" },
        h("div", { class: "clip-prompt", title: clip.prompt }, clip.prompt),
        h("div", { class: "muted clip-meta" }, `${clip.model_name} · ${p.width}×${p.height}${isImageUrl(clip.url) ? "" : ` · ${p.frames}f`} · seed ${p.seed}`),
        h("div", { class: "muted clip-meta" }, `${fmtDate(clip.created)} · ${fmtDuration(clip.seconds)}`, current ? h("strong", { class: "now" }, " · Showing") : null),
        h("div", { class: "row-8" },
          h("button", { class: "btn btn-ghost btn-icon", type: "button", title: "Reuse settings", "aria-label": "Reuse settings", onclick: () => loadSettings(clip) }, icon("load")),
          h("button", { class: "btn btn-ghost btn-icon", type: "button", title: "Re-run with the same seed", "aria-label": "Re-run", disabled: activeRunning(), onclick: () => rerun(clip) }, icon("rerun")),
          h("a", { class: "btn btn-ghost btn-icon", href: clip.url, download: `reel-${clip.id}.${isImageUrl(clip.url) ? "png" : "mp4"}`, title: "Download", "aria-label": "Download" }, icon("download")),
          h("button", { class: "btn btn-ghost btn-icon", type: "button", title: "Delete", "aria-label": "Delete clip", onclick: async () => {
            try { await api.del(`/api/history/${clip.id}`); if (shownClipId === clip.id) { shownClipId = null; if (activeJob?.clip?.id === clip.id) activeJob = null; } await loadHistory(); renderOutput(); }
            catch (e) { toast(e.message, "danger"); }
          } }, icon("trash")))));
  }

  // ---- polling ----
  async function poll() {
    clearTimeout(timer);
    if (!alive) return;
    try {
      const { job } = await api.get("/api/generate/current");
      if (job && (!activeJob || activeJob.id === job.id || job.status === "running")) {
        const wasRunning = activeJob && activeJob.status === "running";
        activeJob = job;
        if (job.status !== "running" && wasRunning) {
          await loadHistory();
          if (job.status === "done") toast(job.kind === "image" ? "Image finished." : "Clip finished.", "success", 3000);
        }
        if (job.status === "done" && job.clip) shownClipId = job.clip.id;
        renderOutput();
      }
    } catch { /* server busy: try again */ }
    if (alive && activeRunning()) timer = setTimeout(poll, 1000);
  }

  // ---- go ----
  await loadModels();
  {
    const byQuery = models.find((m) => m.id === q.get("model"));
    kind = (byQuery && byQuery.kind) || (["image", "video"].includes(saved.kind) ? saved.kind : null)
      || (models.length ? models[0].kind : "image");
    if (!models.some((m) => m.kind === kind) && models.length) kind = models[0].kind;
  }
  buildForm();
  await loadHistory();
  const { job } = await api.get("/api/generate/current").catch(() => ({ job: null }));
  if (job && (job.status === "running" || job.status === "failed")) activeJob = job;
  else if (job && job.status === "done" && job.clip && history.clips.some((c) => c.id === job.clip.id)) { activeJob = job; shownClipId = job.clip.id; }
  else if (history.clips.length) shownClipId = history.clips[0].id;
  renderOutput();
  if (activeRunning()) poll();
  return { unmount() { alive = false; clearTimeout(timer); } };
}
