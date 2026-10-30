import { h, icon, clear, toast, fmtDuration, fmtDate, progressBar, statusBadge, viewLog } from "../ui.js";
import { api } from "../api.js";
import { KEEPS_MEMORY, crashNotice, holdsText, holdsTitle, idleText, loadButton, stateChip, stateOf, watchWorker } from "../worker.js";

const FORM_KEY = "reel.generate.form";
const PHASES = {
  video: [["loading", "Load model"], ["denoising", "Generate frames"], ["decoding", "Decode video"], ["encoding", "Write mp4"]],
  image: [["loading", "Load model"], ["denoising", "Generate image"], ["decoding", "Decode image"], ["encoding", "Save PNG"]],
  upscale: [["loading", "Load upscaler"], ["denoising", "Upscale"], ["encoding", "Save"]],
};
const KIND_LABEL = { image: "Image", video: "Video" };
const isImageUrl = (url) => /\.png$/i.test(url);
// Element.append turns null into the text "null"; optional children are dropped here instead.
const put = (el, ...kids) => el.append(...kids.filter((k) => k !== null && k !== undefined && k !== false));

const UNLOCK_KEY = "reel.generate.unlock";   // { model_id: true } - the toggle is remembered per model
const loadUnlock = () => { try { return JSON.parse(localStorage.getItem(UNLOCK_KEY)) || {}; } catch { return {}; } };
const saveUnlock = (v) => { try { localStorage.setItem(UNLOCK_KEY, JSON.stringify(v)); } catch { /* ignore */ } };
const ATTN_LABEL = {
  auto: "Auto (per the diffusers docs for this GPU)", native: "Native (PyTorch)", sage_hub: "SageAttention (sage_hub)",
  flash_hub: "FlashAttention-2 (flash_hub)", _flash_3_hub: "FlashAttention-3 (_flash_3_hub, H100)", sage_blackwell_hub: "SageAttention 3 (sage_blackwell_hub, RTX 50)",
};
const STATUS_KIND = { ok: "success", risky: "warning", fail: "danger", unknown: "neutral" };
const STATUS_LABEL = { ok: "OK", risky: "Risky", fail: "Likely to fail", unknown: "Unknown" };

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
  let models = [];       // every installed generator (image or video)
  let upscalers = [];    // installed upscalers
  let kind = "video";    // which kind the form is showing
  let histFilter = "all";
  let alive = true;
  let timer = null;
  let activeJob = null;
  let shownClipId = null;
  let history = { clips: [], total: 0 };
  const saved = loadForm();
  const unlockMap = loadUnlock();
  let estTimer = null, estSeq = 0;
  let worker = null;     // what is loaded right now (GET /api/worker)
  let watcher = null;
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
    models = d.models.filter((m) => m.installed && m.kind !== "upscale");
    upscalers = d.models.filter((m) => m.installed && m.kind === "upscale");
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

    f.worker = h("div", { class: "stack-8 worker-row" });
    f.model = h("select", { id: "g-model", class: "select", onchange: () => { model = modelById(f.model.value); applyModel(model, true); persist(); renderWorker(); } },
      list.map((m) => h("option", { value: m.id, selected: m.id === model.id },
        `${m.name} ${m.version || ""}`.trim() + (m.native_label ? ` · native ${m.native_label}` : ""))));
    f.prompt = h("textarea", { id: "g-prompt", class: "textarea", rows: 4, maxlength: 4000, oninput: persist,
      placeholder: kind === "image" ? "A watercolor fox in a misty pine forest, soft morning light, detailed fur…"
                                    : "A slow dolly shot through a rain-soaked neon alley at night, reflections on the wet street…" });
    f.negative = h("textarea", { id: "g-neg", class: "textarea", rows: 2, oninput: persist });
    f.res = h("select", { id: "g-res", class: "select", onchange: () => { syncCustom(); persist(); validate(); refreshEstimate(); } });
    f.cw = h("input", { id: "g-cw", class: "input", type: "number", oninput: () => { persist(); validate(); refreshEstimate(); } });
    f.ch = h("input", { id: "g-ch", class: "input", type: "number", oninput: () => { persist(); validate(); refreshEstimate(); } });
    f.custom = h("div", { class: "grid-2", hidden: true }, field("Width", f.cw, null, "g-cw"), field("Height", f.ch, null, "g-ch"));
    f.frames = h("input", { id: "g-frames", type: "range", oninput: () => { syncFrames(); persist(); validate(); refreshEstimate(); } });
    f.framesOut = h("span", { class: "mono" });
    f.framesHint = h("span", { class: "field-hint" });
    f.fps = h("input", { id: "g-fps", class: "input", type: "number", min: 1, max: 60, oninput: () => { syncFrames(); persist(); } });
    f.unlock = h("input", { id: "g-unlock", type: "checkbox", onchange: () => {
      unlockMap[model.id] = f.unlock.checked; saveUnlock(unlockMap);
      applyLimits(); persist(); validate(); refreshEstimate();
    } });
    f.unlockNote = h("span", { class: "field-hint" });
    f.est = h("div", { class: "stack-8 mem-est", "aria-live": "polite" });
    f.seed = h("input", { id: "g-seed", class: "input input-mono", type: "number", min: 0, max: 4294967295, placeholder: "Random", oninput: persist });
    f.steps = h("input", { id: "g-steps", class: "input", type: "number", oninput: persist });
    f.guidance = h("input", { id: "g-guidance", class: "input", type: "number", step: 0.1, min: 0, max: 20, oninput: persist });
    f.offload = h("select", { id: "g-offload", class: "select", onchange: () => { persist(); refreshEstimate(); renderWorker(); } });
    f.quant = h("select", { id: "g-quant", class: "select", onchange: () => { persist(); refreshEstimate(); renderWorker(); } });
    f.tiling = h("input", { id: "g-tiling", type: "checkbox", onchange: () => { persist(); refreshEstimate(); renderWorker(); } });
    f.attn = h("select", { id: "g-attn", class: "select", onchange: () => { persist(); renderWorker(); } });
    f.attnHint = h("span", { class: "field-hint" });
    f.attnField = h("div", { class: "field" }, h("label", { for: "g-attn" }, "Attention backend"), f.attn, f.attnHint);
    f.caps = h("div", { class: "field-hint" });
    f.warn = h("div", { class: "stack-8" });
    f.fit = h("div", { class: "stack-8" });
    f.count = h("select", { id: "g-count", class: "select count-select", "aria-label": "How many to generate", onchange: () => { syncGoLabel(); persist(); } });
    f.goLabel = h("span", {}, "Generate");
    f.go = h("button", { class: "btn btn-primary grow", type: "button", onclick: () => start() }, icon("play"), f.goLabel);
    f.stop = h("button", { class: "btn", type: "button", hidden: true, onclick: () => api.post("/api/generate/cancel").catch((e) => toast(e.message, "danger")) }, icon("stop"), "Cancel");

    formCard.append(
      field("Model", f.model, null, "g-model"),
      f.worker,
      field("Prompt", f.prompt, "Describe subject, action, camera and lighting. Longer, concrete prompts work best.", "g-prompt"),
      field("Negative prompt", f.negative, null, "g-neg"),
      h("div", { class: kind === "video" ? "grid-2" : "" },
        field("Resolution", f.res, null, "g-res"),
        kind === "video" ? h("div", { class: "field" },
          h("label", { for: "g-frames" }, "Frames: ", f.framesOut), f.frames, f.framesHint) : null),
      f.custom,
      h("div", { class: "field" }, h("label", { class: "check" }, f.unlock, "Unlock limits (advanced)"), f.unlockNote),
      h("div", { class: kind === "video" ? "grid-2" : "grid-3" },
        field("Seed", h("div", { class: "row-8" },
          h("div", { class: "grow" }, f.seed),
          h("button", { class: "btn btn-icon", type: "button", title: "Random seed", "aria-label": "Random seed", onclick: () => { f.seed.value = Math.floor(Math.random() * 4294967295); persist(); } }, icon("dice"))), null, "g-seed"),
        field("Steps", f.steps, null, "g-steps"),
        field("Guidance", f.guidance, null, "g-guidance"),
        kind === "video" ? field("Frames per second", f.fps, null, "g-fps") : null),
      h("details", { class: "memory" },
        h("summary", {}, "Memory options (VRAM saving)"),
        h("div", { class: "stack-12" },
          h("div", { class: "grid-2" }, field("Offloading", f.offload, null, "g-offload"), field("Quantization", f.quant, null, "g-quant")),
          f.attnField,
          h("label", { class: "check" }, f.tiling, "Tiled VAE decode"),
          h("div", { class: "field-hint", id: "mem-note" }))),
      f.est, f.caps, f.fit, f.warn,
      h("div", { class: "row-8 action-row" },
        h("div", { class: "count-pick" }, h("label", { for: "g-count", class: "count-label" }, "Count"), f.count), f.go, f.stop),
      h("div", { class: "field-hint" }, "Several at once: the model loads once, then each one uses its own seed (seed, seed+1, …). Every finished one is saved as it completes."));

    applyModel(model, false);
    for (const [k, v] of Object.entries(saved)) {
      if (saved.model_id !== model.id) break;
      if (k === "prompt") f.prompt.value = v;
      else if (k === "negative_prompt") f.negative.value = v;
      else if (k === "seed" && v !== "") f.seed.value = v;
      else if (k === "res" && [...f.res.options].some((o) => o.value === v)) f.res.value = v;
      else if (k === "cw" && v) f.cw.value = v;
      else if (k === "ch" && v) f.ch.value = v;
      else if (k === "fps" && kind === "video" && v) f.fps.value = v;
      else if (k === "frames" && kind === "video") { f.frames.value = snapFrames(Number(v), model.limits.frames); }
      else if (k === "steps") f.steps.value = v;
      else if (k === "guidance") f.guidance.value = v;
      else if (k === "offload" && [...f.offload.options].some((o) => o.value === v)) f.offload.value = v;
      else if (k === "quantization" && [...f.quant.options].some((o) => o.value === v)) f.quant.value = v;
      else if (k === "vae_tiling") f.tiling.checked = !!v;
      else if (k === "attention_backend" && [...f.attn.options].some((o) => o.value === v)) f.attn.value = v;
      else if (k === "count") fillCount(v);
    }
    syncCustom(); syncFrames(); validate(); refreshEstimate(); renderWorker();
  }

  // The loaded model: state chip, what it holds, Load / Unload, and a heads-up when the form's memory options would force a reload.
  function memorySettings() {
    return { quantization: f.quant.value, offload: f.offload.value, vae_tiling: f.tiling.checked,
             attention_backend: f.attnField.hidden ? undefined : f.attn.value };
  }

  function renderWorker() {
    if (!f.worker || !model) return;
    clear(f.worker);
    const w = worker, st = stateOf(w, model.id);
    const holds = st === "unloaded" ? null : holdsText(w);
    const other = w && w.state !== "unloaded" && w.model_id && w.model_id !== model.id ? modelById(w.model_id) : null;
    f.worker.append(h("div", { class: "row-8" },
      stateChip(st, st === "loading" ? w.message : undefined),
      holds ? h("span", { class: "muted mono", title: holdsTitle }, holds) : null,
      h("span", { class: "grow" }),
      loadButton(model, w, { onDone: () => watcher && watcher.poke(), memory: memorySettings, compact: true })));
    const notes = [];
    if (st === "unloaded" && !other) notes.push("Generate loads it automatically. Loading takes a while the first time; later runs reuse it.");
    if (other) notes.push(`${other.name} is loaded. Loading or generating with this model unloads it first: only one model fits on the GPU at a time.`);
    if (st === "ready" || st === "generating") {
      notes.push(`${KEEPS_MEMORY} ${idleText(w) || ""}`.trim());
      const cur = w.settings || {}, want = memorySettings();
      if (st === "ready" && Object.keys(want).some((k) => want[k] !== undefined && cur[k] !== undefined && cur[k] !== want[k])) {
        notes.push("Your memory options differ from what is loaded, so the next Generate reloads the model.");
      }
    }
    for (const n of notes) f.worker.append(h("div", { class: "field-hint" }, n));
    // A crash of the request that is on screen is already explained in the output panel.
    const explained = activeJob && w && w.last_crash && activeJob.id === w.last_crash.request;
    const crash = explained ? null : crashNotice(w, () => watcher && watcher.poke());
    if (crash && w.last_crash.model_id === model.id) f.worker.append(crash);
  }

  function applyModel(m, reset) {
    const lim = m.limits, d = m.defaults, mem = m.memory || {};
    f.unlock.checked = !!unlockMap[m.id];            // remembered per model
    fillResolutions(`${d.width}x${d.height}`, true);
    if (kind === "video") {
      f.fps.value = d.fps ?? lim.fps;                // the recipe's own frame rate (e.g. 24 for HunyuanVideo 1.5)
      applyFrameRange();
      f.frames.value = lim.frames.default;
    }
    f.steps.min = lim.steps?.min ?? 1; f.steps.max = lim.steps?.max ?? 100; f.steps.value = d.steps;
    f.guidance.value = d.guidance;
    f.negative.value = d.negative_prompt || "";
    clear(f.offload);
    for (const o of mem.offload_options || ["model"]) f.offload.append(h("option", { value: o }, { none: "None (fastest, most VRAM)", model: "Model offload", sequential: "Sequential (slowest, least VRAM)" }[o] || o));
    // Start from what suits this PC's GPU: low-memory settings on a small card, no offloading on a big one.
    const rec = (m.fit && m.fit.recommended) || {};
    const pick = (select, wanted, fallback) => ([...select.options].some((o) => o.value === wanted) ? wanted : fallback);
    f.offload.value = pick(f.offload, rec.offload, d.offload);
    clear(f.quant);
    for (const o of mem.quantization_options || ["none"]) f.quant.append(h("option", { value: o }, { none: "Off (bf16)", int8: "8-bit", nf4: "4-bit NF4" }[o] || o));
    f.quant.value = pick(f.quant, rec.quantization, d.quantization);
    clear(f.fit);
    if (m.fit && ["low_memory", "too_big", "no_gpu"].includes(m.fit.level)) {
      f.fit.append(h("div", { class: `notice ${m.fit.level === "low_memory" ? "notice-info" : "notice-warning"}` }, icon(m.fit.level === "low_memory" ? "info" : "alert"),
        h("div", {}, h("strong", {}, m.fit.label), h("div", {}, m.fit.detail))));
    }
    f.tiling.checked = d.vae_tiling;
    f.attnField.hidden = !mem.attention_options;
    f.attnHint.textContent = mem.attention_note || "How the denoiser computes attention. Native is PyTorch's own and always works.";
    clear(f.attn);
    for (const o of mem.attention_options || []) f.attn.append(h("option", { value: o }, ATTN_LABEL[o] || o));
    if (mem.attention_options) f.attn.value = d.attention_backend || "native";
    fillCount(f.count.value);
    const note = document.getElementById("mem-note");
    if (note) note.textContent = mem.notes || "";
    describeLimits();
    syncCustom(); syncFrames(); validate(); refreshEstimate();
  }

  const isUnlocked = () => !!(model && f.unlock.checked);
  const maxSide = (key) => (isUnlocked() ? model.unlock[`max_${key}`] : model.limits[`max_${key}`]);
  const maxFrames = () => (isUnlocked() ? model.unlock.max_frames : model.limits.frames.max);

  // Safe presets, the model's native sizes, and (when unlocked) native sizes above the safe caps plus a custom size.
  function fillResolutions(want, fresh) {
    const lim = model.limits, nat = model.native?.resolutions || [];
    const key = (p) => `${p.w}x${p.h}`;
    const natKeys = new Set(nat.map(key));
    const fitsSafe = (p) => p.w <= lim.max_width && p.h <= lim.max_height;
    const label = (p, extra) => `${p.w} × ${p.h}${natKeys.has(key(p)) ? " · native" : ""}${extra || ""}`;
    const have = new Set(lim.presets.map(key));
    clear(f.res);
    for (const p of lim.presets) f.res.append(h("option", { value: key(p) }, label(p)));
    for (const p of nat) {
      if (have.has(key(p))) continue;
      if (fitsSafe(p)) f.res.append(h("option", { value: key(p) }, label(p)));
      else if (isUnlocked()) f.res.append(h("option", { value: key(p) }, label(p, " · above safe limits")));
    }
    if (isUnlocked()) f.res.append(h("option", { value: "custom" }, "Custom size…"));
    const ok = [...f.res.options].some((o) => o.value === want);
    f.res.value = ok ? want : `${model.defaults.width}x${model.defaults.height}`;
    if (!ok && !fresh && want === "custom") f.res.value = `${model.defaults.width}x${model.defaults.height}`;
  }

  function applyFrameRange() {
    const fr = model.limits.frames;
    f.frames.min = fr.min; f.frames.max = maxFrames(); f.frames.step = fr.step;
  }

  // The unlock toggle changed: widen or narrow what the form offers, keeping the current choices where they still fit.
  function applyLimits() {
    fillResolutions(f.res.value);
    if (kind === "video") { applyFrameRange(); }
    describeLimits();
    syncCustom(); syncFrames();
  }

  function describeLimits() {
    const m = model, lim = m.limits, nat = m.native, u = m.unlock;
    const dims = (w, hh) => `${w} × ${hh}`;
    if (kind === "video") {
      const budget = lim.max_area_frames ? " The width × height × frames total is also capped." : "";
      f.caps.textContent = `Safe limits for ${m.name}: up to ${dims(lim.max_width, lim.max_height)} and ${lim.frames.max} frames, sized for a 16 GB GPU.${budget}`
        + (nat ? ` Native: ${nat.resolutions.map((r) => dims(r.w, r.h)).slice(0, 3).join(", ")}${nat.frames ? `, ${nat.frames.join(" or ")} frames` : ""}${nat.fps ? ` at ${nat.fps} fps` : ""}.` : "");
      f.framesHint.textContent = nat?.frames?.length ? `Native: ${nat.frames.join(" or ")} frames (${lim.frames.step}n+${lim.frames.offset})` : `${lim.frames.step}n+${lim.frames.offset} frames`;
    } else {
      f.caps.textContent = `Safe limits for ${m.name}: up to ${dims(lim.max_width, lim.max_height)} and ${((lim.max_area || lim.max_width * lim.max_height) / 1e6).toFixed(1)} megapixels, sized for a 16 GB GPU.`;
    }
    const up = kind === "video" ? `${dims(u.max_width, u.max_height)} and ${u.max_frames} frames` : dims(u.max_width, u.max_height);
    f.unlockNote.textContent = isUnlocked()
      ? `Unlocked: up to ${up}, any size in multiples of ${lim.size_multiple}. ${u.note}${u.known ? "" : " (Reel is guessing this ceiling.)"} Too big a run can run out of memory or crash.`
      : `Allow every size${kind === "video" ? " and length" : ""} the model supports (up to ${up}), not only what is sized for a 16 GB GPU.`;
  }

  function syncCustom() {
    const custom = f.res.value === "custom";
    f.custom.hidden = !custom;
    if (custom) {
      const mult = model.limits.size_multiple;
      f.cw.step = f.ch.step = mult; f.cw.min = f.ch.min = 64; f.cw.max = maxSide("width"); f.ch.max = maxSide("height");
      if (!f.cw.value) { f.cw.value = model.defaults.width; f.ch.value = model.defaults.height; }
    }
  }

  const getSize = () => (f.res.value === "custom" ? [Number(f.cw.value), Number(f.ch.value)] : f.res.value.split("x").map(Number));

  function fillCount(keep) {
    const max = (model.limits.max_count ?? (kind === "image" ? 8 : 4));
    clear(f.count);
    for (let n = 1; n <= max; n++) f.count.append(h("option", { value: n }, String(n)));
    f.count.value = String(Math.min(Math.max(Number(keep) || 1, 1), max));
    syncGoLabel();
  }

  function syncGoLabel() {
    const n = Number(f.count.value) || 1;
    f.goLabel.textContent = n > 1 ? `Generate ${n} ${kind === "image" ? "images" : "clips"}` : "Generate";
  }

  function syncFrames() {
    if (!model || kind !== "video") return;
    const fr = model.limits.frames;
    const n = snapFrames(Number(f.frames.value), { ...fr, max: maxFrames() });
    f.frames.value = n;
    const fps = Number(f.fps.value) || model.defaults.fps || model.limits.fps;
    f.framesOut.textContent = `${n} · ${(n / fps).toFixed(1)} s`;
  }

  // Blocks only what the pipeline would reject; going above the safe limits is a warning, never a block.
  function validate() {
    clear(f.warn);
    if (!model) return true;
    const lim = model.limits, mult = lim.size_multiple;
    const [w, hgt] = getSize();
    const unlocked = isUnlocked();
    const warn = (msg, kindName = "warning") => f.warn.append(h("div", { class: `notice notice-${kindName}` }, icon("alert"), msg));
    let ok = true;
    let beyond = false;
    if (!w || !hgt || w % mult || hgt % mult) {
      ok = false;
      warn(`Width and height must be multiples of ${mult}.`);
    } else if (w < 64 || hgt < 64 || w > maxSide("width") || hgt > maxSide("height")) {
      ok = false;
      warn(`Width and height must be between 64 and ${maxSide("width")} × ${maxSide("height")} for ${model.name}.${unlocked ? "" : " Turn on Unlock limits (advanced) to go higher."}`);
    } else {
      if (w > lim.max_width || hgt > lim.max_height) beyond = true;
      if (kind === "video") {
        const frames = Number(f.frames.value);
        const budget = lim.max_area_frames;
        if (frames > lim.frames.max) beyond = true;
        if (budget && w * hgt * frames > budget) {
          if (unlocked) beyond = true;
          else { ok = false; warn(`${w} × ${hgt} with ${frames} frames is over the safe size for ${model.name}. Lower the resolution or the frame count, or turn on Unlock limits (advanced).`); }
        }
      } else if (lim.max_area && w * hgt > lim.max_area) {
        if (unlocked) beyond = true;
        else { ok = false; warn(`${w} × ${hgt} is over the safe size for ${model.name}. Lower the resolution, or turn on Unlock limits (advanced).`); }
      }
    }
    if (ok && unlocked && beyond) warn(`Above the safe limits for ${model.name}. It may run out of memory or crash; the memory estimate below shows how likely that is.`);
    if (!activeRunning()) f.go.disabled = !ok;
    return ok;
  }

  // ---- live memory estimate (VRAM and system RAM) against what this PC has right now ----
  function refreshEstimate() {
    clearTimeout(estTimer);
    estTimer = setTimeout(runEstimate, 200);
  }

  async function runEstimate() {
    if (!model || !alive) return;
    const [w, hgt] = getSize();
    if (!w || !hgt) return;
    const seq = ++estSeq;
    try {
      const e = await api.post("/api/estimate", {
        model_id: model.id, width: w, height: hgt, frames: kind === "video" ? Number(f.frames.value) : undefined,
        offload: f.offload.value, quantization: f.quant.value, vae_tiling: f.tiling.checked,
      });
      if (seq === estSeq && alive) renderEstimate(e);
    } catch { /* the estimate is a convenience: never get in the way */ }
  }

  function renderEstimate(e) {
    clear(f.est);
    const row = (name, used, cap, level, title) => h("div", { class: "est-row", title },
      h("span", { class: "est-name" }, name),
      h("span", { class: "mono est-nums" }, cap ? `~${used} GB of ${cap} GB free` : `~${used} GB`),
      statusBadge(STATUS_KIND[level], STATUS_LABEL[level]));
    const b = e.breakdown || {};
    const vramTip = b.weights_gb !== undefined ? `Weights on the GPU ~${b.weights_gb} GB, activations ~${b.activations_gb} GB, VAE decode ~${b.vae_decode_gb} GB` : "Scaled from the model's listed VRAM need";
    f.est.append(
      h("div", { class: "field-label" }, "Memory estimate"),
      row("GPU memory", e.vram_gb, e.vram_cap_gb, e.vram, vramTip),
      row("System RAM", e.ram_gb, e.ram_cap_gb, e.ram, "Python and CUDA libraries, the model's weights while they wait on the CPU, and the decoded frames"));
    if (e.status === "fail") f.est.append(h("div", { class: "notice notice-danger" }, icon("alert"),
      "This will probably run out of memory or crash on this PC. You can still try it. To bring it down: lower the resolution or frame count, use NF4 quantization, or set Offloading to Sequential."));
    else if (e.status === "risky") f.est.append(h("div", { class: "notice notice-warning" }, icon("alert"),
      "Close to what this PC has free right now. It may work, but other programs using memory could make it fail."));
    for (const n of e.notes || []) f.est.append(h("div", { class: "field-hint" }, n));
    f.est.append(h("div", { class: "field-hint" }, "A rough estimate; the real peak is recorded in History after each run."));
  }

  function readForm() {
    const [w, hgt] = getSize();
    return {
      model_id: model.id, prompt: f.prompt.value, negative_prompt: f.negative.value,
      width: w, height: hgt, frames: kind === "video" ? Number(f.frames.value) : undefined, steps: Number(f.steps.value),
      fps: kind === "video" ? Number(f.fps.value) || undefined : undefined, unlock: isUnlocked(),
      guidance: Number(f.guidance.value), seed: f.seed.value === "" ? null : Number(f.seed.value),
      offload: f.offload.value, quantization: f.quant.value, vae_tiling: f.tiling.checked,
      attention_backend: f.attnField.hidden ? undefined : f.attn.value,
      count: Number(f.count.value) || 1,
    };
  }

  function persist() {
    if (!model) return;
    saveForm({
      kind, model_id: model.id, prompt: f.prompt.value, negative_prompt: f.negative.value, seed: f.seed.value,
      res: f.res.value, cw: f.cw.value, ch: f.ch.value, fps: f.fps.value, frames: f.frames.value, steps: f.steps.value, guidance: f.guidance.value,
      offload: f.offload.value, quantization: f.quant.value, vae_tiling: f.tiling.checked, count: f.count.value,
      attention_backend: f.attn.value,
    });
  }

  function loadSettings(clip) {
    const m = modelById(clip.model_id);
    if (!m) { toast(`${clip.model_name} is not installed any more.`, "warning"); return; }
    const p = clip.params;
    if (m.kind !== kind || !model || model.id !== m.id) { kind = m.kind; wantedModel = m.id; buildForm(); }
    f.prompt.value = clip.prompt; f.negative.value = p.negative_prompt || "";
    if (p.unlock && !f.unlock.checked) { f.unlock.checked = true; unlockMap[m.id] = true; saveUnlock(unlockMap); applyLimits(); }
    if ([...f.res.options].some((o) => o.value === `${p.width}x${p.height}`)) f.res.value = `${p.width}x${p.height}`;
    else if (isUnlocked()) { f.res.value = "custom"; f.cw.value = p.width; f.ch.value = p.height; }
    syncCustom();
    if (kind === "video") { f.frames.value = p.frames; if (p.fps) f.fps.value = p.fps; }
    f.steps.value = p.steps;
    f.guidance.value = p.guidance; f.seed.value = p.seed;
    f.offload.value = p.offload; f.quant.value = p.quantization; f.tiling.checked = p.vae_tiling;
    if (p.attention_backend && [...f.attn.options].some((o) => o.value === p.attention_backend)) f.attn.value = p.attention_backend;
    fillCount(1);
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

  // Upscale a finished image or clip. With no upscaler installed the button leads to the Library to get one.
  function upscaleButtons(clip) {
    if (!upscalers.length) {
      return [h("a", { class: "btn", href: "#/library", title: "Install the Real-ESRGAN upscaler from the Library first" }, icon("upscale"), "Get the upscaler")];
    }
    const scales = upscalers[0].limits.scales || [2, 4];
    return scales.map((n) => h("button", { class: "btn", type: "button", disabled: activeRunning(),
      title: `Make this ${n}× bigger with ${upscalers[0].name}; the result is saved as a new item in History`,
      onclick: () => upscale(clip, n) }, icon("upscale"), `Upscale ${n}×`));
  }

  async function upscale(clip, scale) {
    try {
      activeJob = await api.post("/api/upscale", { clip_id: clip.id, scale });
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
      const up = !!(job.params && job.params.upscale);
      const phases = up ? PHASES.upscale : PHASES[job.kind || "video"];
      const order = phases.map((p) => p[0]);
      const idx = Math.max(0, order.indexOf(job.phase === "encoding-prompt" ? "loading" : job.phase));
      const inner = up
        ? (job.phase === "denoising" ? 0.05 + 0.9 * (job.step / Math.max(job.total, 1)) : job.phase === "encoding" ? 0.97 : 0.03)
        : job.phase === "denoising" || job.phase === "decoding"
          ? 0.1 + 0.8 * (job.step / Math.max(job.total, 1)) + (job.phase === "decoding" ? 0.05 : 0)
          : job.phase === "encoding" ? 0.97 : 0.05;
      const frac = ((job.item - 1) + inner) / Math.max(job.items, 1);
      const of = job.items > 1 ? `${job.kind === "image" ? "Image" : "Clip"} ${job.item} of ${job.items} · ` : "";
      put(outputCard,
        h("div", { class: "row-between" }, h("h3", {}, up ? `Upscaling ${job.params.upscale.scale}× with ${job.model_name}` : job.items > 1 ? `Generating ${job.items} with ${job.model_name}` : `Generating with ${job.model_name}`), statusBadge("info", "Running", "play")),
        h("ol", { class: "steps", "aria-label": "Generation steps" },
          phases.map(([id, label], i) => h("li", { class: "step", "data-state": i < idx ? "done" : i === idx ? "active" : "pending", "aria-current": i === idx ? "step" : undefined },
            icon(i < idx ? "check" : i === idx ? "play" : "clock"), h("span", {}, label)))),
        progressBar(frac, { indeterminate: job.phase === "loading" && job.step === 0, label: up ? "Upscale progress" : "Generation progress" }),
        h("div", { class: "progress-meta" },
          h("span", {}, job.phase === "denoising" && !up ? `${of}Step ${job.step} of ${job.total}` : `${of}${job.message}`),
          h("span", {}, [`${fmtDuration(job.elapsed)} elapsed`, job.eta ? `~${fmtDuration(job.eta)} left` : null].filter(Boolean).join(" · "))),
        job.reused ? h("div", { class: "field-hint" }, "The model was already loaded, so this run skipped loading.") : null,
        h("p", { class: "muted" }, job.params.prompt),
        job.clips.length ? h("div", { class: "stack-8" }, h("h4", {}, `Finished so far (${job.clips.length})`), batchStrip(job, false)) : null);
      return;
    }

    if (job && job.status === "failed") {
      outputCard.append(h("div", { class: "notice notice-danger", role: "alert" }, icon("error"),
        h("div", { class: "stack-8 grow" },
          h("strong", {}, "Generation failed"),
          h("div", {}, job.error || "Unknown error"),
          job.error_hint ? h("div", {}, job.error_hint) : null,
          job.error_detail ? h("pre", { class: "log mono", tabindex: 0, "aria-label": "Last error output" }, job.error_detail) : null,
          job.log_name ? h("div", { class: "row-8" },
            h("button", { class: "btn", type: "button", onclick: () => viewLog(job.log_name) }, icon("log"), "View full log")) : null)));
    } else if (job && job.status === "cancelled") {
      outputCard.append(h("div", { class: "notice notice-info" }, icon("info"), job.message || "Cancelled."));
    }

    const clip = history.clips.find((c) => c.id === shownClipId) || (job && job.clip);
    if (clip) { showClip(clip); return; }
    outputCard.append(h("div", { class: "empty" }, icon(kind === "image" ? "image" : "generate"),
      h("div", {}, h("strong", {}, kind === "image" ? "Your image appears here" : "Your clip appears here"), h("div", {}, "Pick a model, write a prompt and press Generate."))));
  }

  function showClip(clip) {
    shownClipId = clip.id;
    const p = clip.params;
    const image = isImageUrl(clip.url);
    const upscaled = !!p.upscale_scale;
    const media = image
      ? h("img", { class: "player", src: clip.url, alt: clip.prompt })
      : h("video", { class: "player", controls: true, loop: true, muted: true, autoplay: true, playsinline: true, src: clip.url });
    put(outputCard, media,
      activeJob && activeJob.clips && activeJob.clips.length > 1 ? batchStrip(activeJob, true) : null,
      h("p", {}, clip.prompt),
      h("div", { class: "stats" },
        stat("Model", clip.model_name), stat("Size", upscaled ? `${p.source_width} × ${p.source_height} → ${p.width} × ${p.height}` : `${p.width} × ${p.height}`),
        image ? null : stat("Frames", `${p.frames} @ ${p.fps} fps`),
        stat("Seed", String(p.seed)), stat("Time", fmtDuration(clip.seconds)),
        clip.device && clip.device !== "cpu" && clip.device !== "cuda" ? stat("GPU", clip.device.replace(/^NVIDIA\s+/, "").replace(/^GeForce\s+/, "")) : null,
        clip.peak_vram_gb ? stat("Peak VRAM", `${clip.peak_vram_gb} GB`) : null),
      h("div", { class: "row-8" },
        h("a", { class: "btn", href: clip.url, download: `reel-${clip.id}.${image ? "png" : "mp4"}` }, icon("download"), "Download"),
        upscaled ? null : h("button", { class: "btn", type: "button", onclick: () => loadSettings(clip) }, icon("load"), "Reuse settings"),
        upscaled ? null : h("button", { class: "btn", type: "button", disabled: activeRunning(), onclick: () => rerun(clip) }, icon("rerun"), "Re-run"),
        ...upscaleButtons(clip)));
    renderHistory();
  }

  // Thumbnails of one batch's results. While running they are informational; afterwards they switch the viewer.
  function batchStrip(job, interactive) {
    return h("div", { class: "batch-strip", role: "list", "aria-label": "This batch" }, job.clips.map((c, i) => {
      const img = isImageUrl(c.url)
        ? h("img", { src: c.url, alt: "", loading: "lazy" })
        : h("video", { src: `${c.url}#t=0.1`, preload: "metadata", muted: true, playsinline: true, tabindex: -1 });
      const on = c.id === shownClipId;
      return interactive
        ? h("button", { type: "button", class: "batch-thumb", role: "listitem", "aria-current": on ? "true" : undefined, "aria-label": `Result ${i + 1} of ${job.clips.length}, seed ${c.params.seed}`,
            title: `Seed ${c.params.seed}`, onclick: () => { shownClipId = c.id; renderOutput(); } }, img)
        : h("div", { class: "batch-thumb", role: "listitem", title: `Seed ${c.params.seed}` }, img);
    }));
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
          p.upscale_scale ? null : h("button", { class: "btn btn-ghost btn-icon", type: "button", title: "Reuse settings", "aria-label": "Reuse settings", onclick: () => loadSettings(clip) }, icon("load")),
          p.upscale_scale ? null : h("button", { class: "btn btn-ghost btn-icon", type: "button", title: "Re-run with the same seed", "aria-label": "Re-run", disabled: activeRunning(), onclick: () => rerun(clip) }, icon("rerun")),
          upscalers.length ? h("button", { class: "btn btn-ghost btn-icon", type: "button", title: `Upscale ${upscalers[0].limits.scales[0]}×`, "aria-label": "Upscale", disabled: activeRunning(), onclick: () => upscale(clip, upscalers[0].limits.scales[0]) }, icon("upscale")) : null,
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
    if (document.hidden) { if (activeRunning()) timer = setTimeout(poll, 2000); return; }
    try {
      const { job } = await api.get("/api/generate/current");
      if (job && (!activeJob || activeJob.id === job.id || job.status === "running")) {
        const wasRunning = activeJob && activeJob.status === "running";
        const had = activeJob && activeJob.id === job.id ? activeJob.clips.length : 0;
        activeJob = job;
        if (job.clips.length > had) await loadHistory();           // each item appears in History as it finishes
        if (job.status !== "running" && wasRunning) {
          await loadHistory();
          if (job.status === "done" && job.params.upscale) toast("Upscale finished. It is saved in History.", "success", 3500);
          else if (job.status === "done") toast(job.items > 1 ? `${job.items} ${job.kind === "image" ? "images" : "clips"} finished.` : (job.kind === "image" ? "Image finished." : "Clip finished."), "success", 3000);
        }
        if (job.status !== "running" && job.clip) shownClipId = job.clip.id;
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
  watcher = watchWorker((w) => { worker = w; renderWorker(); });
  buildForm();
  await loadHistory();
  const { job } = await api.get("/api/generate/current").catch(() => ({ job: null }));
  if (job && (job.status === "running" || job.status === "failed")) activeJob = job;
  else if (job && job.status === "done" && job.clip && history.clips.some((c) => c.id === job.clip.id)) { activeJob = job; shownClipId = job.clip.id; }
  else if (history.clips.length) shownClipId = history.clips[0].id;
  renderOutput();
  if (activeRunning()) poll();
  const onWake = () => { if (alive && activeRunning()) poll(); else if (alive) renderOutput(); };
  document.addEventListener("reel:wake", onWake);
  return { unmount() { alive = false; clearTimeout(timer); clearTimeout(estTimer); if (watcher) watcher.stop(); document.removeEventListener("reel:wake", onWake); } };
}
