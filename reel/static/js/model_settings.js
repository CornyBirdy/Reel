// Model settings, the way LM Studio does them: how a model loads (memory) and what it generates with by default.
// Opened from the Library or Generate. Saved per model; Load and Generate start from them. Any size up to what the
// pipeline supports can be chosen, and the memory estimate warns instead of anything being blocked.
import { h, icon, clear, toast } from "./ui.js";
import { api } from "./api.js";
import { fetchEstimate, renderEstimate } from "./estimate.js";
import { ATTN_LABEL, OFFLOAD_HINT, OFFLOAD_LABEL, QUANT_LABEL, describeCeiling, dims, resolutionChoices, snapFrames } from "./limits.js";

const field = (label, control, hint, id) => h("div", { class: "field" }, h("label", { for: id }, label), control, hint ? h("span", { class: "field-hint" }, hint) : null);

/** Open the dialog for one model (an entry of /api/models). onSaved runs after a save, a reset or a load. */
export async function openModelSettings(model, { onSaved } = {}) {
  let s;
  try { s = await api.get(`/api/models/${model.id}/settings`); } catch (e) { toast(e.message, "danger"); return; }
  const video = model.kind === "video";
  const lim = model.limits, u = model.unlock, mem = model.memory || {};

  const c = {};
  c.res = h("select", { id: "ms-res", class: "select", onchange: () => { syncCustom(); changed(); } },
    resolutionChoices(model).map((o) => h("option", { value: o.value }, o.label)));
  c.cw = h("input", { id: "ms-cw", class: "input", type: "number", min: 64, max: u.max_width, step: lim.size_multiple, oninput: changed });
  c.ch = h("input", { id: "ms-ch", class: "input", type: "number", min: 64, max: u.max_height, step: lim.size_multiple, oninput: changed });
  c.custom = h("div", { class: "grid-2", hidden: true }, field("Width", c.cw, null, "ms-cw"), field("Height", c.ch, null, "ms-ch"));
  c.frames = h("input", { id: "ms-frames", type: "range", oninput: () => { syncFrames(); changed(); } });
  c.framesOut = h("span", { class: "mono" });
  c.fps = h("input", { id: "ms-fps", class: "input", type: "number", min: 1, max: 60, oninput: () => { syncFrames(); } });
  c.steps = h("input", { id: "ms-steps", class: "input", type: "number", min: lim.steps?.min ?? 1, max: lim.steps?.max ?? 100 });
  c.guidance = h("input", { id: "ms-guid", class: "input", type: "number", step: 0.1, min: 0, max: 20 });
  c.offload = h("select", { id: "ms-offload", class: "select", onchange: changed },
    (mem.offload_options || ["model"]).map((o) => h("option", { value: o }, OFFLOAD_LABEL[o] || o)));
  c.quant = h("select", { id: "ms-quant", class: "select", onchange: changed },
    (mem.quantization_options || ["none"]).map((o) => h("option", { value: o }, QUANT_LABEL[o] || o)));
  c.tiling = h("input", { id: "ms-tiling", type: "checkbox", onchange: changed });
  c.attn = h("select", { id: "ms-attn", class: "select" }, (mem.attention_options || []).map((o) => h("option", { value: o }, ATTN_LABEL[o] || o)));
  c.est = h("div", { class: "stack-8 mem-est", "aria-live": "polite" });
  if (video) { c.frames.min = lim.frames.min; c.frames.max = u.max_frames; c.frames.step = lim.frames.step; }

  const set = (v) => {
    const want = `${v.width}x${v.height}`;
    if ([...c.res.options].some((o) => o.value === want)) c.res.value = want;
    else { c.res.value = "custom"; c.cw.value = v.width; c.ch.value = v.height; }
    if (video) { c.frames.value = v.frames; c.fps.value = v.fps; }
    c.steps.value = v.steps; c.guidance.value = v.guidance;
    c.offload.value = v.offload; c.quant.value = v.quantization; c.tiling.checked = !!v.vae_tiling;
    if (mem.attention_options) c.attn.value = v.attention_backend;
    syncCustom(); syncFrames(); changed();
  };
  const size = () => (c.res.value === "custom" ? [Number(c.cw.value), Number(c.ch.value)] : c.res.value.split("x").map(Number));
  function syncCustom() { c.custom.hidden = c.res.value !== "custom"; if (!c.custom.hidden && !c.cw.value) { c.cw.value = s.effective.width; c.ch.value = s.effective.height; } }
  function syncFrames() {
    if (!video) return;
    c.frames.value = snapFrames(Number(c.frames.value), { ...lim.frames, max: u.max_frames });
    c.framesOut.textContent = `${c.frames.value} · ${(c.frames.value / (Number(c.fps.value) || lim.fps)).toFixed(1)} s`;
  }
  let timer = null, seq = 0;
  function changed() {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const [w, hh] = size();
      if (!w || !hh) return;
      const mine = ++seq;
      try {
        const e = await fetchEstimate(model.id, { width: w, height: hh, frames: video ? Number(c.frames.value) : undefined,
          offload: c.offload.value, quantization: c.quant.value, vae_tiling: c.tiling.checked });
        if (mine === seq) renderEstimate(c.est, e);
      } catch { /* the estimate is a convenience */ }
    }, 200);
  }
  const values = () => {
    const [w, hh] = size();
    const v = { width: w, height: hh, steps: Number(c.steps.value), guidance: Number(c.guidance.value),
      offload: c.offload.value, quantization: c.quant.value, vae_tiling: c.tiling.checked };
    if (video) Object.assign(v, { frames: Number(c.frames.value), fps: Number(c.fps.value) });
    if (mem.attention_options) v.attention_backend = c.attn.value;
    return v;
  };

  const dlg = h("dialog", { class: "dialog dialog-wide", "aria-labelledby": "ms-title" });
  const close = () => { clearTimeout(timer); seq++; dlg.close(); };
  const save = async (thenLoad) => {
    try {
      await api.put(`/api/models/${model.id}/settings`, values());
      if (thenLoad) await api.post(`/api/models/${model.id}/load`);
      toast(thenLoad ? `Settings saved. Loading ${model.name}…` : "Settings saved.", "success", 3000);
      close();
      if (onSaved) onSaved();
    } catch (e) { toast(e.message, "danger", 9000); }
  };
  const reset = async () => {
    try { s = await api.del(`/api/models/${model.id}/settings`); set(s.effective); toast("Back to the recommended settings. Press Save to keep them.", "info", 3500); }
    catch (e) { toast(e.message, "danger"); }
  };

  dlg.append(
    h("h3", { id: "ms-title" }, `${model.name} ${model.version || ""} · settings`.trim()),
    h("p", { class: "muted" }, "Used when this model loads and as Generate's starting values. Nothing here is capped at what suits a typical card: the memory estimate warns you instead."),
    h("div", { class: "ms-body stack-16" },
      h("h4", {}, "Output"),
      h("div", { class: video ? "grid-2" : "" }, field("Resolution", c.res, null, "ms-res"),
        video ? h("div", { class: "field" }, h("label", { for: "ms-frames" }, "Frames: ", c.framesOut), c.frames, h("span", { class: "field-hint" }, `${lim.frames.step}n+${lim.frames.offset}, up to ${u.max_frames}`)) : null),
      c.custom,
      h("div", { class: video ? "grid-3" : "grid-2" }, field("Steps", c.steps, null, "ms-steps"), field("Guidance", c.guidance, null, "ms-guid"),
        video ? field("Frames per second", c.fps, null, "ms-fps") : null),
      h("div", { class: "field-hint" }, describeCeiling(model, model.kind)),
      h("h4", {}, "Loading (memory)"),
      h("div", { class: "grid-2" }, field("Offloading", c.offload, OFFLOAD_HINT, "ms-offload"), field("Quantization", c.quant, null, "ms-quant")),
      mem.attention_options ? field("Attention backend", c.attn, mem.attention_note || "How the denoiser computes attention. Native is PyTorch's own and always works.", "ms-attn") : null,
      h("label", { class: "check" }, c.tiling, "Tiled VAE decode"),
      mem.notes ? h("div", { class: "field-hint" }, mem.notes) : null,
      c.est),
    h("div", { class: "dialog-actions ms-actions" },
      h("button", { class: "btn btn-ghost", type: "button", onclick: reset }, icon("rerun"), "Recommended"),
      h("span", { class: "grow" }),
      h("button", { class: "btn", type: "button", onclick: close }, "Cancel"),
      h("button", { class: "btn", type: "button", onclick: () => save(false) }, "Save"),
      h("button", { class: "btn btn-primary", type: "button", onclick: () => save(true) }, icon("load"), "Save and load")));
  dlg.addEventListener("close", () => dlg.remove());
  document.body.append(dlg);
  dlg.showModal();
  set(s.effective);
}
