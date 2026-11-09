// The "Load model" dialog: context length and where the model runs, with a live memory estimate.
import { h, icon, put, clear, toast, fmtTokens, statusBadge } from "../ui.js";
import { api } from "../api.js";

export function modelTitle(m) {
  const base = (m.meta && m.meta.name) || m.name;
  return m.quant && !base.toUpperCase().includes(m.quant) ? `${base} · ${m.quant}` : base;
}

const GB = 1e9;

// Resolves to true if loading was started, false if the dialog was cancelled.
export async function openLoadDialog(model, overview) {
  let info;
  try { info = await api.get(`/api/llm/models/${model.id}/suggest`); }
  catch (e) { toast(e.message, "danger"); return false; }
  const { suggested: sug, last_used: last } = info;
  const gpus = overview.gpu || [];

  return new Promise((resolve) => {
    const placement = h("select", { id: "ll-gpu", class: "select", onchange: () => update(true) },
      h("option", { value: "" }, "Default (see Settings)"),
      gpus.map((g) => h("option", { value: g.uuid }, `${g.name.replace(/^NVIDIA\s+/, "").replace(/^GeForce\s+/, "")} · ${(g.vram_total_mb / 1024).toFixed(0)} GB`)),
      gpus.length > 1 ? h("option", { value: "all" }, "All GPUs (split the model)") : null,
      h("option", { value: "cpu" }, "CPU only"));
    placement.value = (last && last.gpu) || "";
    const ctx = h("select", { id: "ll-ctx", class: "select", onchange: () => update(false) },
      sug.context_choices.map((c) => h("option", { value: c }, `${c.toLocaleString()} tokens`)));
    ctx.value = String(last && last.context && sug.context_choices.includes(last.context) ? last.context : sug.context);
    const layers = h("input", { id: "ll-layers", type: "range", min: 0, max: sug.layers || 99, step: 1, oninput: () => update(false) });
    layers.value = last && last.gpu_layers != null ? Math.min(last.gpu_layers, sug.layers || 99) : Math.min(sug.gpu_layers, sug.layers || 99);
    const layersOut = h("span", { class: "mono" });
    const estimate = h("div", { class: "stack-8" });
    const go = h("button", { class: "btn btn-primary", type: "button", onclick: submit }, icon("load"), "Load model");

    function vramGb() {
      const sel = placement.value;
      if (sel === "cpu") return 0;
      const pick = sel === "all" ? gpus : gpus.filter((g) => g.uuid === sel);
      const use = pick.length ? pick : (gpus.length ? [gpus[0]] : []);
      return use.reduce((a, g) => a + g.vram_total_mb, 0) / 1024 * 1.0737 * 0.92;
    }

    function update(placementChanged) {
      const cpu = placement.value === "cpu" || !gpus.length;
      const total = sug.layers || 99;
      if (placementChanged) layers.value = cpu ? 0 : total;
      const n = Number(layers.value);
      layersOut.textContent = cpu ? "0 (CPU)" : n >= total ? `All ${total}` : `${n} of ${total}`;
      layers.disabled = cpu;
      const c = Number(ctx.value);
      const kv = sug.kv_per_token * c;
      const size = info.model.size;
      const frac = total ? n / total : 1;
      const onGpu = (size * frac + kv * frac) / GB + 0.6, inRam = (size * (1 - frac) + kv * (1 - frac)) / GB;
      clear(estimate);
      const budget = vramGb();
      const over = !cpu && onGpu > budget;
      put(estimate, h("dl", { class: "kv" },
        h("dt", {}, "Model file"), h("dd", {}, `${(size / GB).toFixed(1)} GB`),
        h("dt", {}, "Context memory"), h("dd", {}, `${(kv / GB).toFixed(1)} GB`),
        h("dt", {}, "Needs on the GPU"), h("dd", {}, cpu ? "none" : `~${onGpu.toFixed(1)} GB of ${budget.toFixed(0)} GB`),
        h("dt", {}, "Needs in RAM"), h("dd", {}, `~${inRam.toFixed(1)} GB`)),
        over ? h("div", { class: "notice notice-warning" }, icon("alert"), "This is more than the GPU can hold. Lower the GPU layers or the context length, or the load will fail.")
          : (!cpu && frac < 1 ? h("div", { class: "notice notice-info" }, icon("info"), "Part of the model stays in system RAM. It works, but is slower.") : null));
    }

    const dlg = h("dialog", { class: "dialog", "aria-labelledby": "ll-title" },
      h("h3", { id: "ll-title" }, `Load ${modelTitle(model)}`),
      h("div", { class: "stack-12" },
        h("div", { class: "field" }, h("label", { for: "ll-gpu" }, "Run on"), placement),
        h("div", { class: "field" }, h("label", { for: "ll-ctx" }, "Context length"), ctx,
          h("span", { class: "field-hint" }, `How much of the conversation the model can see. This model supports up to ${sug.max_context.toLocaleString()} tokens; longer needs more memory.`)),
        h("div", { class: "field" }, h("label", { for: "ll-layers" }, h("span", {}, "GPU layers: "), layersOut), layers,
          h("span", { class: "field-hint" }, "How much of the model sits on the graphics card. All is fastest.")),
        estimate),
      h("div", { class: "dialog-actions" }, h("button", { class: "btn", type: "button", onclick: () => dlg.close("cancel") }, "Cancel"), go));

    async function submit() {
      go.disabled = true;
      try {
        await api.post("/api/llm/load", { model_id: model.id, context: Number(ctx.value), gpu_layers: Number(layers.value), gpu: placement.value });
        dlg.close("ok");
      } catch (e) { toast(e.message, "danger", 9000); go.disabled = false; }
    }
    dlg.addEventListener("close", () => { resolve(dlg.returnValue === "ok"); dlg.remove(); });
    document.body.append(dlg);
    update(false);
    dlg.showModal();
  });
}
