// The live memory estimate (VRAM and system RAM) against what this PC has free right now. It warns; it never blocks.
import { h, icon, clear, statusBadge } from "./ui.js";
import { api } from "./api.js";

const STATUS_KIND = { ok: "success", risky: "warning", fail: "danger", unknown: "neutral" };
const STATUS_LABEL = { ok: "OK", risky: "Risky", fail: "Likely to fail", unknown: "Unknown" };

/** Estimate for these settings: { width, height, frames?, offload, quantization, vae_tiling }. */
export const fetchEstimate = (modelId, p) => api.post("/api/estimate", { model_id: modelId, ...p });

/** Why "free" is less than the card's size: Windows, the browser and Reel's own window use some of it. */
function vramShare(e) {
  if (e.vram_total_gb == null || e.vram_other_gb == null) return "";
  return `The card has ${e.vram_total_gb} GB (a "16 GB" card is 15.9 GB in the units PyTorch and Task Manager use); `
    + `${e.vram_other_gb} GB are in use by Windows and other programs, which leaves ${e.vram_cap_gb} GB for Reel.`;
}

export function renderEstimate(el, e) {
  clear(el);
  const row = (name, used, cap, level, title) => h("div", { class: "est-row", title },
    h("span", { class: "est-name" }, name),
    h("span", { class: "mono est-nums" }, cap ? `~${used} GB of ${cap} GB free` : `~${used} GB`),
    statusBadge(STATUS_KIND[level], STATUS_LABEL[level]));
  const b = e.breakdown || {};
  const vramTip = b.weights_gb !== undefined ? `Weights on the GPU ~${b.weights_gb} GB, activations ~${b.activations_gb} GB, VAE decode ~${b.vae_decode_gb} GB` : "Scaled from the model's listed VRAM need";
  el.append(
    h("div", { class: "field-label" }, "Memory estimate"),
    row("GPU memory", e.vram_gb, e.vram_cap_gb, e.vram, `${vramTip}. ${vramShare(e)}`),
    row("System RAM", e.ram_gb, e.ram_cap_gb, e.ram, "Python and CUDA libraries, the model's weights while they wait on the CPU, and the decoded frames"));
  if (e.status === "fail") el.append(h("div", { class: "notice notice-danger" }, icon("alert"),
    "This will probably run out of memory or crash on this PC. You can still try it. To bring it down: lower the resolution or frame count, use NF4 quantization, or set Offloading to Sequential."));
  else if (e.status === "risky") el.append(h("div", { class: "notice notice-warning" }, icon("alert"),
    "Close to what this PC has free right now. It may work, but other programs using memory could make it fail."));
  if (e.vram_other_gb >= 0.5) el.append(h("div", { class: "field-hint" }, vramShare(e)));
  for (const n of e.notes || []) el.append(h("div", { class: "field-hint" }, n));
  el.append(h("div", { class: "field-hint" }, "A rough estimate; the real peak is recorded in History after each run."));
}
