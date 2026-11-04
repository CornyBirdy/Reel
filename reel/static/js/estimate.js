// The live memory estimate (VRAM and system RAM) against what this PC has free right now. It warns; it never blocks.
import { h, icon, clear, statusBadge, tip } from "./ui.js";
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
  const row = (name, used, cap, level, title, total) => h("div", { class: "est-row", title },
    h("span", { class: "est-name" }, name),
    h("span", { class: "mono est-nums" }, cap && total
      ? [h("div", {}, `~${used} GB needed`), h("div", { class: "est-sub" }, `card ${total} GB · ${cap} free`)]
      : cap ? `~${used} GB of ${cap} GB free` : `~${used} GB`),
    statusBadge(STATUS_KIND[level], STATUS_LABEL[level]));
  const b = e.breakdown || {};
  const vramTip = b.weights_gb !== undefined ? `Weights on the GPU ~${b.weights_gb} GB, activations ~${b.activations_gb} GB, VAE decode ~${b.vae_decode_gb} GB` : "Scaled from the model's listed VRAM need";
  // One line while all is well: the two verdicts. The working is one click away.
  el.append(
    h("div", { class: "est-head" },
      h("span", { class: "field-label" }, "Memory"),
      statusBadge(STATUS_KIND[e.vram], `GPU ~${e.vram_gb} GB`, null, `${vramTip}. ${vramShare(e)}`),
      statusBadge(STATUS_KIND[e.ram], `RAM ~${e.ram_gb} GB`, null, "Python and CUDA libraries, the model's weights while they wait on the CPU, and the decoded frames"),
      tip("A rough estimate against what this PC has free right now. The real peak is recorded in History after each run.")));
  if (e.status === "fail") el.append(h("div", { class: "notice notice-danger" }, icon("alert"),
    "Probably too much for this PC. You can still try it; lower the resolution or frames, use 4-bit quantization, or set Offloading to Sequential to bring it down."));
  else if (e.status === "risky") el.append(h("div", { class: "notice notice-warning" }, icon("alert"),
    "Close to what this PC has free right now. Other programs using memory could make it fail."));
  const more = [];
  if (e.vram_other_gb >= 0.5) more.push(h("div", { class: "field-hint" }, vramShare(e)));
  for (const n of e.notes || []) more.push(h("div", { class: "field-hint" }, n));
  el.append(h("details", { class: "est-more" }, h("summary", {}, "Where the memory goes"), h("div", { class: "stack-8" },
    row("GPU memory", e.vram_gb, e.vram_cap_gb, e.vram, `${vramTip}. ${vramShare(e)}`, e.vram_total_gb),
    row("System RAM", e.ram_gb, e.ram_cap_gb, e.ram, "Python and CUDA libraries, the model's weights while they wait on the CPU, and the decoded frames"),
    ...more)));
}
