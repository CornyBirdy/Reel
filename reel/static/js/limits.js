// What a model accepts and the labels for its options. Shared by Generate and the model settings dialog.

export const ATTN_LABEL = {
  auto: "Auto (per the diffusers docs for this GPU)", native: "Native (PyTorch)", sage_hub: "SageAttention (sage_hub)",
  flash_hub: "FlashAttention-2 (flash_hub)", _flash_3_hub: "FlashAttention-3 (_flash_3_hub, H100)", sage_blackwell_hub: "SageAttention 3 (sage_blackwell_hub, RTX 50)",
};
export const OFFLOAD_LABEL = { none: "None (fastest, most VRAM)", model: "Model offload", sequential: "Sequential (slowest, least VRAM)" };
export const QUANT_LABEL = { none: "Off (bf16)", int8: "8-bit", nf4: "4-bit NF4" };

/** The nearest valid frame count (the pipeline's step/offset rule) within [min, max]. */
export const snapFrames = (v, fr) => {
  const n = Math.round((v - fr.offset) / fr.step) * fr.step + fr.offset;
  return Math.max(fr.min, Math.min(fr.max, n));
};

export const dims = (w, h) => `${w} × ${h}`;

/**
 * The sizes to offer in a list: the recipe's presets, then the model's own native sizes (marked as heavy when they are above
 * what the recipe calls safe), then "custom". Nothing is hidden or blocked: any size up to the pipeline's ceiling can be typed in.
 */
export function resolutionChoices(model) {
  const lim = model.limits, nat = model.native?.resolutions || [];
  const key = (p) => `${p.w}x${p.h}`;
  const natKeys = new Set(nat.map(key));
  const safe = (p) => p.w <= lim.max_width && p.h <= lim.max_height;
  const out = [], seen = new Set();
  for (const p of [...lim.presets, ...nat]) {
    if (seen.has(key(p))) continue;
    seen.add(key(p));
    out.push({ value: key(p), label: `${dims(p.w, p.h)}${natKeys.has(key(p)) ? " · native" : ""}${safe(p) ? "" : " · heavy"}` });
  }
  out.push({ value: "custom", label: "Custom size…" });
  return out;
}

/** One line on what is safe and what the pipeline accepts. */
export function describeCeiling(model, kind) {
  const lim = model.limits, u = model.unlock, nat = model.native;
  const parts = [`Sized for a typical card: up to ${dims(lim.max_width, lim.max_height)}${kind === "video" ? ` and ${lim.frames.max} frames` : ""}.`];
  parts.push(`The pipeline accepts up to ${dims(u.max_width, u.max_height)}${kind === "video" ? ` and ${u.max_frames} frames` : ""}, in multiples of ${lim.size_multiple}${u.known ? "" : " (Reel is guessing this ceiling)"}.`);
  if (nat) parts.push(`Native: ${nat.resolutions.map((r) => dims(r.w, r.h)).slice(0, 3).join(", ")}${nat.frames ? `, ${nat.frames.join(" or ")} frames` : ""}${nat.fps ? ` at ${nat.fps} fps` : ""}.`);
  parts.push("Anything above the safe size is allowed: watch the memory estimate.");
  return parts.join(" ");
}
