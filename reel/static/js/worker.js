// The loaded model: its state chip, what it holds, and the Load / Unload buttons. Used by Library and Generate.
import { fmtDuration, h, icon, statusBadge, toast, viewLog } from "./ui.js";
import { api } from "./api.js";

export const WORKER_STATE = {
  unloaded:  ["neutral", "Unloaded", "box"],
  loading:   ["info", "Loading", "download"],
  ready:     ["success", "Ready", "check"],
  generating: ["info", "Generating", "play"],
  unloading: ["info", "Unloading", "trash"],
};

export const KEEPS_MEMORY = "A loaded model keeps its memory until it is unloaded. With CPU offload that is mostly system RAM, so it can slow down other programs.";

/** The state of one model: only the model in memory is anything but unloaded. */
export const stateOf = (worker, modelId) => (worker && worker.model_id === modelId ? worker.state : "unloaded");

export function stateChip(state, title) {
  const [kind, text, ic] = WORKER_STATE[state] || WORKER_STATE.unloaded;
  return statusBadge(kind, text, ic, title);
}

const gb = (v) => (v === null || v === undefined ? "-" : `${v.toFixed(1)} GB`);

/** "VRAM 9.3 GB · RAM 14.1 GB" for the loaded model, or null. */
export function holdsText(worker) {
  if (!worker || worker.state === "unloaded" || !worker.memory) return null;
  const { vram_gb: v, ram_gb: r } = worker.memory;
  const chose = worker.settings?.offload === "auto" && worker.offload_active ? `Auto chose ${worker.offload_active} offload` : null;
  const parts = [v != null ? `VRAM ${gb(v)}` : null, r != null ? `RAM ${gb(r)}` : null, chose].filter(Boolean);
  return parts.length ? `Holds ${parts.join(" · ")}` : null;
}

/** While a model loads: what the model process is doing right now, for how long, and a way to read its log. */
export function loadingNote(worker) {
  if (!worker || worker.state !== "loading") return null;
  const s = worker.loading_seconds;
  return h("div", { class: "stack-8" },
    h("div", { class: "field-hint" }, `${worker.message || "Loading"}${s != null ? ` · ${fmtDuration(s)}` : ""}`),
    s != null && s > 90 && /PyTorch/.test(worker.message || "") && worker.python
      ? h("div", { class: "notice notice-warning" }, icon("alert"), h("div", { class: "stack-8" },
          h("div", {}, "Starting PyTorch is taking very long. That is Windows loading its large DLLs, often slowed by antivirus scanning the model folder. The first start is the slowest."),
          h("div", {}, "To time it outside Reel, run this in a Command Prompt:"),
          h("pre", { class: "log mono", tabindex: 0 }, `"${worker.python}" -c "import time; t=time.time(); import torch; print(torch.__version__, round(time.time()-t), 'seconds')"`)))
      : null,
    s != null && s > 300 ? h("div", { class: "field-hint" }, "Still loading. A big model takes several minutes the first time (it reads tens of GB from disk). The log says which step it is on, and Cancel load stops it.") : null,
    worker.log_name ? h("div", { class: "row-8" }, h("button", { class: "btn btn-ghost", type: "button", onclick: () => viewLog(worker.log_name) }, icon("log"), "View log")) : null);
}
export const loadingKey = (w) => (w && w.state === "loading" ? `${w.message}|${Math.floor(w.loading_seconds || 0)}` : "");

export const holdsTitle = "VRAM is what PyTorch has reserved in the model process (Task Manager adds about 0.5 GB for the CUDA context). RAM is the process's working set.";

export function idleText(worker) {
  if (!worker || worker.state !== "ready") return null;
  const m = worker.idle_unload_minutes;
  return m ? `Unloads itself after ${m} min without use (Settings)` : "Stays loaded until you unload it (Settings: idle unload is off)";
}

/** Load / Unload for one model. `memory` (optional) supplies the form's memory settings for a load. */
export function loadButton(model, worker, { onDone, memory, compact } = {}) {
  const state = stateOf(worker, model.id);
  const busyElsewhere = worker && worker.state !== "unloaded" && worker.model_id !== model.id && ["loading", "generating", "unloading"].includes(worker.state);
  const run = async (fn) => {
    try { await fn(); } catch (e) { toast(e.message, "danger", 9000); }
    if (onDone) onDone();
  };
  if (state === "ready") {
    return h("button", { class: "btn", type: "button", title: "Free this model's GPU and RAM", onclick: () => run(() => api.post(`/api/models/${model.id}/unload`)) },
      icon("eject"), compact ? "Unload" : "Unload from memory");
  }
  if (state === "loading") {
    return h("button", { class: "btn", type: "button", title: "Stop loading", onclick: () => run(() => api.post(`/api/models/${model.id}/unload`)) }, icon("stop"), "Cancel load");
  }
  if (state === "generating" || state === "unloading") {
    return h("button", { class: "btn", type: "button", disabled: true, title: state === "generating" ? "Cancel the running generation first" : "Unloading" }, icon("eject"), "Unload");
  }
  return h("button", { class: "btn", type: "button", disabled: busyElsewhere || undefined,
    title: busyElsewhere ? "Another model is busy" : (worker && worker.model_id && worker.state === "ready" ? "Loads this model and unloads the current one" : "Load this model into memory now, so the first Generate starts at once"),
    onclick: () => run(() => api.post(`/api/models/${model.id}/load`, memory ? memory() : undefined)) }, icon("load"), compact ? "Load" : "Load into memory");
}

/** A notice for a model that crashed or failed to load, with the log and a way to dismiss it. */
export function crashNotice(worker, onDone) {
  const c = worker && worker.last_crash;
  if (!c) return null;
  return h("div", { class: "notice notice-warning", role: "alert" }, icon("alert"),
    h("div", { class: "stack-8 grow" },
      h("div", {}, c.message),
      c.request ? h("div", { class: "muted" }, `The request that was running: ${c.request}.`) : null,
      c.hint ? h("div", {}, c.hint) : null,
      h("div", { class: "row-8" },
        c.log_name ? h("button", { class: "btn", type: "button", onclick: () => viewLog(c.log_name) }, icon("log"), "View full log") : null,
        h("button", { class: "btn btn-ghost", type: "button", onclick: async () => { try { await api.post("/api/worker/dismiss"); } catch { /* ignore */ } if (onDone) onDone(); } }, "Dismiss"))));
}

/** Poll /api/worker. Faster while something is happening. Returns a stop function. */
export function watchWorker(onChange) {
  let timer = null, alive = true, last = "";
  async function tick() {
    let delay = 3000;
    try {
      const w = await api.get("/api/worker");
      const key = JSON.stringify(w);
      if (key !== last) { last = key; if (alive) onChange(w); }
      if (["loading", "generating", "unloading"].includes(w.state)) delay = 1000;
    } catch { /* server busy */ }
    if (alive) timer = setTimeout(tick, delay);
  }
  tick();
  return { stop() { alive = false; clearTimeout(timer); }, poke() { clearTimeout(timer); tick(); } };
}
