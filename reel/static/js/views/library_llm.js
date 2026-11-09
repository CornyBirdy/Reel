// Language-model cards for the Library, and the chat-engine banner.
import { h, icon, put, toast, confirmDialog, fmtBytes, fmtDuration, progressBar, statusBadge, kindBadge, setText, once } from "../ui.js";
import { api } from "../api.js";
import { modelTitle, openLoadDialog } from "./llm_load.js";

const stat = (label, value) => h("div", { class: "stat" }, h("span", { class: "stat-label" }, label), h("span", { class: "stat-value" }, value));
const fmtParams = (label) => (label ? String(label) : "-");

// The same rule the server uses for its estimate: file size plus a little working space.
export function fitBadge(sizeBytes, gpu) {
  const need = sizeBytes / 1e9 + 1;
  if (!gpu) return statusBadge("neutral", "Runs on the CPU", "chip", "No NVIDIA GPU was found");
  const vram = (gpu.vram_total_mb / 1024) * 1.0737;
  if (need <= vram * 0.92) return statusBadge("success", `Fits your ${vram.toFixed(0)} GB GPU`, "check", "Estimate from the file size");
  if (sizeBytes / 1e9 <= vram * 3) return statusBadge("warning", "Partly on the GPU (slower)", "alert", `Needs about ${need.toFixed(0)} GB; your GPU has ${vram.toFixed(0)} GB, the rest goes to system RAM`);
  return statusBadge("danger", "Too big for this PC", "error", `Needs about ${need.toFixed(0)} GB`);
}

export function LlmCard(initial, ctx) {
  const badgeSlot = h("div", { class: "row-8 badges" });
  const fitSlot = h("div", { class: "row-8" });
  const bar = progressBar(0, { label: "Download progress" });
  const barFill = bar.querySelector(".bar");
  const metaLeft = h("span"), metaMid = h("span"), metaRight = h("span");
  const noticeSlot = h("div", { class: "stack-8" });
  const panel = h("div", { class: "stack-12 install-panel", hidden: true }, bar, h("div", { class: "progress-meta" }, metaLeft, metaMid, metaRight), noticeSlot);
  const statsSlot = h("div", { class: "stats" });
  const actions = h("div", { class: "row-8 card-actions" });
  const titleEl = h("h3");
  const subEl = h("div", { class: "muted text-sm" });
  const el = h("article", { class: "card model-card stack-16", "data-model": initial.id }, h("div", { class: "stack-8" }, h("div", {}, titleEl, subEl), badgeSlot), statsSlot, fitSlot, panel, actions);

  async function act(fn, ok) {
    try { await fn(); if (ok) toast(ok, "success"); } catch (e) { toast(e.message, "danger", 9000); }
    ctx.refresh();
  }

  function update(m) {
    const info = m.meta || {};
    setText(titleEl, modelTitle(m));
    once(subEl, m.repo + (m.license || ""), () => {
      put(subEl, h("a", { href: `https://huggingface.co/${m.repo}`, target: "_blank", rel: "noopener" }, m.repo), m.license ? ` · ${m.license}` : "");
    });
    const loaded = m.loaded, loading = m.loading;
    const [kind, text, ic] = loaded ? ["success", "Loaded", "check"] : loading ? ["info", "Loading", "clock"]
      : { ready: ["neutral", "Downloaded", "check"], downloading: ["info", "Downloading", "download"], partial: ["warning", "Incomplete", "alert"] }[m.state];
    const dl = m.download;
    const [bk, bt, bi] = dl && dl.status === "failed" && m.state === "partial" ? ["danger", "Download failed", "error"] : [kind, text, ic];
    once(badgeSlot, [bk, bt, m.quant].join("|"), () => {
      put(badgeSlot, kindBadge("language"), m.quant ? statusBadge("neutral", m.quant, "box", "Quantization: smaller numbers are smaller and a little less accurate") : null, statusBadge(bk, bt, bi));
    });
    once(statsSlot, [m.size, m.quant, info.context_length, info.size_label].join("|"), () => {
      put(statsSlot, stat("Size", fmtBytes(m.size)), stat("Parameters", fmtParams(info.size_label)),
        stat("Max context", info.context_length ? `${info.context_length.toLocaleString()} tokens` : "-"), info.layers ? stat("Layers", String(info.layers)) : null);
    });
    once(fitSlot, [m.size, ctx.gpu && ctx.gpu.vram_total_mb].join("|"), () => fitSlot.append(fitBadge(m.size, ctx.gpu)));

    const show = m.state === "downloading" || (m.state === "partial");
    panel.hidden = !show;
    if (show) {
      const have = dl && dl.status === "running" ? dl.bytes_done : m.bytes_on_disk;
      const frac = m.size ? Math.min(1, have / m.size) : 0;
      barFill.style.width = `${frac * 100}%`;
      bar.setAttribute("aria-valuenow", String(Math.round(frac * 100)));
      setText(metaLeft, `${fmtBytes(have)} of ${fmtBytes(m.size)}`);
      setText(metaMid, dl && dl.status === "running" ? [dl.speed > 0 ? `${fmtBytes(dl.speed)}/s` : null, dl.eta ? `${fmtDuration(dl.eta)} left` : null].filter(Boolean).join(" · ") : "");
      setText(metaRight, `${Math.round(frac * 100)}%`);
      once(noticeSlot, JSON.stringify([dl && dl.status, dl && dl.error, dl && dl.message]), () => {
        if (dl && dl.status === "failed") noticeSlot.append(h("div", { class: "notice notice-danger", role: "alert" }, icon("error"), dl.error));
        else if (dl && dl.status === "cancelled") noticeSlot.append(h("div", { class: "notice notice-info" }, icon("info"), dl.message));
        else if (dl && dl.status === "running" && /retry/i.test(dl.message)) noticeSlot.append(h("div", { class: "notice notice-warning" }, icon("alert"), dl.message));
      });
    }

    once(actions, [m.state, m.loaded, m.loading, ctx.engineReady].join("|"), () => {
      const del = (label, body) => h("button", { class: "btn btn-danger", type: "button", onclick: async () => {
        const ok = await confirmDialog({ title: `${label} ${modelTitle(m)}?`, danger: true, confirmLabel: label,
          body: h("div", { class: "stack-8" }, h("div", {}, body), h("div", {}, "Frees ", h("strong", {}, fmtBytes(m.state === "ready" ? m.size : m.bytes_on_disk)), ". Your chats are not touched.")) });
        if (ok) act(() => api.del(`/api/llm/models/${m.id}`), `${modelTitle(m)} removed.`);
      } }, icon("trash"), label);
      if (m.state === "downloading") {
        actions.append(h("button", { class: "btn", type: "button", onclick: () => act(() => api.post(`/api/llm/models/${m.id}/cancel`)) }, icon("stop"), "Cancel download"));
      } else if (m.state === "partial") {
        actions.append(h("button", { class: "btn btn-primary", type: "button", onclick: () => act(() => api.post(`/api/llm/models/${m.id}/resume`)) }, icon("download"), "Resume download"),
          del("Remove files", "This deletes the partly downloaded files."));
      } else {
        actions.append(
          h("a", { class: "btn btn-primary", href: `#/chat?model=${encodeURIComponent(m.id)}` }, icon("chat"), "Chat"),
          m.loaded ? h("button", { class: "btn", type: "button", onclick: () => act(() => api.post("/api/llm/unload")) }, icon("eject"), "Eject")
            : h("button", { class: "btn", type: "button", disabled: !ctx.engineReady || m.loading || undefined, title: ctx.engineReady ? undefined : "Install the chat engine first",
                onclick: async () => { if (await openLoadDialog(m, ctx.overview)) ctx.refresh(); } }, icon("load"), "Load…"),
          del("Delete", "This deletes the model file from this PC. You can download it again later."));
      }
    });
  }
  update(initial);
  return { el, update };
}

// Shown above the cards when the chat engine is missing or being installed.
export function EngineBanner(ctx) {
  const el = h("div", { class: "stack-8" });
  function update(engine) {
    const job = engine.job;
    const running = job && job.status === "running";
    const show = !engine.installed && ctx.needed();
    el.hidden = !show;
    once(el, [show, running, job && job.status, job && job.error].join("|"), () => {
      if (!show) return;
      el.append(h("div", { class: "notice notice-warning engine-banner" }, icon("alert"),
        h("div", { class: "stack-8 grow" },
          h("div", {}, h("strong", {}, "The chat engine is not installed. "), "Language models need it to run. It is a one-time download of a few hundred MB (llama.cpp, from its official releases)."),
          running ? h("div", { class: "stack-8" }, progressBar(job.progress, { indeterminate: !job.bytes_total, label: "Engine download" }), h("div", { class: "progress-meta" }, h("span", {}, job.message))) : null,
          job && job.status === "failed" ? h("div", { class: "mono" }, job.error) : null,
          h("div", {}, h("button", { class: "btn btn-primary", type: "button", disabled: !!running, onclick: async () => {
            try { await api.post("/api/llm/engine/install"); } catch (e) { toast(e.message, "danger", 9000); }
            ctx.refresh();
          } }, icon("download"), running ? "Installing…" : "Install chat engine")))));
    });
    if (running) {      // keep the progress bar moving without rebuilding the banner
      const bar = el.querySelector(".bar");
      if (bar && job.bytes_total) { bar.parentElement.classList.remove("indeterminate"); bar.style.width = `${job.progress * 100}%`; }
      const msg = el.querySelector(".progress-meta span");
      if (msg) setText(msg, job.bytes_total ? `${job.message} · ${fmtBytes(job.bytes_done)} of ${fmtBytes(job.bytes_total)}` : job.message);
    }
  }
  return { el, update };
}
