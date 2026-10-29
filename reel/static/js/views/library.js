import { h, icon, clear, toast, confirmDialog, fmtBytes, fmtDuration, progressBar, statusBadge, kindBadge } from "../ui.js";
import { api } from "../api.js";
import { openAddFromHF } from "./hf_dialog.js";

const FILTER_KEY = "reel.library.filter";
const loadFilter = () => { try { return localStorage.getItem(FILTER_KEY) || "all"; } catch { return "all"; } };

const STATE_BADGE = {
  installed:    ["success", "Installed", "check"],
  installing:   ["info", "Installing", "download"],
  uninstalling: ["info", "Removing", "trash"],
  partial:      ["warning", "Incomplete", "alert"],
  not_installed: ["neutral", "Not installed", "box"],
};

function stat(label, value) {
  return h("div", { class: "stat" }, h("span", { class: "stat-label" }, label), h("span", { class: "stat-value" }, value));
}

function stepIcon(state) {
  return { done: "check", active: "download", failed: "error", pending: "clock" }[state] || "clock";
}

function ModelCard(initial, ctx) {
  const m0 = initial;
  const badgeSlot = h("div", { class: "row-8 badges" });
  const fitSlot = h("div", { class: "row-8" });
  const diskSlot = h("div", { class: "muted icon-line" });
  const stepsEl = h("ol", { class: "steps", "aria-label": "Install steps" });
  const barSlot = h("div");
  const metaEl = h("div", { class: "progress-meta" });
  const msgEl = h("div", { class: "muted mono install-msg" });
  const noticeSlot = h("div", { class: "stack-8" });
  const logEl = h("pre", { class: "log mono" });
  const logDetails = h("details", {}, h("summary", {}, "Install log"), logEl);
  const panel = h("div", { class: "stack-12 install-panel", hidden: true }, stepsEl, barSlot, metaEl, msgEl, noticeSlot, logDetails);
  const actions = h("div", { class: "row-8 card-actions" });

  const el = h("article", { class: "card model-card stack-16", "data-model": m0.id },
    h("div", { class: "stack-8" },
      h("div", {},
        h("h3", {}, m0.name, " ", h("span", { class: "muted model-version" }, m0.version || "")),
        h("div", { class: "muted", style: { fontSize: "var(--text-sm)" } }, m0.family || "",
          m0.license ? [" · ", m0.license_url ? h("a", { href: m0.license_url, target: "_blank", rel: "noopener" }, m0.license) : m0.license] : null)),
      badgeSlot),
    h("p", { class: "muted" }, m0.summary || ""),
    h("div", { class: "stats" },
      stat("Download", `${m0.download_gb} GB`),
      stat("VRAM needed", `~${m0.vram_gb} GB`),
      stat("Max resolution", `${m0.limits.max_width} × ${m0.limits.max_height}`),
      m0.kind === "video" ? stat("Max frames", String(m0.limits.frames.max)) : null,
      m0.kind === "video" ? stat("Frame rate", `${m0.limits.fps} fps`) : null),
    fitSlot, diskSlot, panel, actions);

  let lastActionsKey = "";
  let lastStepsKey = "";

  async function act(fn, okMsg) {
    try { await fn(); if (okMsg) toast(okMsg, "success"); }
    catch (e) { toast(e.message, "danger", 9000); }
    ctx.refresh();
  }

  function renderActions(m) {
    const job = m.install;
    const key = [m.state, m.status, !!m.blocked_reason, job && job.status, m.source].join("|");
    if (key === lastActionsKey) return;
    lastActionsKey = key;
    clear(actions);
    if (m.status === "placeholder") {
      actions.append(h("button", { class: "btn", disabled: true, title: m.blocked_reason }, icon("download"), "Install"),
        h("span", { class: "muted", style: { fontSize: "var(--text-sm)" } }, "Adapter not written yet"));
      return;
    }
    const install = (label) => h("button", {
      class: "btn btn-primary", disabled: !!m.blocked_reason, title: m.blocked_reason || undefined,
      onclick: () => act(() => api.post(`/api/models/${m.id}/install`)),
    }, icon("download"), label);
    const blockedNote = () => m.blocked_reason
      ? h("span", { class: "muted", style: { fontSize: "var(--text-sm)" } }, m.blocked_reason) : null;
    const remove = (label, body) => h("button", {
      class: "btn btn-danger", onclick: async () => {
        const ok = await confirmDialog({
          title: `${label} ${m.name}?`, danger: true, confirmLabel: label,
          body: h("div", { class: "stack-8" },
            h("div", {}, body),
            h("div", {}, "Frees ", h("strong", {}, fmtBytes(m.disk.total)), ". Your generated clips are not touched.")),
        });
        if (ok) act(() => api.del(`/api/models/${m.id}`), `${m.name} removed.`);
      },
    }, icon("trash"), label);
    switch (m.state) {
      case "installing":
        actions.append(h("button", { class: "btn", onclick: () => act(() => api.post(`/api/models/${m.id}/cancel`)) }, icon("stop"), "Cancel install"));
        break;
      case "installed":
        actions.append(
          h("a", { class: "btn btn-primary", href: `#/generate?model=${encodeURIComponent(m.id)}` }, icon("generate"), "Generate"),
          remove("Uninstall", "This deletes only this model's folder: its environment and weights."));
        break;
      case "partial":
        actions.append(install(job && job.status === "failed" ? "Retry install" : "Resume install"),
          remove("Remove files", "This deletes the partly installed model folder."));
        break;
      case "uninstalling":
        actions.append(h("button", { class: "btn", disabled: true }, "Removing…"));
        break;
      default: {
        const items = [install("Install"), blockedNote()];
        if (m.source === "user") {
          items.push(h("button", { class: "btn btn-ghost", type: "button", onclick: async () => {
            const ok = await confirmDialog({ title: `Remove ${m.name} from the Library?`, confirmLabel: "Remove",
              body: "This only removes the entry. You can add it again from Hugging Face." });
            if (ok) act(() => api.del(`/api/models/${m.id}/recipe`), `${m.name} removed.`);
          } }, "Remove from Library"));
        }
        actions.append(...items.filter(Boolean));
      }
    }
  }

  function renderInstall(m) {
    const job = m.install;
    const show = m.state === "installing" || (job && (job.status === "failed" || job.status === "cancelled") && m.state !== "installed");
    panel.hidden = !show;
    if (!show) { lastStepsKey = ""; return; }
    const stepsKey = job.phases.map((p) => p.state).join();
    if (stepsKey !== lastStepsKey) {
      lastStepsKey = stepsKey;
      clear(stepsEl);
      for (const p of job.phases) {
        stepsEl.append(h("li", { class: "step", "data-state": p.state, "aria-current": p.state === "active" ? "step" : undefined },
          icon(stepIcon(p.state)), h("span", {}, p.label), h("span", { class: "sr-only" }, ` - ${p.state}`)));
      }
    }
    const weights = job.phase === "weights" && job.bytes_total > 0;
    clear(barSlot);
    barSlot.append(progressBar(job.progress, {
      indeterminate: job.status === "running" && !weights,
      label: `${m.name} install progress`,
    }));
    clear(metaEl);
    if (job.status === "running" && weights) {
      metaEl.append(
        h("span", {}, `${fmtBytes(job.bytes_done)} of ${fmtBytes(job.bytes_total)}`),
        h("span", {}, [job.speed > 0 ? `${fmtBytes(job.speed)}/s` : null, job.eta ? `${fmtDuration(job.eta)} left` : null].filter(Boolean).join(" · ")),
        h("span", {}, `${Math.round(job.progress * 100)}% overall`));
    } else {
      metaEl.append(h("span", {}, job.status === "running" ? job.phases.find((p) => p.state === "active")?.label ?? "" : ""),
        h("span", {}, `${Math.round(job.progress * 100)}% overall`));
    }
    msgEl.textContent = job.status === "running" ? job.message : "";
    clear(noticeSlot);
    for (const w of job.warnings || []) noticeSlot.append(h("div", { class: "notice notice-warning" }, icon("alert"), w));
    if (job.status === "failed") noticeSlot.append(h("div", { class: "notice notice-danger", role: "alert" }, icon("error"), job.error || "Install failed."));
    if (job.status === "cancelled") noticeSlot.append(h("div", { class: "notice notice-info" }, icon("info"), job.message));
    const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 8;
    logEl.textContent = job.log.join("\n");
    if (atBottom) logEl.scrollTop = logEl.scrollHeight;
  }

  function update(m) {
    const job = m.install;
    let [kind, text, ic] = m.status === "placeholder" ? ["neutral", "Placeholder", "box"] : STATE_BADGE[m.state];
    if (m.state === "partial" && job && job.status === "failed") [kind, text, ic] = ["danger", "Install failed", "error"];
    clear(badgeSlot);
    badgeSlot.append(kindBadge(m.kind));
    if (m.verified === false) badgeSlot.append(statusBadge("neutral", "Unverified", "info", "Reel has not run this model itself yet. It should work, but you may be the first to try it."));
    if (m.gated) badgeSlot.append(ctx.hfTokenSet ? statusBadge("info", "Gated", "info", "Needs accepted terms on Hugging Face (token saved)")
                                                 : statusBadge("warning", "Needs HF token", "alert", "Gated model: accept its terms on Hugging Face and save a token in Settings"));
    badgeSlot.append(statusBadge(kind, text, ic));

    clear(fitSlot);
    const gpu = ctx.gpu;
    if (gpu) {
      const totalGb = gpu.vram_total_mb / 1024;
      if (m.vram_gb > totalGb) fitSlot.append(statusBadge("warning", `Needs ~${m.vram_gb} GB; your GPU has ${totalGb.toFixed(0)} GB`, "alert"));
      else fitSlot.append(statusBadge("success", `Should fit your ${totalGb.toFixed(0)} GB GPU (estimate)`, "check"));
      if (m.vram_note) fitSlot.append(h("span", { class: "muted", style: { fontSize: "var(--text-xs)" } }, m.vram_note));
    }

    diskSlot.hidden = m.disk.total === 0;
    clear(diskSlot);
    if (m.disk.total) {
      diskSlot.append(icon("disk"), " On disk: ", h("strong", {}, fmtBytes(m.disk.total)),
        ` (weights ${fmtBytes(m.disk.weights)} · environment ${fmtBytes(m.disk.venv)})`);
    }
    renderInstall(m);
    renderActions(m);
  }

  update(initial);
  return { el, update };
}

export async function mount(root) {
  const grid = h("div", { class: "model-grid" });
  const summary = h("p", {});
  const notices = h("div", { class: "stack-8" });
  const empty = h("div", { class: "empty", hidden: true }, icon("box"), h("div", {}, "No models of this type yet. Use Add from Hugging Face to get one."));
  let filter = loadFilter();
  const filterSeg = h("div", { class: "seg", role: "group", "aria-label": "Show models" });
  const addBtn = h("button", { class: "btn btn-primary", type: "button", onclick: () => openAddFromHF({ onAdded: () => refresh() }) },
    icon("plus"), "Add from Hugging Face");
  root.append(h("div", { class: "view" },
    h("div", { class: "row-between" },
      h("div", { class: "page-head" }, h("h1", {}, "Library"), summary),
      h("div", { class: "row-8" }, filterSeg, addBtn)),
    notices, grid, empty));

  const cards = new Map();
  let timer = null;
  let alive = true;
  let lastData = null;
  const ctx = { gpu: null, hfTokenSet: false, refresh };

  function drawFilter() {
    clear(filterSeg);
    for (const [v, label, ic] of [["all", "All", null], ["image", "Image", "image"], ["video", "Video", "generate"]]) {
      filterSeg.append(h("button", { type: "button", "aria-pressed": String(filter === v), onclick: () => {
        filter = v; try { localStorage.setItem(FILTER_KEY, v); } catch { /* ignore */ }
        drawFilter(); applyFilter();
      } }, ic ? icon(ic) : null, label));
    }
  }

  function applyFilter() {
    let shown = 0;
    for (const [id, c] of cards) {
      const m = lastData && lastData.models.find((x) => x.id === id);
      const visible = !m || filter === "all" || m.kind === filter;
      c.el.hidden = !visible;
      if (visible) shown++;
    }
    empty.hidden = shown > 0 || !lastData;
  }

  async function refresh() {
    let data, sys, settings;
    try { [data, sys, settings] = await Promise.all([api.get("/api/models"), api.get("/api/system"), api.get("/api/settings")]); }
    catch (e) { clear(notices); notices.append(h("div", { class: "notice notice-danger" }, icon("error"), `Cannot reach Reel: ${e.message}`)); schedule(5000); return; }
    if (!alive) return;
    lastData = data;
    ctx.gpu = sys.gpu.gpus[0] || null;
    ctx.hfTokenSet = settings.hf_token_set;
    const installed = data.models.filter((m) => m.state === "installed");
    const used = data.models.reduce((a, m) => a + m.disk.total, 0);
    summary.textContent = `${installed.length} of ${data.models.length} installed · ${fmtBytes(used)} used · ${fmtBytes(data.disk_free)} free in ${data.models_dir}`;
    clear(notices);
    if (sys.gpu.cuda_12_8_ok === false) {
      notices.append(h("div", { class: "notice notice-warning" }, icon("alert"),
        `NVIDIA driver ${sys.gpu.driver} is too old for CUDA 12.8 (RTX 50-series needs 570 or newer). Update the driver before generating.`));
    }
    for (const [file, errs] of Object.entries(data.recipe_problems || {})) {
      notices.append(h("div", { class: "notice notice-danger" }, icon("error"), `Recipe ${file} was skipped: ${errs.join("; ")}`));
    }
    // Drop cards for models that no longer exist (e.g. an imported model that was removed).
    const ids = new Set(data.models.map((m) => m.id));
    for (const [id, c] of cards) if (!ids.has(id)) { c.el.remove(); cards.delete(id); }
    // Order: placeholders last; then built-in before imported; otherwise as listed.
    const rank = (m) => (m.status === "placeholder" ? 2 : m.source === "user" ? 1 : 0);
    const ordered = [...data.models].sort((a, b) => rank(a) - rank(b));
    for (const m of ordered) {
      let c = cards.get(m.id);
      if (!c) { c = ModelCard(m, ctx); cards.set(m.id, c); }
      c.update(m);
      grid.append(c.el); // re-appending an existing node only reorders it
    }
    applyFilter();
    schedule(data.models.some((m) => m.state === "installing" || m.state === "uninstalling") ? 1000 : 5000);
  }

  function schedule(ms) { clearTimeout(timer); if (alive) timer = setTimeout(refresh, ms); }
  drawFilter();
  await refresh();
  return { unmount() { alive = false; clearTimeout(timer); } };
}
