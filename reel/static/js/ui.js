// Small DOM + UI helpers. No framework, no build step.

export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "value") el.value = value;
    else if (key === "checked" || key === "disabled" || key === "selected" || key === "open") el[key] = !!value;
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, value);
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function clear(el) { el.replaceChildren(); return el; }

// --- Icons: 24-unit grid, 1.75 outline, round caps and joins (suite icon style) ---
const ICONS = {
  library: '<rect x="3.5" y="3.5" width="7" height="7" rx="1.75"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.75"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.75"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.75"/>',
  generate: '<rect x="2.5" y="4.5" width="19" height="15" rx="3"/><path d="M10 9.2 15.2 12 10 14.8z"/>',
  settings: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2.5 12h2M19.5 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/>',
  monitor: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
  download: '<path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M5 19.5h14"/>',
  trash: '<path d="M4 7h16M10 7V4.5h4V7M6.5 7l1 12.5h9l1-12.5M10 11v5M14 11v5"/>',
  play: '<path d="M8 5.5v13l11-6.5z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  rerun: '<path d="M20 11a8 8 0 0 0-14.3-4.5M4 5v4h4M4 13a8 8 0 0 0 14.3 4.5M20 19v-4h-4"/>',
  dice: '<rect x="4" y="4" width="16" height="16" rx="3"/><circle cx="9" cy="9" r=".6"/><circle cx="15" cy="9" r=".6"/><circle cx="9" cy="15" r=".6"/><circle cx="15" cy="15" r=".6"/>',
  check: '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/>',
  alert: '<path d="M12 3.5 21.5 20h-19z"/><path d="M12 10v4.5M12 17.5v.01"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.01"/>',
  error: '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  chip: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2.5v3.5M15 2.5v3.5M9 18v3.5M15 18v3.5M2.5 9H6M2.5 15H6M18 9h3.5M18 15h3.5"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  disk: '<rect x="3" y="13" width="18" height="7" rx="2"/><path d="M3 13l2.5-7a2 2 0 0 1 1.9-1.3h9.2A2 2 0 0 1 18.5 6L21 13M7 16.5v.01"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
  load: '<path d="M4 12h12M11 7l5 5-5 5M20 4v16"/>',
  image: '<rect x="3.5" y="4.5" width="17" height="15" rx="3"/><circle cx="9" cy="10" r="1.6"/><path d="M4 17l5-4.5 4 3.5 3-2.5 4 3.5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>',
  back: '<path d="M10 6l-6 6 6 6M4 12h16"/>',
  box: '<path d="M12 3 20.5 7.5v9L12 21l-8.5-4.5v-9z"/><path d="M3.5 7.5 12 12l8.5-4.5M12 12v9"/>',
};

export function icon(name) {
  const wrap = document.createElement("span");
  wrap.innerHTML =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${ICONS[name] || ""}</svg>`;
  return wrap.firstChild;
}

// --- Toasts ---
const TOAST_ICON = { success: "check", danger: "error", info: "info", warning: "alert" };

export function toast(message, kind = "info", ms = 5000) {
  const host = document.getElementById("toasts");
  if (!host) return;
  const el = h("div", { class: `toast toast-${kind}`, role: kind === "danger" ? "alert" : "status" },
    icon(TOAST_ICON[kind] || "info"), h("div", {}, message));
  host.append(el);
  setTimeout(() => el.remove(), ms);
}

// --- Confirm dialog ---
export function confirmDialog({ title, body, confirmLabel = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    const dlg = h("dialog", { class: "dialog", "aria-labelledby": "dlg-title" },
      h("h3", { id: "dlg-title" }, title),
      h("div", { class: "muted" }, body),
      h("div", { class: "dialog-actions" },
        h("button", { class: "btn", type: "button", onclick: () => dlg.close("cancel") }, "Cancel"),
        h("button", { class: `btn ${danger ? "btn-danger" : "btn-primary"}`, type: "button", onclick: () => dlg.close("ok") }, confirmLabel)));
    dlg.addEventListener("close", () => { resolve(dlg.returnValue === "ok"); dlg.remove(); });
    document.body.append(dlg);
    dlg.showModal();
  });
}

// --- Formatting ---
export function fmtBytes(n) {
  if (n === null || n === undefined) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = n;
  while (v >= 1000 && i < units.length - 1) { v /= 1000; i++; }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "-";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m ${s % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function fmtDate(ts) {
  return new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function progressBar(fraction, { indeterminate = false, label = "Progress" } = {}) {
  const bar = h("div", { class: "bar" });
  const el = h("div", {
    class: `progress${indeterminate ? " indeterminate" : ""}`, role: "progressbar",
    "aria-label": label, "aria-valuemin": 0, "aria-valuemax": 100,
    "aria-valuenow": indeterminate ? undefined : Math.round((fraction || 0) * 100),
  }, bar);
  if (!indeterminate) bar.style.width = `${Math.max(0, Math.min(1, fraction || 0)) * 100}%`;
  return el;
}

export function statusBadge(kind, text, iconName, title) {
  return h("span", { class: `badge badge-${kind}`, title }, icon(iconName || { success: "check", danger: "error", warning: "alert", info: "info", neutral: "box" }[kind] || "info"), text);
}

export function kindBadge(kind) {
  return statusBadge("neutral", kind === "image" ? "Image" : "Video", kind === "image" ? "image" : "generate");
}

export function fmtCount(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}K`;
  return String(n || 0);
}

// Ask the server to restart (start.bat relaunches it), show a blocking notice, and reload once it is back.
export async function restartReel(api) {
  const dlg = h("dialog", { class: "dialog", "aria-labelledby": "rs-title" },
    h("h3", { id: "rs-title" }, "Restarting Reel"),
    h("p", { class: "muted", id: "rs-msg" }, "Reel will close and reopen on the new version. This takes a few seconds (longer if new components need installing)."),
    progressBar(0, { indeterminate: true, label: "Restarting" }));
  dlg.addEventListener("cancel", (e) => e.preventDefault());
  document.body.append(dlg);
  dlg.showModal();
  try {
    const r = await api.post("/api/update/restart");
    if (!r.restarting) {
      dlg.querySelector("#rs-msg").textContent = "Reel was not started with start.bat, so it cannot restart itself. Close this window and run start.bat again.";
      dlg.append(h("div", { class: "dialog-actions" }, h("button", { class: "btn", type: "button", onclick: () => dlg.close() }, "OK")));
      return;
    }
  } catch (e) {
    dlg.close(); dlg.remove(); toast(e.message, "danger", 9000); return;
  }
  let sawDown = false;
  for (let i = 0; i < 90; i++) {
    await new Promise((res) => setTimeout(res, 1000));
    try {
      const ok = (await fetch("/api/health", { cache: "no-store" })).ok;
      if (ok && (sawDown || i > 20)) { location.reload(); return; }
    } catch { sawDown = true; }
  }
  dlg.querySelector("#rs-msg").textContent = "Reel did not come back. Check the console window, or run start.bat again.";
}
