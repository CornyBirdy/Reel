import { h, icon, clear } from "./ui.js";
import { getTheme, setTheme, applyTheme } from "./theme.js";
import { api } from "./api.js";

const ROUTES = {
  generate:   { label: "Generate", icon: "generate", load: () => import("./views/generate.js") },
  library:    { label: "Library",  icon: "library",  load: () => import("./views/library.js") },
  settings:   { label: "Settings", icon: "settings", load: () => import("./views/settings.js") },
  styleguide: { label: "Styleguide", hidden: true,    load: () => import("./views/styleguide.js") },
};
const DEFAULT_ROUTE = "generate";

const viewEl = document.getElementById("view");
const navEl = document.getElementById("nav");
let current = null;

function routeFromHash() {
  const name = location.hash.replace(/^#\/?/, "").split("?")[0];
  return ROUTES[name] ? name : DEFAULT_ROUTE;
}

async function render() {
  const name = routeFromHash();
  if (current && current.unmount) { try { current.unmount(); } catch (e) { console.error(e); } }
  current = null;
  for (const a of navEl.querySelectorAll("a")) {
    if (a.dataset.route === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  }
  clear(viewEl);
  viewEl.scrollTop = 0;
  try {
    const mod = await ROUTES[name].load();
    current = (await mod.mount(viewEl)) || {};
  } catch (err) {
    console.error(err);
    viewEl.append(h("div", { class: "view" }, h("div", { class: "notice notice-danger" }, icon("error"), `Could not load this page: ${err.message}`)));
  }
}

function buildNav() {
  for (const [name, r] of Object.entries(ROUTES)) {
    if (r.hidden) continue;
    navEl.append(h("a", { class: "nav-item", href: `#/${name}`, "data-route": name }, icon(r.icon), h("span", { class: "label" }, r.label)));
  }
}

function buildThemeToggle() {
  const host = document.getElementById("theme-toggle");
  const options = [["system", "Auto", "monitor"], ["light", "Light", "sun"], ["dark", "Dark", "moon"]];
  const draw = () => {
    clear(host);
    const mode = getTheme();
    for (const [value, label, ic] of options) {
      host.append(h("button", {
        type: "button", "aria-pressed": String(mode === value), title: `Theme: ${label}`,
        onclick: () => { setTheme(value); draw(); },
      }, icon(ic), h("span", {}, label)));
    }
  };
  draw();
}

// "Update available" pill: a cached check against GitHub (the server caches it for 30 minutes).
const pill = document.getElementById("update-pill");
function showUpdate(u) {
  const available = u && u.state === "available";
  pill.hidden = !available;
  if (available) pill.querySelector("span").textContent = `Update available (${u.behind})`;
}
async function checkForUpdate() {
  try { showUpdate(await api.get("/api/update/check")); } catch { /* offline: no pill */ }
}
document.addEventListener("reel:update-state", (e) => showUpdate(e.detail));

async function refreshGpuChip() {
  const chip = document.getElementById("gpu-chip");
  try {
    const sys = await api.get("/api/system");
    const gpu = sys.gpu.gpus[0];
    clear(chip);
    if (!gpu) {
      chip.append(icon("chip"), h("div", {}, h("strong", {}, "No NVIDIA GPU"), "CPU only"));
    } else {
      const gb = (mb) => (mb / 1024).toFixed(0);
      chip.append(icon("chip"), h("div", {}, h("strong", {}, gpu.name.replace("NVIDIA GeForce ", "")), `${gb(gpu.vram_used_mb)} / ${gb(gpu.vram_total_mb)} GB VRAM`));
    }
  } catch { /* backend unreachable: leave chip empty */ }
}

applyTheme(getTheme());
buildNav();
buildThemeToggle();
window.addEventListener("hashchange", () => { render(); refreshGpuChip(); });
refreshGpuChip();
setInterval(refreshGpuChip, 15000);
checkForUpdate();
setInterval(checkForUpdate, 30 * 60 * 1000);
render();
