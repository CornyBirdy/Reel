import { h, icon, clear, toast } from "./ui.js";
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

const shortGpu = (name) => name.replace(/^NVIDIA\s+/, "").replace(/^GeForce\s+/, "");
// What "Automatic" resolves to: the GPU with the most VRAM (same rule as the server).
const autoGpu = (gpus) => [...gpus].sort((a, b) => b.vram_total_mb - a.vram_total_mb || a.index - b.index)[0];
let gpuKey = "";
let gpuLine = null;

async function refreshGpuChip() {
  const chip = document.getElementById("gpu-chip");
  let sys;
  try { sys = await api.get("/api/system"); } catch { return; }
  const gpus = sys.gpu.gpus, sel = sys.gpu.selected_gpu;
  const gb = (mb) => (mb / 1024).toFixed(0);
  // Rebuild only when the GPU list or the choice changes: replacing an open <select> would close its menu.
  const key = JSON.stringify([gpus.map((g) => g.uuid), sys.gpu.selected, sys.gpu.auto]);
  if (key !== gpuKey) {
    gpuKey = key;
    clear(chip);
    if (!sel) {
      chip.append(icon("chip"), h("div", {}, h("strong", {}, "No NVIDIA GPU"), "CPU only"));
      gpuLine = null;
    } else {
      gpuLine = h("span");
      const body = h("div", { class: "grow" });
      if (gpus.length > 1) {
        const pick = h("select", { class: "select gpu-select", "aria-label": "GPU to use", onchange: async () => {
          try {
            await api.put("/api/settings", { set_gpu: true, gpu: pick.value === "auto" ? "" : pick.value });
            toast("GPU changed. It applies to the next generation.", "success", 4000);
          } catch (e) { toast(e.message, "danger", 8000); }
          refreshGpuChip();
        } },
          h("option", { value: "auto", selected: sys.gpu.auto }, `Automatic (${shortGpu(autoGpu(gpus).name)})`),
          gpus.map((g) => h("option", { value: g.uuid, selected: !sys.gpu.auto && g.uuid === sys.gpu.selected }, `${shortGpu(g.name)} · ${gb(g.vram_total_mb)} GB`)));
        body.append(pick, h("div", {}, gpuLine));
      } else {
        body.append(h("strong", {}, shortGpu(sel.name)), gpuLine);
      }
      chip.append(icon("chip"), body);
    }
  }
  if (gpuLine && sel) gpuLine.textContent = `${gb(sel.vram_used_mb)} / ${gb(sel.vram_total_mb)} GB VRAM`;
}

// Coming back from minimised (or another window): WebView2 can show a blank page until something forces a
// repaint, most often while a download keeps the page busy. Force a reflow, then tell the views to refresh now
// instead of waiting for their next timer.
function wake() {
  if (document.hidden) return;
  const body = document.body;
  body.style.display = "none"; void body.offsetHeight; body.style.display = "";
  document.dispatchEvent(new CustomEvent("reel:wake"));
  refreshGpuChip();
}
document.addEventListener("visibilitychange", wake);
window.addEventListener("focus", wake);
window.addEventListener("pageshow", wake);
window.addEventListener("resize", () => { if (!document.hidden) document.dispatchEvent(new CustomEvent("reel:wake")); });

applyTheme(getTheme());
buildNav();
buildThemeToggle();
window.addEventListener("hashchange", () => { render(); refreshGpuChip(); });
refreshGpuChip();
setInterval(refreshGpuChip, 15000);
checkForUpdate();
setInterval(checkForUpdate, 30 * 60 * 1000);
render();
