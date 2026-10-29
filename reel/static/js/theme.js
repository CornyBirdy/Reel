// Theme: follows the OS by default; manual choice persists in localStorage.
const KEY = "reel.theme";

export function getTheme() {
  try { return localStorage.getItem(KEY) || "system"; } catch { return "system"; }
}

// Browser chrome colour follows the active theme. Read from the --ground token
// (resolved through a probe element) so no colour literal lives outside tokens.css.
export function syncThemeColor() {
  const meta = document.querySelector('meta[name="theme-color"]');
  if (!meta || !document.body) return;
  const probe = document.createElement("i");
  probe.style.cssText = "display:none;background:var(--ground)";
  document.body.append(probe);
  meta.content = getComputedStyle(probe).backgroundColor;
  probe.remove();
}

export function applyTheme(mode) {
  const root = document.documentElement;
  if (mode === "light" || mode === "dark") root.setAttribute("data-theme", mode);
  else root.removeAttribute("data-theme");
  syncThemeColor();
}

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncThemeColor);

export function setTheme(mode) {
  try {
    if (mode === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, mode);
  } catch { /* storage blocked: the choice just won't persist */ }
  applyTheme(mode);
}
