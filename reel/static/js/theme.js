// Theme: follows the OS by default; manual choice persists in localStorage.
const KEY = "reel.theme";

export function getTheme() {
  try { return localStorage.getItem(KEY) || "system"; } catch { return "system"; }
}

export function applyTheme(mode) {
  const root = document.documentElement;
  if (mode === "light" || mode === "dark") root.setAttribute("data-theme", mode);
  else root.removeAttribute("data-theme");
}

export function setTheme(mode) {
  try {
    if (mode === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, mode);
  } catch { /* storage blocked: the choice just won't persist */ }
  applyTheme(mode);
}
