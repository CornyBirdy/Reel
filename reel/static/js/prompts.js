// Prompt helpers: snippets to insert, and a list of saved prompts. Nothing here talks to a model; it is a notebook next to the prompt box.
import { h, icon, clear, toast } from "./ui.js";

const KEY = "reel.prompts";
const MAX_SAVED = 100;

const CAMERA = [
  ["Camera: slow push in", "The camera slowly pushes in."],
  ["Camera: handheld", "Handheld camera, slight natural shake."],
  ["Camera: drone orbit", "A drone shot orbits the subject."],
  ["Camera: tracking shot", "A smooth tracking shot follows the subject."],
  ["Lens: shallow depth of field", "Shallow depth of field, soft background."],
];
const LOOK = [
  ["Light: golden hour", "Warm golden-hour light, long soft shadows."],
  ["Light: neon night", "Night street lit by neon signs, reflections on wet ground."],
  ["Style: film look", "Cinematic colour grading, fine film grain."],
];
const SOUND = [
  ["Sound: ambience", "Ambient sound of the scene: footsteps, distant traffic, wind."],
  ["Sound: dialogue", "A character says, <d>Hello there.</d>"],
  ["Sound: music", "Upbeat background music with a steady beat."],
];
const EDIT = [
  ["Keep everything else", "Keep everything else exactly the same."],
  ["Match lighting", "Match the lighting and shadows of the original."],
  ["Use picture 2 for…", "Use the outfit from picture 2."],
];

// [label, text] pairs for the Prompt tools menu: what suits the current mode.
export function promptIdeas(kind, mode) {
  if (mode === "edit") return EDIT;
  if (kind === "image") return LOOK;
  return [...CAMERA, ...LOOK, ...(mode === "reference" || mode === "video" || mode === "frames" ? SOUND : [])];
}

export function loadSaved() {
  try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; }
}
function store(list) { try { localStorage.setItem(KEY, JSON.stringify(list.slice(0, MAX_SAVED))); } catch { /* storage may be unavailable */ } }

export function addSaved(text) {
  const t = text.trim();
  if (!t) return;
  store([{ text: t, at: Date.now() }, ...loadSaved().filter((p) => p.text !== t)]);
}

// A dialog with the saved prompts: click one to use it.
export function openSavedPrompts({ current, onUse }) {
  const list = h("ul", { class: "saved-list" });
  const dlg = h("dialog", { class: "dialog dialog-wide", "aria-labelledby": "sp-title" },
    h("h3", { id: "sp-title" }, "Saved prompts"),
    list,
    h("div", { class: "dialog-actions" },
      h("button", { class: "btn", type: "button", onclick: () => { addSaved(current()); render(); } }, icon("bookmark"), "Save the current prompt"),
      h("span", { class: "grow" }),
      h("button", { class: "btn btn-primary", type: "button", onclick: () => dlg.close() }, "Done")));
  function render() {
    clear(list);
    const items = loadSaved();
    if (!items.length) { list.append(h("li", { class: "muted" }, "Nothing saved yet. Write a prompt and press “Save the current prompt”.")); return; }
    items.forEach((p, i) => list.append(h("li", { class: "saved-row" },
      h("button", { type: "button", class: "saved-text", title: "Use this prompt", onclick: () => { onUse(p.text); dlg.close(); } }, p.text),
      h("button", { type: "button", class: "btn btn-ghost btn-icon", title: "Delete", "aria-label": "Delete saved prompt",
        onclick: () => { const all = loadSaved(); all.splice(i, 1); store(all); render(); toast("Removed.", "info", 1200); } }, icon("trash")))));
  }
  dlg.addEventListener("close", () => dlg.remove());
  render();
  document.body.append(dlg);
  dlg.showModal();
}
