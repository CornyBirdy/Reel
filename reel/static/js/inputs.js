// Pickers for the pictures and videos a model takes as input (edit, start/end frames, replace a person).
// An item is { ref, url, kind, w, h }: `ref` is what the server understands ("in:<id>" for an upload, "clip:<id>" for a
// History item), `w`/`h` are the natural size once the browser has read it.
import { h, icon, clear, toast } from "./ui.js";
import { api } from "./api.js";

const ACCEPT = { image: "image/png,image/jpeg,image/webp", video: "video/mp4,video/webm" };
const NOUN = { image: "picture", video: "video" };

export const itemFromClip = (clip) => ({ ref: `clip:${clip.id}`, url: clip.url, kind: clip.kind });

// Read an item's natural size into item.w / item.h, then call `done`.
export function measure(item, done) {
  if (item.w) return done();
  if (item.kind === "video") {
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => { item.w = v.videoWidth; item.h = v.videoHeight; done(); };
    v.onerror = done;
    v.src = item.url;
  } else {
    const img = new Image();
    img.onload = () => { item.w = img.naturalWidth; item.h = img.naturalHeight; done(); };
    img.onerror = done;
    img.src = item.url;
  }
}

function thumb(item) {
  return item.kind === "video"
    ? h("video", { src: `${item.url}#t=0.1`, preload: "metadata", muted: true, playsinline: true, tabindex: -1 })
    : h("img", { src: item.url, alt: "", loading: "lazy" });
}

// A dialog listing finished History items of one kind; resolves to the chosen items.
async function chooseFromHistory(kind, room) {
  let clips = [];
  try { clips = (await api.get(`/api/history?limit=200&kind=${kind}`)).clips; } catch (e) { toast(e.message, "danger"); return []; }
  clips = clips.filter((c) => c.kind === kind);
  return new Promise((resolve) => {
    const chosen = [];
    const cells = [];
    const grid = h("div", { class: "pick-grid" });
    const count = h("span", { class: "muted" });
    const dlg = h("dialog", { class: "dialog dialog-wide", "aria-labelledby": "pick-title" },
      h("h3", { id: "pick-title" }, `Choose ${room > 1 ? `up to ${room} ` : "a "}${NOUN[kind]}${room > 1 ? "s" : ""} from History`),
      clips.length ? grid : h("div", { class: "empty" }, icon("clock"), h("div", {}, `No ${NOUN[kind]}s in History yet.`)),
      h("div", { class: "dialog-actions" }, count,
        h("button", { class: "btn", type: "button", onclick: () => dlg.close("cancel") }, "Cancel"),
        h("button", { class: "btn btn-primary", type: "button", onclick: () => dlg.close("ok") }, "Use selected")));
    const sync = () => { count.textContent = `${chosen.length} selected`; };
    for (const c of clips) {
      const btn = h("button", { type: "button", class: "batch-thumb pick-thumb", "aria-pressed": "false", title: c.prompt, "aria-label": c.prompt.slice(0, 60),
        onclick: () => {
          const at = chosen.indexOf(c);
          if (at >= 0) chosen.splice(at, 1);
          else if (chosen.length < room) chosen.push(c);
          else if (room === 1) chosen.splice(0, 1, c);
          btn.setAttribute("aria-pressed", String(chosen.includes(c)));
          for (const [other, b] of cells) b.setAttribute("aria-pressed", String(chosen.includes(other)));
          sync();
        } }, thumb(itemFromClip(c)));
      cells.push([c, btn]);
      grid.append(btn);
    }
    dlg.addEventListener("close", () => { resolve(dlg.returnValue === "ok" ? chosen.map(itemFromClip) : []); dlg.remove(); });
    sync();
    document.body.append(dlg);
    dlg.showModal();
  });
}

// One input of type image or video, with up to `rule.max` items.
export function filePicker(name, rule, { initial = [], onChange = () => {} } = {}) {
  const kind = rule.type;
  const max = rule.max || 1;
  let items = initial.slice(0, max);
  const list = h("div", { class: "batch-strip input-items" });
  const status = h("span", { class: "muted field-hint" });
  const file = h("input", { type: "file", accept: ACCEPT[kind], multiple: max > 1 || undefined, hidden: true, "aria-label": `Choose ${NOUN[kind]} file` });
  const addBtn = h("button", { class: "btn", type: "button", onclick: () => file.click() }, icon("upload"), `Add ${NOUN[kind]}`);
  const histBtn = h("button", { class: "btn", type: "button", onclick: async () => {
    const got = await chooseFromHistory(kind, max - items.length);
    if (got.length) add(got);
  } }, icon("clock"), "From History…");
  const zone = h("div", { class: "input-zone stack-8", role: "group", "aria-label": rule.label || name },
    list, h("div", { class: "row-8" }, addBtn, histBtn, status), file);

  function render() {
    clear(list);
    items.forEach((it, i) => list.append(h("div", { class: "batch-thumb input-thumb", title: it.w ? `${it.w} × ${it.h}` : undefined },
      thumb(it),
      h("button", { type: "button", class: "input-remove", title: "Remove", "aria-label": `Remove ${NOUN[kind]} ${i + 1}`, onclick: () => { items.splice(i, 1); changed(); } }, icon("x")))));
    const full = items.length >= max;
    addBtn.disabled = histBtn.disabled = full && max > 1;          // a single slot is replaced by adding another
    status.textContent = max > 1 ? `${items.length} of ${max}` : "";
  }
  function changed() { render(); onChange(items.slice()); }
  function add(more) {
    const room = max - items.length;
    if (more.length > room && max > 1) toast(`Only ${max} allowed: the first ${room} were added.`, "warning", 3500);
    if (max === 1) items = [];
    items.push(...more.slice(0, max === 1 ? 1 : room));
    render();
    let pending = items.length;
    const done = () => { if (--pending <= 0) onChange(items.slice()); };
    for (const it of items) measure(it, done);
  }

  async function upload(files) {
    const wanted = [...files].filter((f) => f.type.startsWith(`${kind}/`));
    if (!wanted.length) { toast(`That is not a ${NOUN[kind]}.`, "warning"); return; }
    status.textContent = "Uploading…";
    try {
      const got = [];
      for (const f of wanted.slice(0, Math.max(max - items.length, max === 1 ? 1 : 0))) {
        const r = await api.upload(f);
        got.push({ ref: r.ref, url: r.url, kind: r.kind });
      }
      add(got);
    } catch (e) { toast(e.message, "danger", 8000); render(); }
  }
  file.addEventListener("change", () => { upload(file.files); file.value = ""; });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.dataset.over = "true"; });
  zone.addEventListener("dragleave", () => { delete zone.dataset.over; });
  zone.addEventListener("drop", (e) => { e.preventDefault(); delete zone.dataset.over; if (e.dataTransfer.files.length) upload(e.dataTransfer.files); });

  render();
  items.forEach((it) => measure(it, () => {}));
  return {
    el: zone,
    items: () => items.slice(),
    refs: () => items.map((i) => i.ref),
  };
}

// "Which person?": the first frame of the video, cropped the way the model will see it; a click records where, as 0..1 of the
// frame. The box takes the aspect ratio of the output size, because the video is centre-cropped to that.
export function pointPicker(name, rule, { initial = null, onChange = () => {} } = {}) {
  let point = initial;
  let src = null;
  const dot = h("span", { class: "point-dot", hidden: true });
  const video = h("video", { muted: true, playsinline: true, preload: "metadata", tabindex: -1 });
  const box = h("div", { class: "point-box", role: "button", tabindex: 0, "aria-label": "Click the person to replace" }, video, dot);
  const clearBtn = h("button", { class: "btn btn-ghost", type: "button", hidden: true, onclick: () => set(null) }, icon("x"), "Clear");
  const hint = h("span", { class: "field-hint" });
  const el = h("div", { class: "stack-8 point-picker", hidden: true }, box, h("div", { class: "row-8" }, clearBtn, hint));

  function set(p) {
    point = p;
    dot.hidden = !p;
    clearBtn.hidden = !p;
    if (p) { dot.style.left = `${p[0] * 100}%`; dot.style.top = `${p[1] * 100}%`; }
    hint.textContent = p ? "That person will be replaced." : (rule.help || "");
    onChange(point);
  }
  box.addEventListener("click", (e) => {
    const r = box.getBoundingClientRect();
    set([Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1), Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1)]);
  });
  hint.textContent = rule.help || "";
  if (point) { dot.hidden = false; clearBtn.hidden = false; dot.style.left = `${point[0] * 100}%`; dot.style.top = `${point[1] * 100}%`; hint.textContent = "That person will be replaced."; }
  return {
    el,
    value: () => point,
    // Show the video's first frame; with no video the picker is hidden.
    setVideo(url) {
      if (url === src) return;
      const kept = src === null && point;      // a click restored with the form belongs to this same video
      src = url;
      el.hidden = !url;
      if (!url) { set(null); return; }
      video.src = `${url}#t=0.1`;
      if (!kept) set(null);
    },
    setAspect(w, hh) { if (w && hh) box.style.aspectRatio = `${w} / ${hh}`; },
  };
}
