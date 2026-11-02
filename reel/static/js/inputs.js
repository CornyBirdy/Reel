// Pickers for the pictures, videos and recordings a model takes as input (edit, start/end frames, replace a person, references).
// An item is { ref, url, kind, w, h }: `ref` is what the server understands ("in:<id>" for an upload, "clip:<id>" for a
// History item), `w`/`h` are the natural size once the browser has read it.
import { h, icon, clear, toast, lazyVideo } from "./ui.js";
import { api } from "./api.js";

const ACCEPT = { image: "image/png,image/jpeg,image/webp", video: "video/mp4,video/webm", audio: "audio/wav,audio/mpeg,audio/flac,audio/ogg,audio/mp4,audio/x-m4a,.wav,.mp3,.flac,.ogg,.m4a" };
const NOUN = { image: "picture", video: "video", audio: "recording" };
const TAG = { image: "Picture", video: "Video", audio: "Audio" };        // how H3's prompt guide names references: <Picture 1>, <Video 1>, <Audio 1>
export const kindsOf = (rule) => rule.type === "media" ? Object.keys(rule.max_by_kind || { image: 9, video: 3, audio: 3 }) : [rule.type];

export const itemFromClip = (clip) => ({ ref: `clip:${clip.id}`, url: clip.url, kind: clip.kind });

// Read an item's natural size into item.w / item.h, then call `done`.
export function measure(item, done) {
  if (item.w || item.kind === "audio") return done();
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
  if (item.kind === "audio") return h("div", { class: "audio-tile", title: item.name || "Recording" }, icon("music"));
  return item.kind === "video"
    ? lazyVideo(item.url)
    : h("img", { src: item.url, alt: "", loading: "lazy" });
}

// A dialog listing finished History items of one kind; resolves to the chosen items.
async function chooseFromHistory(kinds, room) {
  let clips = [];
  const kind = kinds.length === 1 ? kinds[0] : "item";
  try { clips = (await api.get(`/api/history?limit=200${kinds.length === 1 ? `&kind=${kinds[0]}` : ""}`)).clips; } catch (e) { toast(e.message, "danger"); return []; }
  clips = clips.filter((c) => kinds.includes(c.kind));
  return new Promise((resolve) => {
    const chosen = [];
    const cells = [];
    const grid = h("div", { class: "pick-grid" });
    const count = h("span", { class: "muted" });
    const dlg = h("dialog", { class: "dialog dialog-wide", "aria-labelledby": "pick-title" },
      h("h3", { id: "pick-title" }, `Choose ${room > 1 ? `up to ${room} ` : "a "}${NOUN[kind] || "picture or video"}${room > 1 ? "s" : ""} from History`),
      clips.length ? grid : h("div", { class: "empty" }, icon("clock"), h("div", {}, `Nothing in History yet.`)),
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

// A flag: one checkbox.
export function flagPicker(name, rule, { initial = false, onChange = () => {} } = {}) {
  const box = h("input", { type: "checkbox", id: `flag-${name}`, checked: !!initial, onchange: () => onChange(box.checked) });
  return { el: h("label", { class: "check" }, box, rule.label || name), value: () => box.checked };
}

// One input of type image, video, audio, or "media" (an ordered mix of them), with up to `rule.max` items.
// With more than one slot the items are numbered, can be moved up and down the order, and a click on a number calls onInsert
// (the form puts "picture 2" or "<Picture 2>" into the prompt), so several pictures can be mixed by name.
export function filePicker(name, rule, { initial = [], onChange = () => {}, onInsert = null } = {}) {
  const kinds = kindsOf(rule);
  const mixed = rule.type === "media";
  const max = rule.max || 1;
  const perKind = rule.max_by_kind || {};
  const numbered = max > 1;
  const noun = mixed ? "item" : NOUN[rule.type];
  let items = initial.filter((i) => kinds.includes(i.kind)).slice(0, max);
  const list = h("div", { class: "batch-strip input-items" });
  const status = h("span", { class: "muted field-hint" });
  const accept = kinds.map((k) => ACCEPT[k]).join(",");
  const file = h("input", { type: "file", accept, multiple: max > 1 || undefined, hidden: true, "aria-label": `Choose ${noun} file` });
  const addBtn = h("button", { class: "btn", type: "button", onclick: () => file.click() }, icon("upload"), mixed ? "Add files" : `Add ${noun}`);
  const histKinds = kinds.filter((k) => k !== "audio");
  const histBtn = histKinds.length ? h("button", { class: "btn", type: "button", onclick: async () => {
    const got = await chooseFromHistory(histKinds, max - items.length);
    if (got.length) add(got);
  } }, icon("clock"), "From History…") : null;
  const zone = h("div", { class: "input-zone stack-8", role: "group", "aria-label": rule.label || name },
    list, h("div", { class: "row-8" }, addBtn, histBtn, status), file);

  const countOf = (k, upTo = items.length) => items.slice(0, upTo).filter((i) => i.kind === k).length;
  const tagFor = (i) => mixed ? `<${TAG[items[i].kind]} ${countOf(items[i].kind, i + 1)}>` : `${NOUN[items[i].kind]} ${i + 1}`;
  const roomFor = (k) => Math.min(max - items.length, perKind[k] === undefined ? max : perKind[k] - countOf(k));

  function render() {
    clear(list);
    items.forEach((it, i) => {
      const label = numbered ? (mixed ? tagFor(i).replace(/[<>]/g, "") : String(i + 1)) : null;
      list.append(h("div", { class: "batch-thumb input-thumb" + (it.kind === "audio" ? " is-audio" : ""), title: it.w ? `${it.w} × ${it.h}` : it.name || undefined },
        thumb(it),
        label ? h("button", { type: "button", class: "input-tag", disabled: !onInsert || undefined,
          title: onInsert ? `Put "${tagFor(i)}" in the prompt` : label, "aria-label": `Refer to ${tagFor(i)} in the prompt`,
          onclick: () => onInsert && onInsert(tagFor(i)) }, label) : null,
        h("button", { type: "button", class: "input-remove", title: "Remove", "aria-label": `Remove ${noun} ${i + 1}`, onclick: () => { items.splice(i, 1); changed(); } }, icon("x")),
        numbered && items.length > 1 ? h("span", { class: "input-move" },
          h("button", { type: "button", title: "Move earlier", "aria-label": "Move earlier", disabled: i === 0 || undefined, onclick: () => move(i, -1) }, "\u2039"),
          h("button", { type: "button", title: "Move later", "aria-label": "Move later", disabled: i === items.length - 1 || undefined, onclick: () => move(i, 1) }, "\u203a")) : null));
    });
    const full = items.length >= max;
    addBtn.disabled = full && max > 1;
    if (histBtn) histBtn.disabled = full && max > 1;          // a single slot is replaced by adding another
    status.textContent = max > 1 ? `${items.length} of ${max}${mixed ? ` · ${kinds.map((k) => `${countOf(k)}/${perKind[k]} ${NOUN[k]}s`).join(", ")}` : ""}` : "";
  }
  function changed() { render(); onChange(items.slice()); }
  function move(i, by) { const j = i + by; if (j < 0 || j >= items.length) return; [items[i], items[j]] = [items[j], items[i]]; changed(); }
  function add(more) {
    if (max === 1) items = [];
    let skipped = 0;
    for (const it of more) {
      if (max === 1 ? items.length >= 1 : roomFor(it.kind) <= 0) { skipped++; continue; }
      items.push(it);
    }
    if (skipped) toast(`${skipped === 1 ? "One was" : `${skipped} were`} left out: ${mixed ? "the limit for that kind is reached" : `only ${max} allowed`}.`, "warning", 3500);
    render();
    let pending = items.length;
    const done = () => { if (--pending <= 0) onChange(items.slice()); };
    if (!pending) onChange(items.slice());
    for (const it of items) measure(it, done);
  }

  async function upload(files) {
    const typeOk = (f) => kinds.some((k) => f.type.startsWith(`${k}/`) || (k === "audio" && /\.(wav|mp3|flac|ogg|m4a)$/i.test(f.name)));
    const wanted = [...files].filter(typeOk);
    if (!wanted.length) { toast(`That is not a ${mixed ? "picture, video or recording" : noun}.`, "warning"); return; }
    status.textContent = "Uploading…";
    try {
      const got = [];
      const taken = { image: countOf("image"), video: countOf("video"), audio: countOf("audio") };
      for (const f of wanted) {
        if (max > 1 && (items.length + got.length >= max)) break;
        if (max === 1 && got.length >= 1) break;
        const r = await api.upload(f);
        if (max > 1 && perKind[r.kind] !== undefined && taken[r.kind] + got.filter((g) => g.kind === r.kind).length >= perKind[r.kind]) continue;
        got.push({ ref: r.ref, url: r.url, kind: r.kind, name: f.name });
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
    add: (one) => add([one]),
  };
}

// The frame of a playing or paused video, as a PNG upload (a reference for Start frame, Reference, ...). The browser draws it,
// so no tool is needed on the PC. `where` is "current" or "last".
export async function captureFrame(videoUrl, where = "current", at = null) {
  const v = document.createElement("video");
  v.muted = true; v.preload = "auto"; v.crossOrigin = "anonymous"; v.src = videoUrl;
  await new Promise((ok, no) => { v.onloadedmetadata = ok; v.onerror = () => no(new Error("The browser could not read that video.")); });
  const target = where === "last" ? Math.max(v.duration - 0.05, 0) : (at ?? 0);
  await new Promise((ok) => { v.onseeked = ok; v.currentTime = target; });
  const c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  const blob = await new Promise((ok) => c.toBlob(ok, "image/png"));
  if (!blob) throw new Error("Could not take the frame.");
  const r = await api.upload(new File([blob], "frame.png", { type: "image/png" }));
  return { ref: r.ref, url: r.url, kind: "image", w: c.width, h: c.height };
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
