import { h, icon, put, clear, toast, confirmDialog, progressBar, statusBadge, fmtBytes, fmtAgo, fmtTokens, copyText } from "../ui.js";
import { api } from "../api.js";
import { renderMarkdown, splitThinking } from "../markdown.js";
import { openLoadDialog, modelTitle } from "./llm_load.js";

const LAST_CHAT = "reel.chat.last";
const LAST_MODEL = "reel.chat.model";
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* ignore */ } },
};
const DEFAULT_PARAMS = { temperature: 0.7, top_p: 0.95, max_tokens: null };

export async function mount(root) {
  root.classList.add("main-fill");
  const q = new URLSearchParams(location.hash.split("?")[1] || "");

  let ov = await api.get("/api/llm");                 // engine, server, models, gpu
  let chatList = (await api.get("/api/chats")).chats;
  let chat = null;                                     // the open conversation (null = a new one that is not saved yet)
  let draftModel = null;
  let alive = true, timer = null;
  let stream = null;                                   // the reply being written: { ac, content, reasoning, el, ... }
  let sideOpen = store.get("reel.chat.side") === "1";
  let draftSystem = "";
  let draftParams = { ...DEFAULT_PARAMS };
  let ctxUsed = 0;
  let lastError = null;                                // shown under the conversation until the next action

  const ready = () => ov.models.filter((m) => m.state === "ready");
  const modelById = (id) => ov.models.find((m) => m.id === id);
  const srv = () => ov.server;
  const loadedId = () => (srv().state === "ready" || srv().state === "loading" ? srv().model_id : null);
  const currentModelId = () => (chat && chat.model_id && modelById(chat.model_id) ? chat.model_id : draftModel);
  const params = () => ({ ...DEFAULT_PARAMS, ...(chat ? chat.params : draftParams) });
  const systemPrompt = () => (chat ? chat.system : draftSystem);

  // ---------------------------------------------------------------- layout
  const listEl = h("nav", { class: "chat-list", "aria-label": "Conversations" });
  const headEl = h("header", { class: "chat-head" });
  const logEl = h("div", { class: "chat-log", role: "log", "aria-label": "Conversation", tabindex: 0 });
  const composerEl = h("div", { class: "chat-composer-wrap" });
  const sideEl = h("aside", { class: "chat-side", "aria-label": "Chat settings" });
  const layout = h("div", { class: "chat" },
    listEl,
    h("section", { class: "chat-main" }, headEl, logEl, composerEl),
    sideEl);
  root.append(layout);
  layout.classList.toggle("side-open", sideOpen);

  // ---------------------------------------------------------------- data refresh
  async function refreshStatus() {
    try { ov = await api.get("/api/llm"); } catch { return; }
    if (!alive) return;
    drawHead(); drawComposer(); drawSide();
    document.dispatchEvent(new CustomEvent("reel:llm-state", { detail: ov.server }));
    schedule();
  }
  function schedule() {
    clearTimeout(timer);
    const busy = srv().state === "loading" || (ov.engine.job && ov.engine.job.status === "running");
    if (alive) timer = setTimeout(refreshStatus, busy ? 800 : 6000);
  }
  async function reloadChats() { chatList = (await api.get("/api/chats")).chats; drawList(); }

  // ---------------------------------------------------------------- conversation list
  function drawList() {
    clear(listEl);
    listEl.append(
      h("div", { class: "chat-list-head" }, h("h2", {}, "Chats"),
        h("button", { class: "btn btn-primary", type: "button", onclick: () => { newChat(); layout.classList.remove("list-open"); } }, icon("plus"), "New chat")));
    if (!chatList.length) {
      listEl.append(h("p", { class: "muted chat-list-empty" }, "Your conversations are saved here, on this PC."));
      return;
    }
    const ul = h("ul", { class: "chat-items" });
    for (const c of chatList) {
      const active = chat && chat.id === c.id;
      ul.append(h("li", {},
        h("div", { class: "item chat-item", "aria-current": active ? "true" : undefined },
          h("button", { type: "button", class: "chat-item-open", onclick: () => { openChat(c.id); layout.classList.remove("list-open"); } },
            h("span", { class: "chat-item-title" }, c.title), h("span", { class: "chat-item-sub muted" }, fmtAgo(c.updated))),
          h("button", { type: "button", class: "btn btn-ghost btn-icon chat-item-del", title: "Delete chat", "aria-label": `Delete ${c.title}`, disabled: !!stream,
            onclick: () => removeChat(c) }, icon("trash")))));
    }
    listEl.append(ul);
  }

  async function removeChat(c) {
    const ok = await confirmDialog({ title: "Delete this chat?", body: `"${c.title}" and its messages will be deleted from this PC.`, confirmLabel: "Delete", danger: true });
    if (!ok) return;
    try { await api.del(`/api/chats/${c.id}`); } catch (e) { toast(e.message, "danger"); return; }
    if (chat && chat.id === c.id) newChat();
    await reloadChats();
  }

  async function openChat(id) {
    if (stream) return;
    try { chat = await api.get(`/api/chats/${id}`); } catch (e) { toast(e.message, "danger"); return; }
    store.set(LAST_CHAT, id);
    ctxUsed = 0; lastError = null;
    drawAll();
    scrollDown(true);
  }

  function newChat(modelId) {
    if (stream) return;
    chat = null; lastError = null;
    draftModel = modelId || currentModelId() || loadedId() || store.get(LAST_MODEL);
    if (!modelById(draftModel)) draftModel = (ready()[0] || {}).id || null;
    draftSystem = ""; draftParams = { ...DEFAULT_PARAMS };
    ctxUsed = 0;
    store.set(LAST_CHAT, null);
    drawAll();
    composerInput()?.focus();
  }

  // ---------------------------------------------------------------- header: model + load state
  function drawHead() {
    clear(headEl);
    const models = ready();
    const id = currentModelId();
    const m = modelById(id);
    const loaded = loadedId();
    const isLoadedHere = loaded && loaded === id;
    const s = srv();

    const pick = h("select", { class: "select chat-model", "aria-label": "Model", disabled: !!stream || !models.length || undefined,
      onchange: () => setModel(pick.value) },
      models.length ? models.map((x) => h("option", { value: x.id, selected: x.id === id }, `${modelTitle(x)}${x.loaded ? "  ●" : ""}`))
        : h("option", {}, "No models downloaded"));

    let state;
    if (s.state === "loading" && isLoadedHere) state = statusBadge("info", "Loading…", "clock");
    else if (s.state === "error" && s.model_id === id) state = statusBadge("danger", "Failed to load", "error", s.error);
    else if (isLoadedHere) state = statusBadge("success", `Loaded · ${fmtTokens(s.options.context)} context`, "check", loadedWhere(s));
    else state = statusBadge("neutral", "Not loaded", "box");

    const actions = h("div", { class: "row-8" });
    if (m) {
      if (isLoadedHere && s.state === "ready") {
        actions.append(h("button", { class: "btn", type: "button", disabled: !!stream || undefined, onclick: unload }, icon("eject"), "Eject"));
      } else if (!(isLoadedHere && s.state === "loading")) {
        actions.append(h("button", { class: "btn", type: "button", disabled: !!stream || undefined, onclick: () => loadWithDialog(m) }, icon("load"), "Load…"));
      }
    }
    headEl.append(
      h("button", { class: "btn btn-icon chat-toggle", type: "button", title: "Conversations", "aria-label": "Show conversations", onclick: () => layout.classList.toggle("list-open") }, icon("list")),
      pick, state, h("div", { class: "grow" }), actions,
      h("button", { class: "btn btn-icon", type: "button", title: "Chat settings", "aria-label": "Chat settings", "aria-pressed": String(sideOpen),
        onclick: () => { sideOpen = !sideOpen; store.set("reel.chat.side", sideOpen ? "1" : "0"); layout.classList.toggle("side-open", sideOpen); drawHead(); } }, icon("sliders")));
  }

  function loadedWhere(s) {
    const o = s.options || {};
    return o.gpu_layers === 0 ? "On the CPU" : `${o.gpu_layers >= 999 ? "All layers" : o.gpu_layers + " layers"} on the GPU`;
  }

  async function setModel(id) {
    store.set(LAST_MODEL, id);
    if (chat) { chat = await api.patch(`/api/chats/${chat.id}`, { model_id: id }); reloadChats(); } else draftModel = id;
    drawHead(); drawComposer();
  }

  async function loadWithDialog(m) {
    const r = await openLoadDialog(m, ov);
    if (r) { toast(`Loading ${modelTitle(m)}…`, "info", 2500); await refreshStatus(); }
  }

  async function unload() {
    try { await api.post("/api/llm/unload"); } catch (e) { toast(e.message, "danger"); }
    await refreshStatus();
  }

  // Make sure the selected model is running. Resolves true when it is ready to answer.
  async function ensureLoaded() {
    const id = currentModelId();
    if (srv().state === "ready" && srv().model_id === id) return true;
    try {
      if (!(srv().state === "loading" && srv().model_id === id)) await api.post("/api/llm/load", { model_id: id });
    } catch (e) { toast(e.message, "danger", 9000); return false; }
    for (let i = 0; i < 1500 && alive; i++) {      // up to ~20 minutes: a big model from a slow disk
      await refreshStatus();
      if (srv().state === "ready" && srv().model_id === id) return true;
      if (srv().state === "error") { toast(srv().error || "The model could not be loaded.", "danger", 12000); return false; }
      if (srv().state === "idle") return false;
      await new Promise((r) => setTimeout(r, 700));
    }
    return false;
  }

  // ---------------------------------------------------------------- messages
  function scrollDown(force) {
    const near = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 120;
    if (force || near) logEl.scrollTop = logEl.scrollHeight;
  }

  function bubble(msg, { last, pending } = {}) {
    const isUser = msg.role === "user";
    const body = h("div", { class: "msg-body" });
    fillBody(body, msg.content, msg.reasoning, pending, isUser);
    const tools = h("div", { class: "msg-tools" },
      h("button", { type: "button", class: "btn btn-ghost btn-icon", title: "Copy", "aria-label": "Copy message", onclick: () => copyText(msg.content) }, icon("copy")),
      isUser && !stream ? h("button", { type: "button", class: "btn btn-ghost btn-icon", title: "Edit and resend", "aria-label": "Edit and resend", onclick: () => editMessage(msg) }, icon("edit")) : null,
      !isUser && last && !stream ? h("button", { type: "button", class: "btn btn-ghost btn-icon", title: "Regenerate", "aria-label": "Regenerate", onclick: () => regenerate(msg) }, icon("rerun")) : null,
      !stream && msg.id ? h("button", { type: "button", class: "btn btn-ghost btn-icon", title: "Delete this and everything after it", "aria-label": "Delete from here", onclick: () => deleteFrom(msg) }, icon("trash")) : null);
    const st = msg.stats || {};
    const meta = !isUser && st.tps ? h("div", { class: "msg-meta muted" },
      [`${st.tps.toFixed(1)} tok/s`, st.tokens ? `${st.tokens} tokens` : null, st.ttft != null ? `${st.ttft.toFixed(2)} s to first token` : null, st.stopped ? "stopped" : null, st.truncated ? "hit the length limit" : null]
        .filter(Boolean).join(" · ")) : null;
    return h("article", { class: `msg msg-${msg.role}${pending ? " msg-pending" : ""}`, "data-id": msg.id ?? "" },
      h("div", { class: "msg-role" }, isUser ? "You" : "Assistant"), body, meta, pending ? null : tools);
  }

  function fillBody(body, content, reasoning, pending, isUser) {
    clear(body);
    if (reasoning) {
      const d = h("details", { class: "think", open: pending && !content });
      d.append(h("summary", {}, icon("brain"), pending && !content ? "Thinking…" : "Thought process"), h("div", { class: "think-body muted" }, renderMarkdown(reasoning)));
      body.append(d);
    }
    if (isUser) body.append(h("div", { class: "msg-text" }, content));
    else if (content) body.append(renderMarkdown(content));
    else if (pending && !reasoning) body.append(h("div", { class: "typing", "aria-label": "Waiting for the model" }, h("span"), h("span"), h("span")));
  }

  function drawLog() {
    clear(logEl);
    const msgs = chat ? chat.messages : [];
    if (!msgs.length && !stream) {
      logEl.append(emptyState());
      if (lastError) logEl.append(streamError(lastError));
      return;
    }
    const lastAssistant = [...msgs].reverse().find((m) => m.role === "assistant");
    msgs.forEach((m) => logEl.append(bubble(m, { last: lastAssistant && lastAssistant.id === m.id && msgs[msgs.length - 1] === m })));
    if (stream) { stream.el = bubble({ role: "assistant", content: stream.content, reasoning: stream.reasoning }, { pending: true }); logEl.append(stream.el); }
    if (lastError) logEl.append(streamError(lastError));
  }

  function emptyState() {
    if (!ready().length) {
      return h("div", { class: "empty chat-empty" }, icon("chat"),
        h("div", {}, h("strong", {}, "No language model yet"), h("div", {}, "Download one from Hugging Face, like LM Studio's Discover tab. A small one (4B to 8B) is a good first try.")),
        h("a", { class: "btn btn-primary", href: "#/library?filter=language&add=1" }, icon("plus"), "Find a model"));
    }
    const m = modelById(currentModelId());
    return h("div", { class: "empty chat-empty" }, icon("chat"),
      h("div", {}, h("strong", {}, m ? `Chat with ${modelTitle(m)}` : "Start a conversation"),
        h("div", {}, "Everything stays on this PC. The model loads when you send your first message.")));
  }

  function streamError(err) {
    const tips = /context|exceed|too long/i.test(err) ? " Start a new chat, or load the model again with a longer context (Load…)." : "";
    return h("div", { class: "notice notice-danger", role: "alert" }, icon("error"), h("div", { class: "stack-8" }, h("div", {}, err + tips)));
  }

  // ---------------------------------------------------------------- composer
  const composerInput = () => composerEl.querySelector("textarea");

  function drawComposer() {
    const keep = composerInput()?.value || "";
    clear(composerEl);
    const e = ov.engine;
    if (!e.installed) { composerEl.append(engineCard()); return; }
    if (!ready().length) { return; }
    const ta = h("textarea", { class: "textarea chat-input", rows: 1, placeholder: "Message the model…", "aria-label": "Message", maxlength: 60000,
      oninput: () => { autosize(ta); drawFoot(); },
      onkeydown: (ev) => { if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); submit(); } } });
    ta.value = keep;
    const send = stream
      ? h("button", { class: "btn btn-primary chat-send", type: "button", onclick: stopStream, "aria-label": "Stop" }, icon("stop"), "Stop")
      : h("button", { class: "btn btn-primary chat-send", type: "button", onclick: submit, "aria-label": "Send" }, icon("send"), "Send");
    const foot = h("div", { class: "chat-foot muted", id: "chat-foot" });
    composerEl.append(h("form", { class: "chat-composer", onsubmit: (ev) => { ev.preventDefault(); submit(); } }, ta, send), foot);
    autosize(ta);
    drawFoot();
  }

  function drawFoot() {
    const foot = document.getElementById("chat-foot");
    if (!foot) return;
    const s = srv();
    const ctx = s.state === "ready" ? s.options.context : null;
    foot.textContent = [
      "Enter to send · Shift+Enter for a new line",
      ctx && ctxUsed ? `Context ${fmtTokens(ctxUsed)} of ${fmtTokens(ctx)} used` : null,
    ].filter(Boolean).join("   ·   ");
    foot.classList.toggle("ctx-high", !!(ctx && ctxUsed / ctx > 0.85));
  }

  function autosize(ta) { ta.style.height = "auto"; ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`; }

  function engineCard() {
    const job = ov.engine.job;
    const running = job && job.status === "running";
    const btn = h("button", { class: "btn btn-primary", type: "button", disabled: !!running, onclick: async () => {
      try { await api.post("/api/llm/engine/install"); } catch (e) { toast(e.message, "danger", 9000); }
      refreshStatus();
    } }, icon("download"), running ? "Installing…" : "Install chat engine");
    return h("div", { class: "card stack-12 engine-card" },
      h("div", { class: "row-between" }, h("h3", {}, "One-time setup: the chat engine"), statusBadge("warning", "Not installed", "alert")),
      h("p", { class: "muted" }, "Language models run on llama.cpp, an open-source engine. Reel downloads the right build for your graphics card from its official GitHub releases (a few hundred MB) into its own folder. Nothing is installed system-wide."),
      running ? h("div", { class: "stack-8" }, progressBar(job.progress, { indeterminate: !job.bytes_total, label: "Engine download" }), h("div", { class: "progress-meta" }, h("span", {}, job.message),
        h("span", {}, job.bytes_total ? `${fmtBytes(job.bytes_done)} of ${fmtBytes(job.bytes_total)}` : ""))) : null,
      job && job.status === "failed" ? h("div", { class: "notice notice-danger", role: "alert" }, icon("error"), job.error) : null,
      h("div", { class: "row-8" }, btn));
  }

  // ---------------------------------------------------------------- sending and streaming
  async function submit() {
    const ta = composerInput();
    if (!ta || stream) return;
    const text = ta.value.trim();
    if (!text) return;
    if (!ov.engine.installed) { toast("Install the chat engine first.", "warning"); return; }
    const id = currentModelId();
    if (!id) { toast("Pick a model first.", "warning"); return; }
    ta.value = ""; autosize(ta);
    try {
      if (!chat) {
        chat = await api.post("/api/chats", { model_id: id, system: draftSystem, params: draftParams });
        store.set(LAST_CHAT, chat.id);
      }
      const saved = await api.post(`/api/chats/${chat.id}/messages`, { role: "user", content: text });
      chat.messages.push(saved);
    } catch (e) { toast(e.message, "danger"); ta.value = text; return; }
    reloadChats();
    await generate();
  }

  async function generate() {
    const id = currentModelId();
    lastError = null;
    stream = { content: "", reasoning: "", error: null, ac: new AbortController(), el: null };
    drawAll();
    scrollDown(true);
    if (!(await ensureLoaded())) { stream = null; lastError = srv().error || "The model could not be loaded."; drawAll(); return; }

    const p = params();
    const messages = [];
    if (systemPrompt().trim()) messages.push({ role: "system", content: systemPrompt() });
    for (const m of chat.messages) messages.push({ role: m.role, content: m.content });
    const body = { model: id, messages, stream: true, stream_options: { include_usage: true }, temperature: p.temperature, top_p: p.top_p };
    if (p.max_tokens) body.max_tokens = p.max_tokens;

    const t0 = performance.now();
    let first = null, tokens = 0, usage = null, timings = null, finish = null;
    let raf = 0;
    const paint = () => {
      raf = 0;
      if (!stream || !stream.el) return;
      const split = splitThinking(stream.content);
      const reasoning = stream.reasoning || split.reasoning;
      fillBody(stream.el.querySelector(".msg-body"), split.answer, reasoning, true, false);
      scrollDown(false);
    };
    try {
      const res = await fetch("/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: stream.ac.signal });
      if (!res.ok) {
        let msg = res.statusText;
        try { const j = await res.json(); msg = j.error?.message || j.detail || msg; } catch { /* keep */ }
        throw new Error(msg);
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") continue;
          let j; try { j = JSON.parse(data); } catch { continue; }
          if (j.error) throw new Error(j.error.message || "The model returned an error.");
          const choice = (j.choices || [])[0];
          const delta = choice?.delta || {};
          if (delta.reasoning_content) { stream.reasoning += delta.reasoning_content; tokens++; first ??= performance.now(); }
          if (delta.content) { stream.content += delta.content; tokens++; first ??= performance.now(); }
          if (choice?.finish_reason) finish = choice.finish_reason;
          if (j.usage) usage = j.usage;
          if (j.timings) timings = j.timings;
          if (!raf) raf = requestAnimationFrame(paint);
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") stream.error = e.message || String(e);
    }
    cancelAnimationFrame(raf);

    const end = performance.now();
    const split = splitThinking(stream.content);
    const reasoning = stream.reasoning || split.reasoning;
    const answer = split.answer;
    const stopped = stream.ac.signal.aborted;
    const n = usage?.completion_tokens ?? tokens;
    const secs = Math.max((end - (first ?? t0)) / 1000, 0.001);
    const stats = {
      tokens: n, tps: timings?.predicted_per_second ?? (n > 1 ? (n - 1) / secs : 0), ttft: first ? (first - t0) / 1000 : null,
      prompt_tokens: usage?.prompt_tokens ?? null, stopped, truncated: finish === "length",
    };
    if (usage) ctxUsed = (usage.prompt_tokens || 0) + (usage.completion_tokens || 0);
    lastError = stream.error;
    stream = null;
    if (answer.trim() || reasoning.trim()) {
      try {
        const saved = await api.post(`/api/chats/${chat.id}/messages`, { role: "assistant", content: answer, reasoning, stats });
        chat.messages.push(saved);
      } catch (e) { toast(`Could not save the reply: ${e.message}`, "danger"); }
    }
    drawAll();
    scrollDown(true);
    reloadChats();
    refreshStatus();
  }

  function stopStream() { if (stream && stream.ac) stream.ac.abort(); }

  async function regenerate(msg) {
    if (stream || !chat) return;
    try { chat = await api.del(`/api/chats/${chat.id}/messages/${msg.id}`); } catch (e) { toast(e.message, "danger"); return; }
    drawAll();
    await generate();
  }

  async function deleteFrom(msg) {
    if (stream || !chat) return;
    try { chat = await api.del(`/api/chats/${chat.id}/messages/${msg.id}`); } catch (e) { toast(e.message, "danger"); return; }
    drawAll(); reloadChats();
  }

  function editMessage(msg) {
    const article = logEl.querySelector(`[data-id="${msg.id}"] .msg-body`);
    if (!article) return;
    const ta = h("textarea", { class: "textarea", rows: 3, "aria-label": "Edit message" });
    ta.value = msg.content;
    clear(article);
    article.append(ta, h("div", { class: "row-8 msg-edit-actions" },
      h("button", { class: "btn btn-primary", type: "button", onclick: async () => {
        const text = ta.value.trim();
        if (!text) return;
        try {
          chat = await api.del(`/api/chats/${chat.id}/messages/${msg.id}`);
          chat.messages.push(await api.post(`/api/chats/${chat.id}/messages`, { role: "user", content: text }));
        } catch (e) { toast(e.message, "danger"); return; }
        drawAll();
        await generate();
      } }, icon("send"), "Save and resend"),
      h("button", { class: "btn", type: "button", onclick: drawLog }, "Cancel")));
    ta.focus();
  }

  // ---------------------------------------------------------------- side panel: system prompt + sampling
  function drawSide() {
    if (sideEl.contains(document.activeElement) && document.activeElement.matches("textarea, input")) return;   // never steal focus while typing
    clear(sideEl);
    const p = params();
    const save = debounce(async () => {
      const body = { system: sysTa.value, params: readParams() };
      if (chat) { chat = { ...chat, ...(await api.patch(`/api/chats/${chat.id}`, body)), messages: chat.messages }; }
      else { draftSystem = body.system; draftParams = body.params; }
    }, 400);
    const sysTa = h("textarea", { class: "textarea", rows: 5, id: "c-sys", placeholder: "You are a helpful assistant.", oninput: save });
    sysTa.value = systemPrompt();
    const temp = numField("c-temp", "Temperature", p.temperature, { min: 0, max: 2, step: 0.05 }, save, "Lower is more focused, higher is more creative.");
    const top = numField("c-top", "Top P", p.top_p, { min: 0.05, max: 1, step: 0.05 }, save);
    const max = numField("c-max", "Max reply length", p.max_tokens ?? "", { min: 16, max: 131072, step: 16, placeholder: "No limit" }, save, "In tokens. Empty means until the model stops.");
    const readParams = () => ({ temperature: Number(temp.input.value) || 0, top_p: Number(top.input.value) || 1, max_tokens: max.input.value ? Number(max.input.value) : null });
    const s = srv();
    put(sideEl,
      h("div", { class: "row-between" }, h("h2", {}, "Chat settings"),
        h("button", { class: "btn btn-ghost btn-icon chat-side-close", type: "button", "aria-label": "Close settings", onclick: () => { sideOpen = false; store.set("reel.chat.side", "0"); layout.classList.remove("side-open"); drawHead(); } }, icon("x"))),
      h("div", { class: "field" }, h("label", { for: "c-sys" }, "System prompt"), sysTa,
        h("span", { class: "field-hint" }, "Sent before every message in this chat. Use it to set a role, tone or rules.")),
      h("div", { class: "stack-12" }, h("h4", {}, "Sampling"), temp.el, top.el, max.el),
      s.state === "ready" ? h("div", { class: "stack-8" }, h("h4", {}, "Loaded model"),
        h("dl", { class: "kv" }, h("dt", {}, "Model"), h("dd", {}, s.model_name), h("dt", {}, "Context"), h("dd", {}, `${s.options.context.toLocaleString()} tokens`),
          h("dt", {}, "Placement"), h("dd", {}, loadedWhere(s)), h("dt", {}, "Unloads after"), h("dd", {}, s.idle_minutes ? `${s.idle_minutes} min idle` : "never")),
        h("p", { class: "field-hint" }, "To change the context length or GPU use, eject and load again with Load…")) : null);
  }

  function numField(id, label, value, attrs, onchange, hint) {
    const input = h("input", { id, class: "input", type: "number", ...attrs, oninput: onchange });
    input.value = value;
    return { input, el: h("div", { class: "field" }, h("label", { for: id }, label), input, hint ? h("span", { class: "field-hint" }, hint) : null) };
  }

  function debounce(fn, ms) { let t; return () => { clearTimeout(t); t = setTimeout(fn, ms); }; }

  // ---------------------------------------------------------------- go
  function drawAll() { drawList(); drawHead(); drawLog(); drawComposer(); drawSide(); }

  const wanted = q.get("chat") || (!q.get("model") ? store.get(LAST_CHAT) : null);
  if (wanted && chatList.some((c) => c.id === wanted)) {
    try { chat = await api.get(`/api/chats/${wanted}`); } catch { chat = null; }
  }
  if (!chat) {
    draftModel = q.get("model") || loadedId() || store.get(LAST_MODEL);
    if (!modelById(draftModel) || modelById(draftModel).state !== "ready") draftModel = (ready()[0] || {}).id || null;
  }
  drawAll();
  scrollDown(true);
  schedule();
  composerInput()?.focus();
  const onHidden = () => { if (document.hidden) clearTimeout(timer); else refreshStatus(); };
  document.addEventListener("visibilitychange", onHidden);
  document.addEventListener("reel:wake", onHidden);      // this page stays mounted while you are elsewhere
  return {
    unmount() {
      alive = false; clearTimeout(timer);
      document.removeEventListener("visibilitychange", onHidden);
      document.removeEventListener("reel:wake", onHidden);
      if (stream && stream.ac) stream.ac.abort();
      root.classList.remove("main-fill");
    },
  };
}
