import { h, icon, clear, toast, confirmDialog, fmtBytes, statusBadge, restartReel } from "../ui.js";
import { api } from "../api.js";

export async function mount(root) {
  const body = h("div", { class: "stack-16" });
  root.append(h("div", { class: "view" },
    h("div", { class: "page-head" }, h("h1", {}, "Settings"), h("p", {}, "Where models live, access tokens, and what Reel found on this PC.")),
    body));

  async function load() {
    const [s, stored] = await Promise.all([api.get("/api/settings"), api.get("/api/settings/stored-models")]);
    clear(body);
    body.append(updatesCard(s), changelogCard(), gpuCard(s), memoryCard(s), windowCard(s), storageCard(s, stored), tokenCard(s), systemCard(s));
    refreshUpdates(false);
  }

  // ---- Updates ----
  const upd = h("div", { class: "stack-12" });
  let applied = null;

  async function refreshUpdates(force) {
    clear(upd);
    upd.append(h("div", { class: "muted icon-line" }, icon("clock"), force ? "Asking GitHub…" : "Checking…"));
    let u;
    try { u = await api.get(`/api/update/check?force=${force}`); }
    catch (e) { clear(upd); upd.append(h("div", { class: "notice notice-danger" }, icon("error"), e.message)); return; }
    drawUpdates(u);
    document.dispatchEvent(new CustomEvent("reel:update-state", { detail: u }));
  }

  function drawUpdates(u) {
    clear(upd);
    const where = { git: "Git clone", zip: "ZIP download" }[u.mode] || u.mode;
    const ver = u.local.short ? `${u.local.short}${u.local.date ? ` · ${u.local.date}` : ""}` : "unknown";
    const badge = { up_to_date: ["success", "Up to date", "check"], available: ["info", `${u.behind} update${u.behind === 1 ? "" : "s"}`, "download"],
      unknown: ["warning", "Version unknown", "alert"], error: ["danger", "Check failed", "error"], diverged: ["warning", "Diverged", "alert"],
      ahead: ["neutral", "Ahead of GitHub", "info"] }[u.state];
    const canUpdate = (u.state === "available" || u.state === "unknown") && !(u.dirty_files && u.dirty_files.length);
    upd.append(
      h("div", { class: "row-between" }, h("h4", {}, "This copy"), statusBadge(...badge)),
      h("dl", { class: "kv" }, h("dt", {}, "Version"), h("dd", { class: "mono" }, ver),
        h("dt", {}, "Installed as"), h("dd", {}, where),
        h("dt", {}, "Updates from"), h("dd", { class: "mono" }, `${u.repo} (${u.branch})`)));
    if (u.state === "up_to_date" && !applied) upd.append(h("div", { class: "notice notice-success" }, icon("check"), "You're on the latest version."));
    if (u.state === "error") upd.append(h("div", { class: "notice notice-danger", role: "alert" }, icon("error"), u.error));
    if (u.state === "unknown") upd.append(h("div", { class: "notice notice-warning" }, icon("alert"),
      `Reel can't tell which version this copy is. Updating installs the latest (${u.remote ? u.remote.short : "?"}) so it can track versions from then on.`));
    if (u.state === "diverged") upd.append(h("div", { class: "notice notice-warning" }, icon("alert"), "This copy has local commits that GitHub does not have, so it can't be fast-forwarded. Merge it with git yourself."));
    if (u.state === "ahead") upd.append(h("div", { class: "notice notice-info" }, icon("info"), "This copy is ahead of GitHub. Nothing to update."));
    if (u.dirty_files && u.dirty_files.length) upd.append(h("div", { class: "notice notice-warning" }, icon("alert"),
      `You have local changes to ${u.dirty_files.slice(0, 4).join(", ")}${u.dirty_files.length > 4 ? "…" : ""}. Commit, stash or discard them, then update.`));
    if (u.commits && u.commits.length) {
      upd.append(h("h4", {}, "What's new"), h("ul", { class: "changelog" }, u.commits.map((c) =>
        h("li", {}, h("code", {}, c.sha), " ", c.message, h("span", { class: "muted" }, ` · ${c.date}`)))));
    }
    if (applied) {
      upd.append(h("div", { class: "notice notice-success" }, icon("check"),
        `${applied.message} Now on ${applied.new_version}. Restart Reel to use it.${applied.requirements_changed ? " New app dependencies will be installed on restart." : ""}`));
    }
    const row = h("div", { class: "row-8" });
    if (applied) {
      row.append(h("button", { class: "btn btn-primary", type: "button", onclick: () => restartReel(api) }, icon("rerun"), u.can_restart ? "Restart Reel" : "Restart"));
    } else {
      row.append(h("button", { class: "btn btn-primary", type: "button", disabled: !canUpdate, onclick: () => applyUpdate(u) }, icon("download"), "Update now"));
    }
    row.append(h("button", { class: "btn", type: "button", onclick: () => refreshUpdates(true) }, icon("rerun"), "Check for updates"));
    upd.append(row);
  }

  async function applyUpdate(u) {
    const ok = await confirmDialog({ title: "Update Reel?", confirmLabel: "Update",
      body: "Reel downloads the new version and replaces its program files. Your models, clips, history and settings (the data folder) are not touched." });
    if (!ok) return;
    clear(upd); upd.append(h("div", { class: "muted icon-line" }, icon("download"), "Updating…"));
    try { applied = await api.post("/api/update/apply"); toast("Update installed.", "success"); }
    catch (e) { toast(e.message, "danger", 10000); applied = null; }
    // Started by start.bat: restart straight away so the new version is what you are using. Otherwise
    // (e.g. `python -m reel`) the Restart button stays and explains what to do.
    if (applied && applied.restart && u.can_restart) { await restartReel(api); return; }
    refreshUpdates(true);
  }

  function updatesCard(s) {
    const tok = h("input", { id: "gh-token", class: "input input-mono secret", type: "text", name: "reel-gh-pat", autocomplete: "off", spellcheck: "false", "data-lpignore": "true", "data-1p-ignore": "",
      placeholder: s.github_token_set ? "Token saved - enter a new one to replace it" : "github_pat_..." });
    const saveTok = async (value) => {
      try { await api.put("/api/settings", { set_github_token: true, github_token: value }); toast(value ? "Token saved." : "Token cleared.", "success"); await load(); }
      catch (e) { toast(e.message, "danger"); }
    };
    return h("section", { class: "card stack-16" },
      h("h3", {}, "Updates"),
      h("p", { class: "muted" }, "Updates Reel from GitHub in one click. Only the program files change; your models, clips, history and settings stay put. This is the only thing Reel contacts GitHub for."),
      upd,
      h("details", {}, h("summary", {}, "Private repository?"),
        h("div", { class: "stack-12" },
          h("p", { class: "muted" }, "If the GitHub repository is private, a ZIP-installed copy needs a GitHub access token with read access to it (a fine-grained token with Contents: read is enough). Git clones use your Git sign-in instead. Making the repository public removes the need for this."),
          h("div", { class: "field" }, h("label", { for: "gh-token" }, "GitHub access token"), tok),
          h("div", { class: "row-8" },
            h("button", { class: "btn", type: "button", onclick: () => tok.value.trim() ? saveTok(tok.value.trim()) : toast("Enter a token first.", "warning") }, "Save token"),
            h("button", { class: "btn", type: "button", disabled: !s.github_token_set, onclick: () => saveTok("") }, "Clear token")))));
  }

  // ---- Changelog ----
  // CHANGELOG.md is small and hand-written, so a few rules are enough: "## version", "**Section**", "- item".
  function inline(text) {
    const out = [];
    for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/)) {
      if (part.startsWith("**") && part.endsWith("**") && part.length > 4) out.push(h("strong", {}, part.slice(2, -2)));
      else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) out.push(h("code", {}, part.slice(1, -1)));
      else if (part) out.push(part);
    }
    return out;
  }

  function renderChangelog(text) {
    const root = h("div", { class: "stack-12 changelog-text" });
    let list = null;
    for (const raw of text.split("\n")) {
      const line = raw.trimEnd();
      if (line.startsWith("## ")) { list = null; root.append(h("h4", {}, line.slice(3))); }
      else if (/^\*\*[^*]+\*\*$/.test(line)) { list = null; root.append(h("h5", {}, line.slice(2, -2))); }
      else if (line.startsWith("- ")) {
        if (!list) { list = h("ul", { class: "changelog" }); root.append(list); }
        list.append(h("li", {}, inline(line.slice(2))));
      } else if (line.trim() && !line.startsWith("# ")) { list = null; root.append(h("p", { class: "muted" }, inline(line))); }
    }
    return root;
  }

  function changelogCard() {
    const slot = h("div", { class: "muted" }, "Loading…");
    const details = h("details", {}, h("summary", {}, "What's changed in each version"), slot);
    let loaded = false;
    details.addEventListener("toggle", async () => {
      if (!details.open || loaded) return;
      loaded = true;
      try {
        const { text } = await api.get("/api/changelog");
        clear(slot); slot.className = "";
        slot.append(text ? renderChangelog(text) : h("p", { class: "muted" }, "No changelog was found."));
      } catch (e) { slot.textContent = `Could not load it: ${e.message}`; }
    });
    return h("section", { class: "card stack-16" }, h("h3", {}, "Changelog"), details);
  }

  function gpuCard(s) {
    const gpus = s.gpu.gpus;
    const gb = (mb) => `${(mb / 1024).toFixed(0)} GB`;
    const sel = h("select", { id: "gpu-pick", class: "select", disabled: gpus.length < 2 || undefined },
      h("option", { value: "auto", selected: s.gpu.auto }, `Automatic - the one with the most VRAM${gpus.length ? ` (${[...gpus].sort((a, b) => b.vram_total_mb - a.vram_total_mb || a.index - b.index)[0].name})` : ""}`),
      gpus.map((g) => h("option", { value: g.uuid, selected: !s.gpu.auto && g.uuid === s.gpu.selected }, `GPU ${g.index}: ${g.name} · ${gb(g.vram_total_mb)}`)));
    return h("section", { class: "card stack-16" },
      h("h3", {}, "Graphics card"),
      !gpus.length ? h("div", { class: "notice notice-warning" }, icon("alert"), s.gpu.note || "No NVIDIA GPU was found, so generation would run on the CPU.")
        : h("p", { class: "muted" }, "Reel runs each model on one GPU. Automatic picks the card with the most VRAM; choose another here if you want it. Every model process is pinned to your choice, so it cannot wander onto a different card."),
      gpus.length ? h("dl", { class: "kv" }, ...gpus.flatMap((g) => [
        h("dt", {}, `GPU ${g.index}`),
        h("dd", {}, `${g.name} · ${gb(g.vram_total_mb)}`, g.uuid === s.gpu.selected ? " " : "", g.uuid === s.gpu.selected ? statusBadge("success", "In use", "check") : null)])) : null,
      gpus.length ? h("div", { class: "field" }, h("label", { for: "gpu-pick" }, "GPU to use"), sel) : null,
      gpus.length > 1 ? h("div", { class: "row-8" }, h("button", { class: "btn btn-primary", type: "button", onclick: async () => {
        try { await api.put("/api/settings", { set_gpu: true, gpu: sel.value === "auto" ? "" : sel.value }); toast("GPU saved. It applies to the next generation.", "success"); await load(); }
        catch (e) { toast(e.message, "danger"); }
      } }, "Save")) : null);
  }

  function memoryCard(s) {
    const input = h("input", { id: "idle-min", class: "input", type: "number", min: 0, max: 1440, step: 5, value: s.idle_unload_minutes });
    return h("section", { class: "card stack-16" },
      h("h3", {}, "Models in memory"),
      h("p", { class: "muted" }, "Reel keeps a model loaded after it generates, so the next run starts straight away. Only one model is on the GPU at a time; loading another unloads the current one, and Reel unloads everything when it closes."),
      h("div", { class: "field" }, h("label", { for: "idle-min" }, "Unload after this many minutes idle"), input,
        h("span", { class: "field-hint" }, "0 = never. A loaded model keeps its memory (with CPU offload that is mostly system RAM) until it is unloaded.")),
      h("div", { class: "row-8" }, h("button", { class: "btn btn-primary", type: "button", onclick: async () => {
        const v = Number(input.value);
        if (!Number.isInteger(v) || v < 0 || v > 1440) { toast("Enter a whole number of minutes from 0 to 1440.", "warning"); return; }
        try { await api.put("/api/settings", { set_idle_unload: true, idle_unload_minutes: v }); toast(v ? `Models unload after ${v} minutes idle.` : "Models stay loaded until you unload them.", "success"); }
        catch (e) { toast(e.message, "danger"); }
      } }, "Save")));
  }

  function windowCard(s) {
    const sel = h("select", { id: "win-mode", class: "select" },
      [["auto", "Automatic (native window if possible)"], ["native", "Native window"], ["app", "Browser app window"], ["browser", "Default browser tab"]]
        .map(([v, label]) => h("option", { value: v, selected: v === s.window_mode }, label)));
    return h("section", { class: "card stack-16" },
      h("div", { class: "row-between" }, h("h3", {}, "Window"),
        s.native_window.available ? statusBadge("success", "Native window ready", "check") : statusBadge("warning", "Native window unavailable", "alert")),
      h("p", { class: "muted" }, "A native window has no tabs, address bar, password prompts or browser shortcuts. If it can't start, Reel falls back to a browser window."),
      !s.native_window.available ? h("div", { class: "notice notice-warning" }, icon("alert"), s.native_window.note) : null,
      h("div", { class: "field" }, h("label", { for: "win-mode" }, "Open Reel in"), sel),
      h("div", { class: "row-8" }, h("button", { class: "btn btn-primary", type: "button", onclick: async () => {
        try { await api.put("/api/settings", { set_window_mode: true, window_mode: sel.value }); toast("Saved. It applies the next time you start Reel.", "success"); }
        catch (e) { toast(e.message, "danger"); }
      } }, "Save")));
  }

  function storageCard(s, stored) {
    const input = h("input", { id: "models-dir", class: "input input-mono", value: s.models_dir, spellcheck: "false", autocomplete: "off" });
    const save = async (value) => {
      try {
        await api.put("/api/settings", { set_models_dir: true, models_dir: value });
        toast(value ? "Storage folder saved." : "Storage folder reset to the default.", "success");
        await load();
      } catch (e) { toast(e.message, "danger", 9000); }
    };
    return h("section", { class: "card stack-16" },
      h("h3", {}, "Model storage"),
      h("p", { class: "muted" }, "Each model gets its own folder here (environment + weights). Uninstalling deletes only that folder."),
      h("div", { class: "field" },
        h("label", { for: "models-dir" }, "Storage folder"),
        h("div", { class: "row-8" },
          h("div", { class: "grow" }, input),
          h("button", { class: "btn", type: "button", onclick: async () => {
            try { const r = await api.post("/api/settings/browse"); if (r.path) input.value = r.path; }
            catch (e) { toast(e.message, "warning"); }
          } }, icon("folder"), "Browse…")),
        h("span", { class: "field-hint" }, `${fmtBytes(s.disk_free)} free on this drive. Default: ${s.default_models_dir}`)),
      h("div", { class: "row-8" },
        h("button", { class: "btn btn-primary", type: "button", onclick: () => save(input.value.trim()) }, "Save folder"),
        h("button", { class: "btn", type: "button", disabled: s.models_dir_is_default, onclick: () => save("") }, "Reset to default")),
      h("div", { class: "notice notice-info" }, icon("info"),
        "Changing the folder does not move existing models, because Python environments cannot be relocated. Models in the old folder stop appearing in the Library; uninstall them first, or move only the weights and reinstall."),
      stored.models.length
        ? h("div", { class: "stack-8" }, h("h4", {}, "In the current folder"),
            h("dl", { class: "kv" }, ...stored.models.flatMap((m) => [h("dt", { class: "mono" }, m.id), h("dd", {}, fmtBytes(m.bytes))])))
        : h("p", { class: "muted" }, "Nothing stored in this folder yet."));
  }

  function tokenCard(s) {
    const input = h("input", { id: "hf-token", class: "input input-mono secret", type: "text", name: "reel-hf-pat", autocomplete: "off", spellcheck: "false", "data-lpignore": "true", "data-1p-ignore": "", placeholder: s.hf_token_set ? "Token saved - enter a new one to replace it" : "hf_...", autocomplete: "off", spellcheck: "false" });
    const save = async (value) => {
      try { await api.put("/api/settings", { set_hf_token: true, hf_token: value }); toast(value ? "Token saved." : "Token cleared.", "success"); await load(); }
      catch (e) { toast(e.message, "danger"); }
    };
    return h("section", { class: "card stack-16" },
      h("div", { class: "row-between" }, h("h3", {}, "Hugging Face access"),
        s.hf_token_set ? statusBadge("success", "Token saved", "check") : statusBadge("neutral", "No token", "box")),
      h("p", { class: "muted" }, "Optional. Needed only for gated models, and it can speed up downloads. Stored in data/settings.json on this PC and only sent to Hugging Face."),
      h("div", { class: "field" }, h("label", { for: "hf-token" }, "Access token"), input),
      h("div", { class: "row-8" },
        h("button", { class: "btn btn-primary", type: "button", onclick: () => input.value.trim() ? save(input.value.trim()) : toast("Enter a token first.", "warning") }, "Save token"),
        h("button", { class: "btn", type: "button", disabled: !s.hf_token_set, onclick: async () => { if (await confirmDialog({ title: "Clear the saved token?", body: "Downloads will use anonymous access.", confirmLabel: "Clear" })) save(""); } }, "Clear token")));
  }

  function systemCard(s) {
    const gpu = s.gpu.gpus[0];
    return h("section", { class: "card stack-16" },
      h("h3", {}, "This PC"),
      h("dl", { class: "kv" },
        h("dt", {}, "GPU"), h("dd", {}, gpu ? `${gpu.name} · ${(gpu.vram_total_mb / 1024).toFixed(0)} GB VRAM` : (s.gpu.note || "None detected")),
        h("dt", {}, "NVIDIA driver"), h("dd", {}, s.gpu.driver ? h("span", { class: "row-8" }, s.gpu.driver,
          s.gpu.cuda_12_8_ok ? statusBadge("success", "CUDA 12.8 ready", "check") : statusBadge("warning", "Update to 570+ for CUDA 12.8", "alert")) : "-"),
        h("dt", {}, "Python (Reel)"), h("dd", {}, s.python),
        h("dt", {}, "Clips saved to"), h("dd", { class: "mono" }, s.outputs_dir)));
  }

  await load();
}
