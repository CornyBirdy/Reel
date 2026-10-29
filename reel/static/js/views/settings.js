import { h, icon, clear, toast, confirmDialog, fmtBytes, statusBadge } from "../ui.js";
import { api } from "../api.js";

export async function mount(root) {
  const body = h("div", { class: "stack-16" });
  root.append(h("div", { class: "view" },
    h("div", { class: "page-head" }, h("h1", {}, "Settings"), h("p", {}, "Where models live, access tokens, and what Reel found on this PC.")),
    body));

  async function load() {
    const [s, stored] = await Promise.all([api.get("/api/settings"), api.get("/api/settings/stored-models")]);
    clear(body);
    body.append(storageCard(s, stored), tokenCard(s), systemCard(s));
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
    const input = h("input", { id: "hf-token", class: "input input-mono", type: "password", placeholder: s.hf_token_set ? "Token saved - enter a new one to replace it" : "hf_...", autocomplete: "off", spellcheck: "false" });
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
