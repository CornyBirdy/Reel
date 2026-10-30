// "Verify": are this model's files complete and undamaged, and can its Python environment really use the GPU?
import { h, icon, clear, fmtBytes, fmtDuration, progressBar, statusBadge, toast } from "./ui.js";
import { api } from "./api.js";

const line = (ok, text) => h("li", { class: "verify-line" }, statusBadge(ok ? "success" : "danger", ok ? "OK" : "Problem"), h("span", {}, text));

function environmentLines(env) {
  if (!env) return [];
  const out = [];
  if (!env.ok) return [line(false, env.error || "The environment check failed.")];
  out.push(line(env.cuda_available !== false || !env.gpu_present, env.cuda_available
    ? `PyTorch ${env.torch} (CUDA ${env.cuda_build}) sees ${env.device}${env.vram_gb ? `, ${env.vram_gb} GB` : ""}.`
    : `PyTorch ${env.torch} cannot see a graphics card (CUDA build: ${env.cuda_build || "none"}).`));
  if (env.kernels_ok !== undefined) out.push(line(env.kernels_ok, env.kernels_ok ? `PyTorch has kernels for this GPU (${env.capability}).` : `PyTorch has no kernels for this GPU (${env.capability}).`));
  if (env.bnb_ok !== undefined) out.push(line(env.bnb_ok, env.bnb_ok ? `4-bit loading works (bitsandbytes ${env.bitsandbytes}).` : `4-bit loading failed: ${env.bnb_error}`));
  out.push(line(env.torch_import_seconds < 60, `PyTorch started in ${env.torch_import_seconds} s${env.torch_import_seconds < 60 ? "" : " (slow: security software or a slow disk)"}.`));
  return out;
}

export function openVerify(model, { onDone } = {}) {
  let timer = null, alive = true;
  const body = h("div", { class: "stack-12" });
  const actions = h("div", { class: "dialog-actions" });
  const dlg = h("dialog", { class: "dialog dialog-wide", "aria-labelledby": "verify-title" },
    h("h3", { id: "verify-title" }, `Verify ${model.name}`), body, actions);
  dlg.addEventListener("close", () => { alive = false; clearTimeout(timer); dlg.remove(); if (onDone) onDone(); });
  document.body.append(dlg);
  dlg.showModal();

  async function start(deep) {
    try { await api.post(`/api/models/${model.id}/verify`, { deep }); } catch (e) { toast(e.message, "danger", 8000); dlg.close(); return; }
    poll();
  }

  async function poll() {
    clearTimeout(timer);
    let job = null;
    try { job = (await api.get(`/api/models/${model.id}/verify`)).verify; } catch { /* retry */ }
    if (!alive) return;
    render(job);
    if (!job || job.state === "running") timer = setTimeout(poll, 800);
  }

  function render(job) {
    clear(body); clear(actions);
    if (!job || job.state === "running") {
      const frac = job && job.deep && job.bytes_total ? job.bytes_done / job.bytes_total : 0;
      body.append(h("div", {}, job ? job.stage : "Starting…"),
        progressBar(frac, { indeterminate: !(job && job.deep && job.bytes_total), label: "Verification progress" }),
        h("div", { class: "field-hint" }, job ? `${fmtDuration(job.elapsed)} elapsed${job.deep && job.bytes_total ? ` · ${fmtBytes(job.bytes_done)} of ${fmtBytes(job.bytes_total)} read` : ""}` : ""));
      actions.append(h("button", { class: "btn", type: "button", onclick: () => api.post(`/api/models/${model.id}/verify/cancel`).catch(() => {}) }, "Stop"));
      return;
    }
    const f = job.files, bad = f.missing.length + f.wrong_size.length + f.damaged.length + f.bad_hash.length;
    const list = h("ul", { class: "verify-list" });
    list.append(line(bad === 0, bad === 0
      ? `${f.checked} files checked${job.deep ? " against their checksums" : ""}: all present, the right size and readable.`
      : `${bad} of ${f.checked} files have a problem.`));
    for (const p of job.problems) list.append(line(false, p));
    list.append(...environmentLines(job.environment));
    body.append(h("div", { class: "row-8" }, statusBadge(job.ok ? "success" : "danger", job.ok ? "Everything is fine" : "Problems found"),
      h("span", { class: "muted" }, `${fmtDuration(job.elapsed)}${job.online ? "" : " · offline: sizes not compared with Hugging Face"}`)), list);
    if (!job.ok) body.append(h("div", { class: "field-hint" }, "Repair deletes the damaged files and downloads them again, and rebuilds the Python environment if it cannot use your graphics card. Files that are fine are kept."));
    actions.append(
      h("button", { class: "btn", type: "button", title: "Read every file and compare its checksum: slow for a big model, but catches damage a size check cannot",
        onclick: () => start(true) }, icon("search"), job.deep ? "Check again (deep)" : "Deep check"),
      job.ok ? null : h("button", { class: "btn btn-primary", type: "button", onclick: async () => {
        try { await api.post(`/api/models/${model.id}/repair`); toast("Repairing: the install runs again and skips what is fine.", "info", 5000); dlg.close(); }
        catch (e) { toast(e.message, "danger", 9000); }
      } }, icon("rerun"), "Repair"),
      h("button", { class: "btn", type: "button", onclick: () => dlg.close() }, "Close"));
  }

  start(false);
}
