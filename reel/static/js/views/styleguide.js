// Dev-only page (#/styleguide): every component and state, for design and contrast audits.
import { h, icon, toast, confirmDialog, progressBar, statusBadge } from "../ui.js";

const swatch = (name) => h("div", { class: "sg-swatch" },
  h("div", { class: "sg-chip", style: { background: `var(--${name})` } }),
  h("code", {}, `--${name}`));

const section = (title, ...children) => h("section", { class: "card stack-16" }, h("h3", {}, title), ...children);

export function mount(el) {
  el.append(h("div", { class: "view" },
    h("div", { class: "page-head" }, h("h1", {}, "Styleguide"), h("p", {}, "Rack Suite tokens and components, neutral accent.")),

    section("Typography",
      h("h1", {}, "Heading 1 - Segoe UI Variable"), h("h2", {}, "Heading 2"), h("h3", {}, "Heading 3"),
      h("p", {}, "Body text, 14px. The quick brown fox jumps over the lazy dog."),
      h("p", { class: "muted" }, "Muted text for secondary information and hints."),
      h("p", {}, h("code", {}, "mono: python -m reel --port 8765"))),

    section("Buttons",
      h("div", { class: "row-12" },
        h("button", { class: "btn btn-primary" }, icon("download"), "Primary"),
        h("button", { class: "btn" }, icon("rerun"), "Secondary"),
        h("button", { class: "btn btn-danger" }, icon("trash"), "Danger"),
        h("button", { class: "btn btn-ghost" }, "Ghost"),
        h("button", { class: "btn btn-icon", "aria-label": "Settings" }, icon("settings")),
        h("button", { class: "btn btn-primary", disabled: true }, "Disabled primary"),
        h("button", { class: "btn", disabled: true }, "Disabled"))),

    section("Form controls",
      h("div", { class: "row-12" },
        h("div", { class: "field grow" }, h("label", { for: "sg-input" }, "Text input"), h("input", { id: "sg-input", class: "input", placeholder: "Placeholder text" })),
        h("div", { class: "field grow" }, h("label", { for: "sg-select" }, "Select"), h("select", { id: "sg-select", class: "select" }, h("option", {}, "Option one"), h("option", {}, "Option two"))),
        h("div", { class: "field grow" }, h("label", { for: "sg-num" }, "Number"), h("input", { id: "sg-num", class: "input", type: "number", value: 97 }))),
      h("div", { class: "field" }, h("label", { for: "sg-ta" }, "Textarea"), h("textarea", { id: "sg-ta", class: "textarea", placeholder: "Describe the clip..." })),
      h("label", { class: "check" }, h("input", { type: "checkbox", checked: true }), "Checkbox"),
      h("div", { class: "field" }, h("label", { for: "sg-range" }, "Range"), h("input", { id: "sg-range", type: "range", value: 40 }))),

    section("Selected vs hover",
      h("p", { class: "muted" }, "Hover is surface-raised only. Selected is accent-soft + accent-text + a 2px accent bar + heavier weight."),
      h("div", { class: "row-12" },
        h("div", { class: "stack-8", style: { width: "220px" } },
          h("a", { class: "nav-item", href: "#/styleguide" }, icon("library"), h("span", {}, "Default item")),
          h("a", { class: "nav-item", href: "#/styleguide", "aria-current": "page" }, icon("generate"), h("span", {}, "Selected item"))),
        h("div", { class: "seg", role: "group", "aria-label": "Segmented demo" },
          h("button", { "aria-pressed": "false" }, "One"), h("button", { "aria-pressed": "true" }, "Two"), h("button", { "aria-pressed": "false" }, "Three")))),

    section("Status",
      h("div", { class: "row-8" },
        statusBadge("success", "Installed"), statusBadge("warning", "Needs 16 GB VRAM"), statusBadge("danger", "Failed"),
        statusBadge("info", "Installing"), statusBadge("neutral", "Not installed")),
      h("div", { class: "notice notice-info" }, icon("info"), "Info notice - something worth knowing."),
      h("div", { class: "notice notice-warning" }, icon("alert"), "Warning notice - check this before continuing."),
      h("div", { class: "notice notice-danger" }, icon("error"), "Danger notice - this step failed."),
      h("div", { class: "notice notice-success" }, icon("check"), "Success notice - completed.")),

    section("Progress",
      progressBar(0.42, { label: "Determinate" }),
      h("div", { class: "progress-meta" }, h("span", {}, "5.1 GB of 12.3 GB"), h("span", {}, "42%")),
      progressBar(0, { indeterminate: true, label: "Indeterminate" })),

    section("Overlays",
      h("div", { class: "row-12" },
        h("button", { class: "btn", onclick: () => toast("Saved the clip to history.", "success") }, "Success toast"),
        h("button", { class: "btn", onclick: () => toast("Could not reach Hugging Face.", "danger") }, "Danger toast"),
        h("button", { class: "btn", onclick: () => confirmDialog({ title: "Uninstall LTX-Video?", body: "This deletes only that model's folder.", confirmLabel: "Uninstall", danger: true }) }, "Dialog"))),

    section("Palette",
      h("div", { class: "sg-grid" },
        ["ground", "surface-sunken", "surface", "surface-raised", "surface-track", "hairline", "border-control", "ink", "ink-muted"].map(swatch)),
      h("div", { class: "sg-grid" },
        ["success", "warning", "danger", "info", "success-soft", "warning-soft", "danger-soft", "info-soft"].map(swatch)),
      h("div", { class: "sg-grid" },
        ["accent", "on-accent", "accent-text", "accent-soft", "accent-2", "accent-3"].map(swatch)))));
}
