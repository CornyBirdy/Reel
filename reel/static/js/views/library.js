import { h } from "../ui.js";
export function mount(el) {
  el.append(h("div", { class: "view" }, h("div", { class: "page-head" }, h("h1", {}, "Library"), h("p", {}, "Coming in a later stage."))));
}
