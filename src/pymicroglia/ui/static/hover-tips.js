// Instant hover explanations for the correction reviewer's buttons and controls (Jamie 2026-11-20: "when
// hovering over a tool it explains what it does"). Shows the control's own title text at once in a box beside
// it, instead of the browser's delayed tooltip; the title is put back when the pointer leaves, so code that
// reads or sets titles is unaffected.
(() => {
  const box = document.createElement("div");
  box.id = "hover-tip";
  box.style.cssText = "position:fixed;z-index:10000;max-width:340px;padding:7px 9px;border-radius:6px;" +
    "background:#1d2330;color:#e8ecf3;border:1px solid #4a5568;font:12.5px/1.4 system-ui,sans-serif;" +
    "box-shadow:0 4px 14px rgba(0,0,0,.45);pointer-events:none;white-space:pre-line;display:none";
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(box));
  const SEL = "button[title], label[title], select[title], input[type=checkbox][title], input[type=number][title], input[type=text][title]";
  let cur = null, saved = "";

  function hide() {
    if (cur) { if (!cur.title) cur.title = saved; cur = null; }
    box.style.display = "none";
  }
  function place(el) {
    const r = el.getBoundingClientRect();
    box.style.display = "block";
    const w = box.offsetWidth, h = box.offsetHeight;
    let x = r.left, y = r.bottom + 6;
    if (x + w > innerWidth - 8) x = innerWidth - w - 8;
    if (y + h > innerHeight - 8) y = r.top - h - 6;                       // no room below: show above
    if (y < 8) { y = Math.max(8, r.top); x = r.right + 8 > innerWidth - w - 8 ? r.left - w - 8 : r.right + 8; }
    box.style.left = Math.max(8, x) + "px"; box.style.top = y + "px";
  }
  document.addEventListener("mouseover", e => {
    const el = e.target.closest && e.target.closest(SEL);
    if (el === cur) return;
    hide();
    if (!el || !el.title) return;
    cur = el; saved = el.title; el.title = "";                            // stop the browser's own tooltip
    box.textContent = saved;
    place(el);
  });
  document.addEventListener("mousedown", hide, true);                     // a click acts; the box gets out of the way
  document.addEventListener("scroll", hide, true);
  window.addEventListener("blur", hide);
})();
