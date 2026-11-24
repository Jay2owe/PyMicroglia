// Correction reviewer in the workbench: Motion's correction_reviewer.py (Tk) as a page.
// Same layout, pictures, numbers, keys and edits; the pictures come from the same Python code
// (pymicroglia/ui/correction_review.py: bundle, panels). Every decision, note and edit is saved at once;
// nothing reaches Motion until Export.
// Added 2026-11-17 (Jamie): right-click follows cells in their own colours in one fixed view over their frames;
// draw / erase / seed / grow outlines; delete a whole track; the edit history; a timeline (Gantt chart) of every
// cell with flagged events; the tracking tower of the followed cells.
(() => {
  const $ = (id) => document.getElementById(id);
  const PANELS = ["Before", "After", "Raw", "Changes"];
  const VIEW_PANELS = ["Current", "Raw", "Changes"];   // question and follow views: nothing to compare
  const CHANNELS = ["Red", "Green", "Phase"];
  const STATE = {approve: "✓", deny: "✗"};
  const TIPS = {Before: "Approved session: cyan = number that continues, magenta = number being replaced",
    After: "With this join: green = the joined number", Raw: "Raw movie, same crop and contrast",
    Changes: "Red = pixels this join renames, orange = other corrections, purple = your edits",
    Current: "Numbers now. Right-click a cell to follow it in its own colour; Shift+right-click colours it without moving the view"};
  const FLAGS = {start: ["#7ee08a", "starts inside the recording"], end: ["#ff7b7b", "ends inside the recording"],
    gap: ["#9fb4ff", "missing for some frames, then back"], jump: ["#ffae42", "jumps more than the join rule's body radii with no overlap"],
    size: ["#ff6ad5", "area changes by the join rule's ratio or more in one frame"],
    pieces: ["#ffe14d", "number on two or more separate pieces"],
    mine: ["#4da3ff", "your own flags (J, or right-click the timeline)"]};
  const MUTED = "#b8bec7";
  const setting = (k, v) => { try { return v === undefined ? localStorage.getItem("cr-" + k) : localStorage.setItem("cr-" + k, v); } catch (e) { return null; } };
  const S = {root: "", stem: "", set: null, items: [], shown: [], pos: 0, frame: 0, lo: 0, hi: 0, play: null,
             picked: null, mode: null, line: [], undo: [], version: 0, cache: new Map(), geom: {}, cur: null,
             stamp: null, busy: false, drawing: 0, follow: [], fview: null, timeline: null, rows: [], flag: -1,
             gkind: null, zoom: null, pan: null, tool: null, queue: [], queueLine: false, marks: {ticks: {}, flags: [], resolved: []}, splitPreview: null, splitFlip: false,
             splitTimer: 0, splitTicket: 0, box: null,
             hover: null, preview: null, previewTimer: 0, previewTicket: 0};

  async function call(path, options = {}) {
    const r = await fetch(path, {headers: {"Content-Type": "application/json"}, ...options});
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (e) { /* not JSON */ }
      throw new Error(detail);
    }
    return r.json();
  }
  const q = (o) => new URLSearchParams(o).toString();
  const where = () => ({root: S.root, stem: S.stem});
  const post = (path, body) => call(path, {method: "POST", body: JSON.stringify({...where(), ...body})});
  function status(text, colour = MUTED) { const s = $("cr-status"); s.textContent = text || ""; s.title = text || ""; s.style.color = colour; }
  const cur = () => S.fview || S.shown[S.pos];   // what the pictures show: the follow view, else the selected row

  // The tracking tower's identity colours (figure_tables/tracking_tower.py identity_colours), so a cell has one colour everywhere.
  function idColour(n) {
    const h = (n * 0.61803398875) % 1, s = 0.5 + 0.14 * (n % 3), v = 0.88 + 0.05 * (n % 2);
    const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), qq = v * (1 - s * f), t = v * (1 - s * (1 - f));
    const [r, g, b] = [[v, t, p], [qq, v, p], [p, v, t], [p, qq, v], [t, p, v], [v, p, qq]][i % 6];
    return "#" + [r, g, b].map(x => Math.round(x * 255).toString(16).padStart(2, "0")).join("");
  }

  // ------------------------------------------------------------------ data
  async function defaultRoot() {
    const p = new URLSearchParams(location.search);
    if (p.get("root")) return p.get("root");
    if (setting("root")) return setting("root");
    try {
      const listed = await call("/api/motion/sources");
      const first = (listed.sources || listed || [])[0];
      const path = first && (first.path || first.source || first);
      if (typeof path === "string") return path.replace(/[\\/][^\\/]+[\\/]?$/, "");
    } catch (e) { /* Motion not connected */ }
    return "";
  }

  async function openRoot(root) {
    S.root = root; setting("root", root);
    const {reviews} = await call("/api/correction-review?" + q({root}));
    const box = $("cr-stem");
    box.innerHTML = reviews.map(r => `<option>${r.stem}</option>`).join("");
    if (!reviews.length) { status("No review sets in agent_review/ under " + root, "#ff9a9a"); return; }
    const want = new URLSearchParams(location.search).get("stem") || setting("stem");
    await openWell(reviews.some(r => r.stem === want) ? want : reviews[0].stem);
  }

  async function openWell(stem) {
    stop();
    status(`Loading ${stem}...`);
    S.stem = stem; setting("stem", stem); $("cr-stem").value = stem;
    S.set = await call("/api/correction-review/set?" + q(where()));
    S.items = S.set.items; S.mode = null; S.picked = null; S.line = []; S.cache.clear(); S.version++;
    S.fview = null; S.timeline = null; S.flag = -1;
    S.follow = (setting("follow-" + stem) || "").split(",").filter(Boolean).map(Number);
    renderChips();
    $("cr-sel").textContent = "";
    document.querySelectorAll(".cr-show input").forEach(b => { b.disabled = !S.set.channels.includes(b.dataset.ch); });
    S.stamp = await readStamp();
    await loadMarks();
    refill(true);
    status("");
    if (!$("cr-gantt-box").hidden) loadTimeline();
    const todo = S.shown.findIndex(it => dec(it) === undefined);
    select(todo >= 0 ? todo : 0);
  }

  const dec = (it) => S.set.decisions[it.key];
  const ops = () => S.set.edits;

  function refill(first = false) {
    const f = document.querySelector("#cr-filter input:checked").value;
    const keep = {all: () => true, todo: it => dec(it) === undefined, unsure: it => it.kind === "unsure",
                  check: it => it.check, deny: it => dec(it) === "deny"}[f];
    const current = !first && S.shown[S.pos] ? S.shown[S.pos].key : null;
    S.shown = S.items.filter(keep);
    const clash = new Set(S.set.conflicts || []);
    $("cr-rows").innerHTML = "";
    S.shown.forEach((it, i) => {
      const tr = document.createElement("tr");
      tr.dataset.i = i;
      tr.onclick = () => select(i);
      fillRow(tr, it, clash);
      $("cr-rows").append(tr);
    });
    updateProgress();
    if (!first && S.shown.length) {
      const keys = S.shown.map(it => it.key);
      select(keys.includes(current) ? keys.indexOf(current) : 0);
    } else if (!S.shown.length) { $("cr-title").textContent = "Nothing in this list"; }
  }

  function fillRow(tr, it, clash = new Set(S.set.conflicts || [])) {
    const d = dec(it);
    if (it.kind === "question") {   // an assistant question: opens at its cell and frame
      tr.innerHTML = `<td class="s">Q</td><td class="k"></td><td class="join">${it.old.length ? "cell " + it.old[0] : "question"}</td>` +
        `<td class="frames">frame ${it.frame}</td>`;
      tr.title = it.text;
      tr.className = ["question", +tr.dataset.i === S.pos && !S.fview ? "sel" : ""].join(" ");
      return;
    }
    const noted = S.set.notes.some(n => n.item === it.key && n.tag !== "Question");
    const mark = (d ? STATE[d] : (it.kind === "unsure" ? "?" : "·")) + (noted ? "✎" : "");
    const kind = it.kind === "unsure" ? "?" : (it.check ? "!" : "");
    const cells = [mark, kind, `${it.old.join(",")} → ${it.new.join(",")}`, `${it.first}–${it.last}`];
    tr.innerHTML = cells.map((c, j) => `<td class="${["s", "k", "join", "frames"][j]}">${c}</td>`).join("");
    tr.className = [d || (it.kind === "unsure" ? "unsure" : (it.check ? "check" : "")),
                    clash.has(it.key) ? "clash" : "", +tr.dataset.i === S.pos && !S.fview ? "sel" : ""].join(" ");
  }

  function updateRow(i) { const tr = $("cr-rows").children[i]; if (tr) fillRow(tr, S.shown[i]); }

  function updateProgress() {
    const joins = S.items.filter(it => it.kind !== "question");
    const done = joins.filter(it => dec(it) !== undefined).length;
    const unsure = joins.filter(it => it.kind === "unsure" && dec(it) === undefined).length;
    const n = ops().length;
    $("cr-progress").textContent = `${done}/${joins.length} decided · ${unsure} unsure left` +
      (n ? ` · ${n} edit${n > 1 ? "s" : ""}` : "");
    renderQuestions();
    renderHistory();
  }

  // ------------------------------------------------------------------ selection
  async function select(i) {
    if (!S.shown.length) { $("cr-title").textContent = "Nothing in this list"; return; }
    S.fview = null;
    S.pos = Math.max(0, Math.min(i, S.shown.length - 1));
    [...$("cr-rows").children].forEach((tr, j) => tr.classList.toggle("sel", j === S.pos));
    $("cr-rows").children[S.pos]?.scrollIntoView({block: "nearest"});
    const it = S.shown[S.pos];
    const {lo, hi} = await call("/api/correction-review/item?" + q({...where(), key: it.key}));
    if (S.shown[S.pos] !== it || S.fview) return;
    S.lo = lo; S.hi = hi;
    $("cr-slider").min = lo; $("cr-slider").max = hi;
    S.frame = it.kind === "question" ? it.frame : Math.max(lo, it.frame - 1);
    if (it.kind === "question") {
      const [, , y, x] = it.key.split(":").map(Number);
      S.picked = {frame: it.frame, y, x, now: it.old[0] || null};
    }
    $("cr-slider").value = S.frame;
    showTitle(it);
    sendFocus();
    await draw(true);
    startPlay();
  }

  function showTitle(it) {
    $("cr-title").title = "";
    if (it.kind === "follow") {
      const h = (f) => S.set.hours[f] ?? "?";
      $("cr-title").innerHTML = `${S.stem}  Following ` + it.cells.map(n => `<span style="color:${idColour(n)}">${n}</span>`).join(", ") +
        `  ·  ${it.cells.length > 1 ? "in" : "exists in"} frames ${it.present[0]}–${it.present[1]} (${h(it.present[0])}–${h(it.present[1])} h), ` +
        `slider ${it.lo}–${it.hi}  ·  right-click a cell to add or remove it, Esc goes back to the list`;
      return;
    }
    if (it.kind === "question") {
      $("cr-title").textContent = `${S.stem}  Question at frame ${it.frame} (${S.set.hours[it.frame] ?? "?"} h), ` +
        `${it.old.length ? "cell " + it.old[0] + " (yellow ring on this frame)" : "clicked spot (yellow ring)"}: ${it.text.replace(/^Question:\s*/, "")}`;
      $("cr-title").title = it.text;
      return;
    }
    const d = dec(it);
    const what = it.kind === "unsure" ? "unsure, " + (d === "approve" ? "approved" : d === "deny" ? "denied" : "not applied")
      : "applied" + (d === "deny" ? ", denied" : d === "approve" ? ", approved" : "");
    $("cr-title").textContent = `${S.stem} #${S.pos + 1}   ${it.old.join(",")} → ${it.new.join(",")}` +
      `   ·  join at frame ${it.frame} (${S.set.hours[it.frame] ?? "?"} h)` +
      `  ·  moved ${it.radii} body widths  ·  gap ${it.gap} frame${it.gap > 1 ? "s" : ""}` +
      `  ·  size ×${it.area_ratio}  ·  ${what}`;
  }

  // ------------------------------------------------------------------ following cells
  function saveFollow() { setting("follow-" + S.stem, S.follow.join(",")); renderChips(); }

  function renderChips() {
    const box = $("cr-chips");
    box.innerHTML = S.follow.length ? "" : `<span class="cr-muted">right-click cells in the pictures</span>`;
    for (const n of S.follow) {
      const b = document.createElement("button");
      b.className = "chip"; b.style.borderColor = b.style.color = idColour(n); b.tabIndex = -1;
      b.textContent = `${n} ×`; b.title = `Stop following ${n}`;
      b.onclick = () => { S.follow = S.follow.filter(m => m !== n); saveFollow(); refreshFollow(); };
      box.append(b);
    }
  }

  function followRange() {
    const a = $("cr-ffrom").value.trim(), b = $("cr-fto").value.trim();
    return {lo: /^\d+$/.test(a) ? +a : "", hi: /^\d+$/.test(b) ? +b : ""};
  }

  const followKey = () => { const {lo, hi} = followRange(); return `follow:${lo}:${hi}:${[...S.follow].sort((a, b) => a - b).join(",")}`; };

  // One fixed view holding every followed cell over their frames (or the frames typed in the follow box).
  async function openFollow(frame = S.frame, keepStatus = false) {
    if (!S.follow.length) { if (S.fview) backToList(); return; }
    stop();
    let view;
    try { view = await call("/api/correction-review/item?" + q({...where(), key: followKey()})); }
    catch (e) {   // a followed cell is gone (deleted, renumbered): follow the ones still there, else back to the list
      const tl = await call("/api/correction-review/cells?" + q(where())).catch(() => null);
      if (tl) { S.timeline = tl; renderLegend(); }
      const alive = tl ? S.follow.filter(n => tl.cells.some(c => c.n === n)) : [];
      if (alive.length && alive.length < S.follow.length) { S.follow = alive; saveFollow(); return openFollow(frame, keepStatus); }
      if (!keepStatus) status(e.message, "#ff9a9a");
      if (S.fview) { S.fview = null; select(S.pos); }
      return;
    }
    const key = followKey();
    S.fview = {key, kind: "follow", frame: view.present[0], lo: view.lo, hi: view.hi, cells: view.cells, present: view.present};
    [...$("cr-rows").children].forEach(tr => tr.classList.remove("sel"));
    S.lo = view.lo; S.hi = view.hi;
    $("cr-slider").min = S.lo; $("cr-slider").max = S.hi;
    S.frame = Math.max(S.lo, Math.min(S.hi, frame));
    $("cr-slider").value = S.frame;
    showTitle(S.fview);
    sendFocus();
    await draw(true);
  }

  function refreshFollow() { if (S.fview) openFollow(); else draw(); drawGantt(); }

  function backToList() { S.fview = null; select(S.pos); }

  async function followAt(name, e) {
    const p = at(name, e);
    if (!p) return;
    const hit = await call("/api/correction-review/pick?" + q({...where(), key: fkey(cur()), frame: S.frame, y: p.y, x: p.x, panel: name}));
    if (!hit.now) { status("Right-click on a cell to follow it", "#ff9a9a"); return; }
    const had = S.follow.includes(hit.now);
    S.follow = had ? S.follow.filter(n => n !== hit.now) : S.follow.concat([hit.now]);
    saveFollow();
    status(had ? `Stopped following ${hit.now}` : `Following ${hit.now}` + (e.shiftKey ? " (view kept: F shows them together)" : ""), had ? MUTED : idColour(hit.now));
    if (e.shiftKey && !S.fview) { draw(); drawGantt(); return; }
    if (!S.follow.length) { backToList(); drawGantt(); return; }
    await openFollow(S.frame);
    drawGantt();
  }

  // ------------------------------------------------------------------ drawing
  function panelNames() {
    const it = cur();
    const base = it && (it.kind === "follow" || it.kind === "question") ? VIEW_PANELS : PANELS;
    return base.concat(CHANNELS.filter(c => S.set && S.set.channels.includes(c) &&
      document.querySelector(`.cr-show input[data-ch="${c}"]`).checked));
  }

  function layoutPanels() {
    const names = panelNames(), grid = $("cr-grid");
    const n = names.length, cols = n <= 3 ? n : n === 4 ? 2 : n <= 6 ? 3 : 4, rows = Math.ceil(n / cols);
    grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    grid.style.gridTemplateRows = `repeat(${rows}, minmax(0, 1fr))`;
    if (grid.dataset.names === names.join()) return;
    grid.dataset.names = names.join(); grid.innerHTML = "";
    for (const name of names) {
      const c = document.createElement("canvas");
      c.dataset.name = name; c.title = TIPS[name] || `${name}, same crop; outlines as in Before`;
      c.addEventListener("mousedown", e => {
        if (e.button === 2) { e.preventDefault(); followAt(name, e).catch(err => status(err.message, "#ff9a9a")); }
        else if (e.button === 1) { e.preventDefault(); panStart(name, e); }
        else if (e.button === 0) pickAt(name, e);
      });
      c.addEventListener("contextmenu", e => e.preventDefault());
      c.addEventListener("mousemove", e => { if (S.pan) panMove(e); else if (S.box) boxMove(name, e); else { dragAt(name, e); previewHover(name, e); splitHover(e); } });
      c.addEventListener("mouseup", e => { if (e.button === 0) { if (S.box) boxEnd().catch(err => status(err.message, "#ff9a9a")); else releaseAt(name, e); } });
      c.addEventListener("wheel", e => { e.preventDefault(); zoomAt(name, e, e.deltaY < 0 ? 1.25 : 0.8); }, {passive: false});
      grid.append(c);
    }
  }

  function zoomFor(crop) {   // largest whole zoom that fits the first panel, in screen pixels (as the Tk reviewer)
    const c = $("cr-grid").querySelector("canvas"), dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth * dpr, h = c.clientHeight * dpr;
    return Math.max(1, Math.floor(Math.min(w / crop[1], h / crop[0])));
  }

  const loadImage = (src) => new Promise(ok => { const im = new Image(); im.onload = () => ok(im); im.src = src; });
  const every = () => $("cr-every").checked;

  async function getFrame(key, frame, k, prefetch = []) {
    const shown = panelNames().filter(n => CHANNELS.includes(n)).join(",");
    const follow = S.follow.join(","), all = every() ? 1 : 0;
    const id = `${S.stem}|${key}|${frame}|${k}|${S.version}|${shown}|${follow}|${all}`;
    let entry = S.cache.get(id);
    if (!entry) {
      entry = call("/api/correction-review/frame?" + q({...where(), key, frame, k, prefetch: prefetch.join(","),
                                                        channels: shown, follow, every: all ? "true" : "false"}))
        .then(async d => ({...d, images: Object.fromEntries(await Promise.all(Object.entries(d.panels)
          .map(async ([n, src]) => [n, await loadImage(src)]))), ov: await loadImage(d.overview)}));
      S.cache.set(id, entry);
      entry.catch(() => S.cache.delete(id));
      if (S.cache.size > 160) S.cache.delete(S.cache.keys().next().value);
    }
    return entry;
  }

  async function draw(fresh = false) {
    const it = cur();
    if (!it) return;
    layoutPanels();
    const frame = S.frame, ticket = ++S.drawing;
    // First a frame at the current zoom guess, so the crop size is known; then the real zoom.
    let k = S.k || 3;
    const next = S.fview ? [] : S.shown.slice(S.pos + 1, S.pos + 5).concat(S.shown.slice(Math.max(0, S.pos - 1), S.pos)).map(x => x.key);
    let fr;
    try {
      fr = await getFrame(fkey(it), frame, k, fresh ? next : []);
      const want = zoomFor(fr.crop);
      if (want !== k) { S.k = k = want; fr = await getFrame(fkey(it), frame, k); }
    } catch (e) { status(e.message, "#ff9a9a"); return; }
    if (ticket !== S.drawing) return;   // a newer frame was asked for meanwhile
    S.cur = fr;
    const Z = zoomState(it, fr);
    const dpr = window.devicePixelRatio || 1;
    const strong = fr.nb.filter(n => n[3] !== "#c8c8c8");
    for (const c of $("cr-grid").querySelectorAll("canvas")) {
      const name = c.dataset.name;
      c.width = Math.round(c.clientWidth * dpr); c.height = Math.round(c.clientHeight * dpr);
      const g = c.getContext("2d");
      g.imageSmoothingEnabled = false;
      g.fillStyle = "#000"; g.fillRect(0, 0, c.width, c.height);
      const img = fr.images[name];
      if (!img) {
        g.fillStyle = "#888888"; g.font = `${13 * dpr}px Segoe UI`; g.textAlign = "center"; g.textBaseline = "middle";
        g.fillText(`${name}…`, c.width / 2, c.height / 2); continue;
      }
      // A follow view can be larger than the panel even at zoom 1: shrink it to fit.
      // Magnified (mouse wheel) about the view centre Z.vy, Z.vx (crop pixels).
      const fit = Math.min(c.width / img.width, c.height / img.height);   // a widened field fills the panel
      const s = Z.field ? fit : Math.min(1, fit), gk = fr.k * s * Z.z;
      clampView(Z, fr.crop, c, gk);
      const ox = Math.round(c.width / 2 - Z.vx * gk), oy = Math.round(c.height / 2 - Z.vy * gk);
      g.drawImage(img, ox, oy, img.width * s * Z.z, img.height * s * Z.z);
      g.font = `bold ${13.33 * dpr}px Segoe UI`; g.fillStyle = "#e8e8e8"; g.textAlign = "left"; g.textBaseline = "top";
      g.fillText(name, Math.max(ox, 0) + 6 * dpr, Math.max(oy, 0) + 4 * dpr);
      S.geom[name] = {ox, oy, k: gk, origin: fr.origin, crop: fr.crop, dpr};
      if (S.picked && S.picked.frame === frame) {
        const py = S.picked.y - fr.origin[0], px = S.picked.x - fr.origin[1];
        g.strokeStyle = "#ffe14d"; g.lineWidth = 2 * dpr; g.beginPath();
        g.ellipse(ox + (px + 0.5) * gk, oy + (py + 0.5) * gk, Math.max(6 * dpr, 3.5 * gk), Math.max(6 * dpr, 3.5 * gk), 0, 0, 2 * Math.PI); g.stroke();
      }
      const nums = name === "Before" || name === "Current" ? fr.nb : name === "After" ? fr.na : CHANNELS.includes(name) ? strong : null;
      if (nums) {
        g.textAlign = "center"; g.textBaseline = "middle";
        for (const [y, x, t, col] of nums) {
          const followed = S.follow.includes(+t);
          g.font = `bold ${(followed ? 16 : 13.33) * dpr}px Segoe UI`;
          if (followed) { g.lineWidth = 3 * dpr; g.strokeStyle = "#000"; g.strokeText(t, ox + (x + 0.5) * gk, oy + (y + 0.5) * gk); }
          g.fillStyle = col; g.fillText(t, ox + (x + 0.5) * gk, oy + (y + 0.5) * gk);
          if (ticked(t)) { g.fillStyle = "#7ee08a"; g.fillText("\u2713", ox + (x + 0.5) * gk + g.measureText(t).width * 0.5 + 6 * dpr, oy + (y + 0.5) * gk); }
        }
      }
      if (S.preview) drawPreview(g, ox, oy, gk, fr.origin, frame);
      if (S.splitPreview) drawSplit(g, ox, oy, gk, fr.origin, dpr, frame);
      drawLine(g, ox, oy, gk, fr.origin, dpr);
      if (S.box) drawBox(g, ox, oy, gk, fr.origin, dpr, name);
      if (S.queue.length) drawQueue(g, ox, oy, gk, fr.origin, dpr, frame);   // queued spots on top
    }
    const ov = $("cr-overview"), og = ov.getContext("2d");
    og.fillStyle = "#000"; og.fillRect(0, 0, ov.width, ov.height); og.imageSmoothingEnabled = false;
    og.drawImage(fr.ov, 0, 0);
    const [y0, x0, y1, x1] = fr.box;
    og.strokeStyle = "#ffe14d"; og.lineWidth = 1; og.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0, y1 - y0);
    const tag = it.kind === "follow" ? "" : frame === it.frame ? (it.kind === "question" ? "  asked" : "  join") : "";
    $("cr-frame").textContent = `frame ${frame}  ${S.set.hours[frame] ?? "?"} h${tag}${zoomText(fr)}`;
    drawGantt();
    if (fresh && !S.fview) {   // the play loop's frames, ready before they are shown
      for (let f = Math.max(S.lo, it.frame - 3); f <= Math.min(S.hi, it.frame + 3); f++) getFrame(fkey(it), f, k).catch(() => {});
    } else if (S.fview) {      // a follow view plays its whole range: draw the next few frames ahead
      for (let f = frame + 1; f <= Math.min(S.hi, frame + 3); f++) getFrame(fkey(it), f, k).catch(() => {});
    }
  }

  // ------------------------------------------------------------------ sticky tools
  // A tool stays on after use (Jamie 2026-11-17: "the same tool stayed selected for quickly doing actions
  // continuously"): every click applies it until Esc or its key/button again. "Tool stays on" off = one-shot.
  const TOOLS = {
    swap: ["cr-swap", "Swap", "click a cell, then the cell to swap it with"],
    give: ["cr-give", "Give number", "type the number in the box, then click each cell to give it that number"],
    delete: ["cr-delete", "Delete", "click each cell to delete it (this frame to its run end, or the 'to' frame)"],
    track: ["cr-track", "Whole-track delete", "click each cell to delete it from every frame"],
    split: ["cr-split", "Draw split", "draw a line across each cell to cut it"],
    trim: ["cr-trim", "Trim", "draw a line across each cell; the orange side is erased"],
    fence: ["cr-fence", "Fence", "click the cell, then draw round the area it must never enter"],
    keepin: ["cr-keepin", "Keep in", "click the cell, then draw round the area it must stay inside"],
    keep: ["cr-keep", "Keep split", "click each piece to keep its number over the snippet"],
    whole: ["cr-whole", "Whole recording", "click each piece to keep its number over the whole recording"],
    draw: ["cr-draw", "Draw outline", "draw round each area to outline"],
    erase: ["cr-erase", "Erase inside", "draw round each area to erase"],
    seed: ["cr-seed", "Seed", "click empty signal to seed each cell"],
    grow: ["cr-grow", "Grow", "click each cell to grow it into the signal touching it"],
    mgrow: ["cr-mgrow", "Model grow", "click each cell to add the outline model's piece of it"],
    island: ["cr-island", "One island", "click the piece of each cell to keep; its other pieces are freed"]};
  const ONE_SHOT = {swap: () => startSwap(), give: () => startGive(), delete: () => deleteCell(), track: () => deleteTrack(),
    split: () => startLine("split"), trim: () => startLine("trim"), fence: () => startLine("fence"), keepin: () => startLine("keepin"), keep: () => keep(false), whole: () => keep(true), draw: () => startLine("draw"),
    erase: () => startLine("erase"), seed: () => seed(), grow: () => grow(), mgrow: () => mgrow(), island: () => island()};
  const sticky = () => $("cr-sticky").checked;
  const typedNumber = () => { const t = $("cr-give-box").value.trim(); return /^\d+$/.test(t) && +t ? +t : null; };

  function showTool() {
    for (const [t, [id]] of Object.entries(TOOLS)) $(id).classList.toggle("tool-on", S.tool === t);
    $("cr-grid").classList.toggle("tool-active", !!S.tool);
  }
  function toolOff(message = "") {
    const was = S.tool; S.tool = null; S.mode = null; S.line = []; S.preview = null; showTool();
    if (was) status(message || `${TOOLS[was][1]} off`);
    draw();
  }
  function useTool(name) {   // a tool's key or button
    if (!sticky()) return ONE_SHOT[name]();
    if (S.tool === name) return toolOff();
    stop(); S.tool = name; S.mode = null; S.line = []; showTool();
    if (LINE_MODES.includes(name)) return startLine(name);
    const p = S.picked;
    const usable = p && p.frame === S.frame && (name === "seed" ? p.y != null && !p.now : !!p.now) && (name !== "give" || typedNumber());
    if (usable) return applyTool(p);
    status(`${TOOLS[name][1]} on: ${TOOLS[name][2]} (Esc or the key again stops)`, "#ffe14d");
    if (name === "give" && !typedNumber()) $("cr-give-box").focus();
  }
  function applyTool(hit) {   // a click while a (non-line) tool is on
    const t = S.tool;
    if (t === "seed") {
      if (hit.now) { status(`That is cell ${hit.now}: click empty signal to seed (Seed is on)`, "#ff9a9a"); return draw(); }
      return seed();
    }
    if (!hit.now) { status(`Click on a cell (${TOOLS[t][1]} is on; Esc stops)`, "#ff9a9a"); return draw(); }
    if (t === "swap") { S.mode = "swap"; status(`Click the cell to swap ${hit.now} with (Swap is on)`, "#ffe14d"); return draw(); }
    if (t === "give") {
      const n = typedNumber();
      if (!n) { $("cr-give-box").focus(); status("Type the number to give in the box first (Give number is on)", "#ff9a9a"); return; }
      return give(n);
    }
    ({delete: deleteCell, track: deleteTrack, keep: () => keep(false), whole: () => keep(true), grow, mgrow, island})[t]();
  }

  // ------------------------------------------------------------------ queued edits
  // Ctrl+click (or a Ctrl+drawn outline) with a tool on queues it; Enter or "Run queued" applies all of them in
  // order, each spot looked up again after the edits before it (Jamie 2026-11-17).
  const QUEUE_TOOL = {delete: "delete", track: "delete", give: "give", grow: "grow", mgrow: "model_grow", island: "one_island", seed: "seed", keep: "keep",
                      whole: "keep", draw: "draw", erase: "erase", split: "split", trim: "trim"};
  function queueItem(extra) {
    const t = S.tool;
    if (!QUEUE_TOOL[t]) { status(t === "swap" ? "Swap cannot be queued: it needs pairs of clicks" : "Pick a tool first, then Ctrl+click to queue", "#ff9a9a"); return; }
    const item = {tool: t, action: QUEUE_TOOL[t], frame: S.frame, whole: t === "track" || t === "whole",
                  piece_only: $("cr-piece").checked, level: +$("cr-level").value || 0.2, one_piece: $("cr-onepiece").checked,
                  ...(t === "split" || t === "trim" ? {} : range()), ...extra};
    const n = t === "split" || t === "trim" ? null : t === "seed" ? numberSet() : typedNumber();
    if (t === "give") { if (!n) { $("cr-give-box").focus(); status("Type the number to give in the box first", "#ff9a9a"); return; } item.number = n; }
    else if (["draw", "seed", "keep", "whole", "split"].includes(t) && n) item.number = n;
    S.queue.push(item); showQueue(); draw();
    status(`Queued ${TOOLS[t][1]} #${S.queue.length} at frame ${S.frame} (Enter runs all, Esc clears the queue)`, "#ffe14d");
  }
  function showQueue() {
    const n = S.queue.length;
    $("cr-qrun").hidden = $("cr-qclear").hidden = !n;
    $("cr-qrun").textContent = `Run ${n} queued`;
  }
  function clearQueue() { S.queue = []; showQueue(); status("Queue cleared"); draw(); }
  async function runQueue() {
    if (!S.queue.length) return;
    const items = S.queue.map(({tool, ...rest}) => rest), it = cur();
    status(`Running ${items.length} queued edits...`);
    try {
      const r = await post("/api/correction-review/edit-batch", {items, key: it ? fkey(it) : null});
      for (const x of r.results) if (x.ok && x.count) S.undo.push({kind: "edit", stem: S.stem, count: x.count});
      S.queue = []; showQueue();
      await reload(r.message, r.done === items.length ? "#c9a6ff" : "#ff9a9a");
    } catch (err) { status(err.message, "#ff9a9a"); }
  }
  function drawQueue(g, ox, oy, gk, origin, dpr, frame) {   // numbered markers of the queued spots in this frame
    S.queue.forEach((q, i) => {
      if (q.frame !== frame) return;
      const pts = q.points || [[q.y, q.x]];
      const y = pts.reduce((a, p) => a + p[0], 0) / pts.length, x = pts.reduce((a, p) => a + p[1], 0) / pts.length;
      const cx = ox + (x - origin[1] + 0.5) * gk, cy = oy + (y - origin[0] + 0.5) * gk;
      if (q.points) {
        g.strokeStyle = "#ffe14d"; g.setLineDash([4 * dpr, 3 * dpr]); g.lineWidth = 1.5 * dpr; g.beginPath();
        q.points.forEach(([py, px], j) => { const X = ox + (px - origin[1] + 0.5) * gk, Y = oy + (py - origin[0] + 0.5) * gk; j ? g.lineTo(X, Y) : g.moveTo(X, Y); });
        if (q.action !== "split" && q.action !== "trim") g.closePath();
        g.stroke(); g.setLineDash([]);
      }
      g.fillStyle = "#ffe14d"; g.beginPath(); g.arc(cx, cy, 8 * dpr, 0, 2 * Math.PI); g.fill();
      g.fillStyle = "#000"; g.font = `bold ${10 * dpr}px Segoe UI`; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(String(i + 1), cx, cy);
    });
  }

  // ------------------------------------------------------------------ my marks
  // Jamie's own ticks (cells checked as good) and flags (a cell at a frame to revisit, with a note), saved in
  // marks.json beside the review, apart from the notes the agent reads (2026-11-17).
  async function loadMarks() {
    try { S.marks = await call("/api/correction-review/marks?" + q(where())); } catch (e) { S.marks = {ticks: {}, flags: []}; }
    renderMarks();
  }
  async function markCall(body) {
    S.marks = await post("/api/correction-review/mark", body);
    renderMarks();
    if (S.timeline) { renderLegend(); drawGantt(); renderEvents(); }
    draw();
    return S.marks;
  }
  const ticked = (n) => !!(S.marks.ticks || {})[n];
  const markTargets = () => S.picked && S.picked.now && S.picked.frame === S.frame ? [S.picked.now] : S.follow.slice();
  async function tickCells(cells = markTargets()) {
    if (!cells.length) { status("Click a cell (or follow cells) first, then V", "#ff9a9a"); return; }
    const on = !cells.every(ticked);
    try { for (const n of cells) await markCall({op: on ? "tick" : "untick", cell: n, frame: S.frame}); }
    catch (e) { status(e.message, "#ff9a9a"); return; }
    status(`${on ? "Ticked" : "Unticked"} ${cells.join(", ")} as good`, on ? "#7ee08a" : MUTED);
  }
  async function flagCells(cells = markTargets(), frame = S.frame) {
    if (!cells.length) { status("Click a cell (or follow cells) first, then J", "#ff9a9a"); return; }
    try { for (const n of cells) await markCall({op: "flag", cell: n, frame}); }
    catch (e) { status(e.message, "#ff9a9a"); return; }
    $("cr-marks").open = true;
    const last = S.marks.flags[S.marks.flags.length - 1], box = document.querySelector(`.mk-note[data-id="${last.id}"]`);
    if (box) box.focus();
    status(`Flagged ${cells.join(", ")} at frame ${frame}: type a note and press Enter (or leave it blank)`, "#4da3ff");
  }
  function markedCells(keepResolved = false) {   // the timeline's cells with your flags added as events of kind "mine"
    const by = {};
    for (const fl of S.marks.flags) (by[fl.cell] = by[fl.cell] || []).push([fl.frame, "mine",
      `${fl.cell}: your flag at frame ${fl.frame}${fl.note ? " - " + fl.note : ""}`]);
    const done = keepResolved ? new Set() : resolvedSet();
    return S.timeline.cells.map(c => {
      let events = by[c.n] ? c.events.concat(by[c.n]).sort((a, b) => a[0] - b[0]) : c.events;
      if (done.size) events = events.filter(ev => !done.has(evKey(c.n, ev[0], ev[1])));
      return events === c.events ? c : {...c, events};
    });
  }
  function renderMarks() {
    const m = S.marks, ticks = Object.keys(m.ticks).map(Number).sort((a, b) => a - b), box = $("cr-marklist");
    $("cr-marks-sum").textContent = `My marks (${ticks.length} good, ${m.flags.length} flagged)`;
    box.innerHTML = "";
    if (ticks.length) {
      const row = document.createElement("div"); row.className = "mk-ticks";
      row.innerHTML = `<span class="mk-ok">&#10003; good:</span> ` + ticks.map(n =>
        `<span class="mk-chip" data-n="${n}" title="Click: follow ${n}. x: untick">${n}<a data-un="${n}">&times;</a></span>`).join(" ");
      row.onclick = (e) => {
        const un = e.target.dataset.un, n = e.target.closest(".mk-chip")?.dataset.n;
        if (un) { e.stopPropagation(); markCall({op: "untick", cell: +un}); }
        else if (n) { S.follow = [+n]; saveFollow(); renderChips(); openFollow(); }
      };
      box.append(row);
    }
    for (const fl of [...m.flags].sort((a, b) => a.cell - b.cell || a.frame - b.frame)) {
      const row = document.createElement("div"); row.className = "mk-flag";
      row.innerHTML = `<span class="mk-go" title="Follow ${fl.cell} at frame ${fl.frame}">&#9873; ${fl.cell} &middot; f${fl.frame}</span>` +
        `<input class="mk-note" data-id="${fl.id}" placeholder="note to self" tabindex="-1"><a class="mk-x" title="Remove this flag">&times;</a>`;
      const note = row.querySelector(".mk-note"); note.value = fl.note || "";
      note.onkeydown = (e) => { if (e.key === "Enter" || e.key === "Escape") { e.preventDefault(); note.blur(); } };
      note.onchange = () => markCall({op: "note", id: fl.id, note: note.value}).then(() => status(`Note saved on ${fl.cell} at frame ${fl.frame}`, "#4da3ff"));
      row.querySelector(".mk-go").onclick = () => { S.follow = [fl.cell]; saveFollow(); renderChips(); openFollow(fl.frame); };
      row.querySelector(".mk-x").onclick = () => markCall({op: "unflag", id: fl.id});
      box.append(row);
    }
    if (!ticks.length && !m.flags.length) box.innerHTML = `<span class="cr-muted">V ticks the clicked (or followed) cell as good; J flags it at this frame. Right-click the timeline to flag there, click a cell's number to tick it.</span>`;
  }

  // ------------------------------------------------------------------ seed / grow preview
  // With Seed or Grow on, hovering shades what would be taken at the grow level (filled) and the other free
  // signal at that level (faint); changing the level redraws it (Jamie 2026-11-17).
  const PREVIEW_TOOLS = ["seed", "grow"];
  function previewHover(name, e) {
    if (!PREVIEW_TOOLS.includes(S.tool) || S.pan || e.buttons) return;
    const p = at(name, e);
    if (!p) return;
    S.hover = {y: p.y, x: p.x};
    clearTimeout(S.previewTimer); S.previewTimer = setTimeout(loadPreview, 120);
  }
  async function loadPreview() {
    if (!PREVIEW_TOOLS.includes(S.tool) || !S.hover || !S.cur) { if (S.preview) { S.preview = null; draw(); } return; }
    const [y0, x0] = S.cur.origin, [h, w] = S.cur.crop, level = +$("cr-level").value || 0.2, ticket = ++S.previewTicket;
    try {
      const r = await call("/api/correction-review/grow-preview?" + q({...where(), frame: S.frame, y: S.hover.y, x: S.hover.x,
        level, kind: S.tool, y0, y1: y0 + h, x0, x1: x0 + w}));
      if (ticket !== S.previewTicket) return;
      S.preview = r; status(r.message + "  (change the grow level to retune; click applies)", "#ffe14d"); draw();
    } catch (err) { status(err.message, "#ff9a9a"); }
  }
  function drawPreview(g, ox, oy, gk, origin, frame) {
    const P = S.preview;
    if (!P || P.frame !== frame || !PREVIEW_TOOLS.includes(S.tool)) return;
    const fill = (runs, colour) => { g.fillStyle = colour;
      for (const [y, x, n] of runs) g.fillRect(ox + (x - origin[1]) * gk, oy + (y - origin[0]) * gk, n * gk, gk); };
    fill(P.free, "rgba(77,163,255,0.30)");
    fill(P.take, P.kind === "grow" ? "rgba(126,224,138,0.65)" : "rgba(255,225,77,0.65)");
  }

  // ------------------------------------------------------------------ frames, number, split sides, events, box, track on
  // Jamie 2026-11-18: set the frames and number for a seed; see which side of a split keeps the number; go through
  // and resolve the timeline's events (also gaps only); box-select cells to delete; run the tracking on from here.
  const rangeField = (id) => { const v = $(id).value.trim(); return /^\d+$/.test(v) ? +v : null; };
  const range = () => ({range_from: rangeField("cr-rfrom"), range_to: rangeField("cr-rto")});
  function numberSet() {   // the seed number box, else the Give box (a split always cuts off a free number)
    for (const id of ["cr-snum", "cr-give-box"]) { const v = $(id).value.trim(); if (/^\d+$/.test(v) && +v) return +v; }
    return null;
  }

  // split: live sides while drawing (Shift flips which side keeps the number)
  const CUTS = ["split", "trim"];
  function splitHover(e) {
    if (!CUTS.includes(S.mode) || S.line.length < 2) return;
    S.splitFlip = e.shiftKey;
    clearTimeout(S.splitTimer); S.splitTimer = setTimeout(loadSplitPreview, 90);
  }
  async function loadSplitPreview() {
    if (!CUTS.includes(S.mode) || S.line.length < 2) return;
    const ticket = ++S.splitTicket;
    try {
      const r = await post("/api/correction-review/split-preview", {frame: S.frame, points: S.line, flip: S.splitFlip, number: null});
      if (ticket !== S.splitTicket || !CUTS.includes(S.mode)) return;
      S.splitPreview = r;
      status(S.mode === "trim" ? `Trim ${r.host}: green keeps ${r.host}, orange is erased (hold Shift to swap sides; release to trim)`
        : `Split ${r.host}: green keeps ${r.host}, orange becomes ${r.child || "a free number"} (hold Shift to swap sides; release to cut)`, "#ffe14d");
      draw();
    } catch (err) { S.splitPreview = null; draw(); }
  }
  function drawLine(g, ox, oy, gk, origin, dpr) {   // the line being drawn, redrawn with each picture
    if (!S.line.length || !LINE_MODES.includes(S.mode)) return;
    g.strokeStyle = S.mode === "erase" ? "#ff7b7b" : S.mode === "draw" ? "#7ee08a" : "#ffe14d"; g.lineWidth = 2 * dpr; g.beginPath();
    S.line.forEach(([y, x], j) => { const X = ox + (x - origin[1] + 0.5) * gk, Y = oy + (y - origin[0] + 0.5) * gk; j ? g.lineTo(X, Y) : g.moveTo(X, Y); });
    g.stroke();
  }
  function drawSplit(g, ox, oy, gk, origin, dpr, frame) {
    const P = S.splitPreview;
    if (!P || P.frame !== frame || !CUTS.includes(S.mode)) return;
    const fill = (runs, colour, label) => {
      g.fillStyle = colour; let sy = 0, sx = 0, k = 0;
      for (const [y, x, n] of runs) { g.fillRect(ox + (x - origin[1]) * gk, oy + (y - origin[0]) * gk, n * gk, gk); sy += y * n; sx += (x + n / 2) * n; k += n; }
      if (!k) return;
      const cx = ox + (sx / k - origin[1]) * gk, cy = oy + (sy / k - origin[0] + 0.5) * gk;
      g.font = `bold ${15 * dpr}px Segoe UI`; g.textAlign = "center"; g.textBaseline = "middle";
      g.lineWidth = 3 * dpr; g.strokeStyle = "#000"; g.strokeText(label, cx, cy); g.fillStyle = "#fff"; g.fillText(label, cx, cy);
    };
    fill(P.keep, "rgba(126,224,138,0.55)", String(P.host));
    fill(P.cut, "rgba(255,174,66,0.6)", S.mode === "trim" ? "erase" : P.child ? String(P.child) : "new");
  }

  // timeline events: resolved ones (marks.json) drop out of the timeline, the list and , / . stepping
  const evKey = (n, f, kind) => `${n}:${f}:${kind}`;
  const resolvedSet = () => new Set((S.marks.resolved || []).map(r => evKey(r[0], r[1], r[2])));
  function eventList() {
    if (!S.timeline) return [];
    const done = resolvedSet(), all = [];
    for (const c of markedCells(true)) for (const ev of c.events) {
      if (S.gkind && ev[1] !== S.gkind) continue;
      const resolved = done.has(evKey(c.n, ev[0], ev[1]));
      if (resolved && !$("cr-eshow").checked) continue;
      all.push({n: c.n, f: ev[0], kind: ev[1], text: ev[2], resolved});
    }
    return all.sort((a, b) => a.f - b.f || a.n - b.n);
  }
  function renderEvents() {
    const box = $("cr-elist");
    if (!S.timeline) { box.innerHTML = `<span class="cr-muted">Open to read the timeline...</span>`; return; }
    const list = eventList(), open = list.filter(e => !e.resolved).length;
    $("cr-events-sum").textContent = `Timeline events (${open} open${S.gkind ? ", " + S.gkind : ""})`;
    $("cr-ekind").value = S.gkind || "";
    box.innerHTML = "";
    list.forEach((ev, i) => {
      const row = document.createElement("div");
      row.className = "ev-row" + (i === S.flag ? " on" : "") + (ev.resolved ? " done" : "");
      row.innerHTML = `<input type="checkbox" tabindex="-1" title="Resolved (H)" ${ev.resolved ? "checked" : ""}>` +
        `<i style="background:${FLAGS[ev.kind][0]}"></i><span>${ev.text}</span>`;
      row.querySelector("input").onchange = (e) => resolveEvent(ev, e.target.checked);
      row.querySelector("span").onclick = () => goEvent(i);
      box.append(row);
    });
    const on = box.querySelector(".ev-row.on"); if (on) on.scrollIntoView({block: "nearest"});
  }
  async function resolveEvent(ev, on = true) {
    await markCall({op: on ? "resolve" : "unresolve", cell: ev.n, frame: ev.f, note: ev.kind});
  }
  function goEvent(i) {
    const all = eventList();
    if (!all.length) { status("No timeline events" + (S.gkind ? " of this kind" : "") + " left", MUTED); renderEvents(); return; }
    S.flag = (i + all.length) % all.length;
    const ev = all[S.flag];
    S.follow = [ev.n]; saveFollow(); $("cr-ffrom").value = ""; $("cr-fto").value = "";
    renderEvents();
    openFollow(ev.f).then(() => status(`Event ${S.flag + 1}/${all.length}: ${ev.text} (H resolves it and goes to the next)`, FLAGS[ev.kind][0]));
  }
  async function resolveCurrent() {
    const all = eventList(), ev = all[S.flag];
    if (!ev) { status("Step to an event first with , or . (or click one in Timeline events)", "#ff9a9a"); return; }
    await resolveEvent(ev, !ev.resolved);
    const left = eventList();
    if (!left.length) { status("Every event here is resolved", "#7ee08a"); renderEvents(); return; }
    goEvent(Math.min(S.flag, left.length - 1));
  }
  function setKind(kind) { S.gkind = kind || null; S.flag = -1; if (S.timeline) { renderLegend(); drawGantt(); } renderEvents(); }

  // box select: Shift+drag with Delete or Whole-track delete on queues every cell in the box
  function boxStart(name, e, p) { S.box = {name, a: p, b: p}; }
  function boxMove(name, e) { if (!S.box) return; const p = at(name, e); if (p) { S.box.b = p; requestAnimationFrame(() => draw()); } }
  async function boxEnd() {
    const B = S.box; S.box = null;
    if (!B) return;
    const y0 = Math.min(B.a.y, B.b.y), y1 = Math.max(B.a.y, B.b.y) + 1, x0 = Math.min(B.a.x, B.b.x), x1 = Math.max(B.a.x, B.b.x) + 1;
    const r = await call("/api/correction-review/cells-in-box?" + q({...where(), frame: S.frame, y0, y1, x0, x1}));
    for (const c of r.cells) S.queue.push({tool: S.tool, action: "delete", frame: S.frame, whole: S.tool === "track",
                                           piece_only: $("cr-piece").checked, y: c.y, x: c.x});
    showQueue(); draw();
    status(r.cells.length ? `Queued ${r.cells.length} cells to delete: ${r.cells.map(c => c.n).join(", ")} (Enter deletes them all, Esc clears)` : "No cells in the box", r.cells.length ? "#ffe14d" : MUTED);
  }
  function drawBox(g, ox, oy, gk, origin, dpr, name) {
    const B = S.box; if (!B || B.name !== name) return;
    const y0 = Math.min(B.a.y, B.b.y), x0 = Math.min(B.a.x, B.b.x);
    g.strokeStyle = "#ff7b7b"; g.lineWidth = 2 * dpr; g.setLineDash([5 * dpr, 4 * dpr]);
    g.strokeRect(ox + (x0 - origin[1]) * gk, oy + (y0 - origin[0]) * gk, (Math.abs(B.b.x - B.a.x) + 1) * gk, (Math.abs(B.b.y - B.a.y) + 1) * gk);
    g.setLineDash([]);
  }

  // fill gaps: seed at the cell's last centroid wherever it goes missing, with its number, until it is back
  function fillGaps() {
    const cells = S.picked && S.picked.now && S.picked.frame === S.frame ? [S.picked.now] : S.follow.slice();
    if (!cells.length) { status("Click a cell (or follow cells) first, then Q", "#ff9a9a"); return; }
    status(`Filling the gaps of ${cells.join(", ")} after frame ${S.frame} at grow level ${+$("cr-level").value || 0.2}...`);
    editCall("fill_gaps", {cells});
  }

  // track on: the clicked cell (or the followed cells) carries its number forward from this frame
  function trackOn() {
    const cells = S.picked && S.picked.now && S.picked.frame === S.frame ? [S.picked.now] : S.follow.slice();
    if (!cells.length) { status("Click a cell (or follow cells) first, then L", "#ff9a9a"); return; }
    status(`Tracking ${cells.join(", ")} on from frame ${S.frame}...`);
    editCall("track_on", {cells});
  }

  // ------------------------------------------------------------------ zoom and field of view
  // Zooming in magnifies the picture on screen (z > 1). Zooming out past the item's own crop asks the server for
  // a wider crop (the field, sent as a key suffix), so a cell that wandered out of the crop can be found again.
  const FIELD = "~field:", ZMAX = 16;
  const fkey = (it) => it && S.zoom && S.zoom.base === it.key && S.zoom.field ? it.key + FIELD + S.zoom.field.join(",") : it && it.key;

  function zoomState(it, fr) {
    let Z = S.zoom;
    if (!Z || Z.base !== it.key) Z = S.zoom = {base: it.key, z: 1, vy: null, vx: null, field: null, crop0: null};
    if (!Z.field) Z.crop0 = fr.crop.slice();             // the item's own crop size
    const id = fr.origin.join() + "|" + fr.crop.join();
    if (Z.id !== id) { Z.id = id; Z.vy = fr.crop[0] / 2; Z.vx = fr.crop[1] / 2; }   // a new crop starts centred
    Z.shape = fr.shape; Z.scale = fr.scale; Z.origin = fr.origin; Z.crop = fr.crop;
    return Z;
  }

  function clampView(Z, crop, c, gk) {   // keep the picture on the panel; at z = 1 it is centred
    if (Z.z <= 1) { Z.z = 1; Z.vy = crop[0] / 2; Z.vx = crop[1] / 2; return; }
    const hy = c.height / 2 / gk, hx = c.width / 2 / gk;
    Z.vy = crop[0] <= 2 * hy ? crop[0] / 2 : Math.min(Math.max(Z.vy, hy), crop[0] - hy);
    Z.vx = crop[1] <= 2 * hx ? crop[1] / 2 : Math.min(Math.max(Z.vx, hx), crop[1] - hx);
  }

  function fieldOf(Z) {   // the crop now shown, full-frame [y0, y1, x0, x1]
    return Z.field ? Z.field.slice() : [Z.origin[0], Z.origin[0] + Z.crop[0], Z.origin[1], Z.origin[1] + Z.crop[1]];
  }

  function setField(Z, cy, cx, h, w) {   // a field of h x w centred near (cy, cx), kept inside the image
    const [H, W] = Z.shape;
    h = Math.min(H, Math.max(Z.crop0[0], Math.round(h))); w = Math.min(W, Math.max(Z.crop0[1], Math.round(w)));
    const y0 = Math.min(Math.max(0, Math.round(cy - h / 2)), H - h), x0 = Math.min(Math.max(0, Math.round(cx - w / 2)), W - w);
    Z.field = [y0, y0 + h, x0, x0 + w];
    Z.z = 1;
    clearTimeout(Z.timer); Z.timer = setTimeout(() => draw(), 200);   // one server crop per burst of wheel steps
  }

  function zoomAt(name, e, factor) {
    const g = S.geom[name], Z = S.zoom;
    if (!g || !Z) return;
    const sx = e.offsetX * g.dpr, sy = e.offsetY * g.dpr, c = e.target;
    const py = (sy - g.oy) / g.k, px = (sx - g.ox) / g.k;   // crop pixel under the mouse
    const wider = Z.field && (Z.field[1] - Z.field[0] > Z.crop0[0] || Z.field[3] - Z.field[2] > Z.crop0[1]);
    if ((factor > 1 && !wider) || (factor < 1 && Z.z > 1)) {   // magnify on screen, keeping the point under the mouse
      const z = Math.min(ZMAX, Math.max(1, Z.z * factor)), gk = g.k / Z.z * z;
      Z.z = z < 1.05 ? 1 : z;
      Z.vy = py - (sy - c.height / 2) / gk; Z.vx = px - (sx - c.width / 2) / gk;
      return draw();
    }
    // a wider or narrower field, keeping the point under the mouse at the same place in it
    const [y0, y1, x0, x1] = fieldOf(Z);
    const fy = Math.min(1, Math.max(0, py / (y1 - y0))), fx = Math.min(1, Math.max(0, px / (x1 - x0)));
    const h = (y1 - y0) / factor, w = (x1 - x0) / factor, Py = y0 + py, Px = x0 + px;
    setField(Z, Py - fy * h + h / 2, Px - fx * w + w / 2, h, w);
    status(`Field ${Z.field[3] - Z.field[2]} x ${Z.field[1] - Z.field[0]} px (wheel up narrows it again, 0 resets)`);
  }

  function zoomStep(factor) {   // buttons and keys: about the panel centre
    const c = $("cr-grid").querySelector("canvas");
    if (!c || !S.zoom) return;
    zoomAt(c.dataset.name, {offsetX: c.clientWidth / 2, offsetY: c.clientHeight / 2, target: c}, factor);
  }

  function zoomReset() {
    if (!S.zoom) return;
    S.zoom.field = null; S.zoom.z = 1; S.zoom.id = null;
    status("View reset to the item's own crop"); draw();
  }

  function zoomText(fr) {
    const Z = S.zoom;
    if (!Z) return "";
    if (Z.z > 1) return `  ·  zoom ${Z.z.toFixed(1)}x`;
    return Z.field ? `  ·  field ${fr.crop[1]} x ${fr.crop[0]} px` : "";
  }

  function panStart(name, e) {
    const g = S.geom[name];
    if (g && S.zoom) S.pan = {x: e.clientX, y: e.clientY, vy: S.zoom.vy, vx: S.zoom.vx, k: g.k, dpr: g.dpr};
  }
  function panMove(e) {
    const P = S.pan, Z = S.zoom;
    if (!P || !(e.buttons & 4)) return;
    if (Z.z > 1) { Z.vx = P.vx - (e.clientX - P.x) * P.dpr / P.k; Z.vy = P.vy - (e.clientY - P.y) * P.dpr / P.k; draw(); }
  }
  function panEnd(e) {   // at z = 1 a drag moves the field itself (one new crop on release)
    const P = S.pan, Z = S.zoom; S.pan = null;
    if (!P || !Z || Z.z > 1) return;
    const dy = -(e.clientY - P.y) * P.dpr / P.k, dx = -(e.clientX - P.x) * P.dpr / P.k;
    if (Math.abs(dy) + Math.abs(dx) < 2) return;
    const [y0, y1, x0, x1] = fieldOf(Z);
    setField(Z, (y0 + y1) / 2 + dy, (x0 + x1) / 2 + dx, y1 - y0, x1 - x0);
  }

  function overviewClick(e) {   // the whole-recording map: move the field so it is centred where clicked
    const Z = S.zoom, ov = $("cr-overview");
    if (!Z || !Z.scale) return;
    const y = e.offsetY * ov.height / ov.clientHeight / Z.scale, x = e.offsetX * ov.width / ov.clientWidth / Z.scale;
    const [y0, y1, x0, x1] = fieldOf(Z);
    setField(Z, y, x, y1 - y0, x1 - x0);
    status(`Field moved to x ${Math.round(x)}, y ${Math.round(y)} (0 resets)`);
  }

  // ------------------------------------------------------------------ playback
  const delay = () => 1000 / Math.max(1, Math.min(30, +$("cr-fps").value || 3));
  function stop() { clearTimeout(S.play); S.play = null; $("cr-play").innerHTML = "&#9654;"; }
  function startPlay() {
    stop();
    const it = cur();
    if (!it) return;
    const loop = [];
    if (S.fview) for (let f = S.frame; f <= S.hi; f++) loop.push(f);   // from here to the end, then round again
    if (S.fview) for (let f = S.lo; f < S.frame; f++) loop.push(f);
    else for (let f = Math.max(S.lo, it.frame - 3); f <= Math.min(S.hi, it.frame + 3); f++) loop.push(f);
    let i = S.fview ? 1 : 0;
    const tick = () => {
      S.play = setTimeout(tick, delay());
      S.frame = loop[i % loop.length]; i++;
      $("cr-slider").value = S.frame; draw();
    };
    $("cr-play").innerHTML = "&#9208;";
    S.play = setTimeout(tick, delay());
  }
  function togglePlay() { if (S.play) stop(); else startPlay(); }
  function step(d) { stop(); S.frame = Math.max(S.lo, Math.min(S.hi, S.frame + d)); $("cr-slider").value = S.frame; draw(); }
  function fpsSet(change = 0) {
    const v = Math.max(1, Math.min(30, (+$("cr-fps").value || +setting("fps") || 3) + change));
    $("cr-fps").value = v; setting("fps", v);
    status(`Playback ${v} fps`);
  }

  // ------------------------------------------------------------------ decisions
  async function decide(value) {
    if (!S.shown.length) return;
    if (S.fview) { status("Decisions are for the list's joins: Esc goes back to the list", "#ffe14d"); return; }
    const it = S.shown[S.pos], before = dec(it);
    if (it.kind === "question") { status("A question has no approve/deny: type your answer in its box on the left", "#ffe14d"); return; }
    if ((before ?? null) === value) return nextUndecided();
    const r = await post("/api/correction-review/decide", {key: it.key, value});
    if (!r.ok) { status(r.message, "#ff9a9a"); return; }
    S.undo.push({kind: "decision", stem: S.stem, key: it.key, prev: before ?? null});
    S.set.decisions = r.decisions; S.set.edits = r.edits; S.version++; S.stamp = await readStamp();
    status("");
    updateRow(S.pos); updateProgress(); staleTimeline();
    if (value === null) { showTitle(it); draw(); } else nextUndecided();
  }

  async function undoLast() {
    const last = S.undo.pop();
    if (!last) return;
    if (last.stem !== S.stem) await openWell(last.stem);
    if (last.kind === "edit") {
      const r = await post("/api/correction-review/undo-edit", {count: last.count});
      return reload(r.message, r.ok ? "#c9a6ff" : "#ff9a9a");
    }
    const decisions = {...S.set.decisions};
    if (last.prev === null) delete decisions[last.key]; else decisions[last.key] = last.prev;
    S.set = await call("/api/correction-review/set", {method: "PUT", body: JSON.stringify({...where(), decisions})});
    S.items = S.set.items; S.version++; S.stamp = await readStamp();
    staleTimeline();
    refill();
    const keys = S.shown.map(x => x.key);
    if (keys.includes(last.key)) select(keys.indexOf(last.key));
  }

  function nextUndecided() {
    const n = S.shown.length;
    for (let d = 1; d <= n; d++) {
      const i = (S.pos + d) % n;
      if (dec(S.shown[i]) === undefined) return select(i);
    }
    select(Math.min(S.pos + 1, n - 1));
  }

  async function reload(message, colour) {   // after an edit or undo: fresh lists and pictures
    const fresh = await call("/api/correction-review/set?" + q(where()));
    S.set = fresh; S.items = fresh.items; S.version++; S.stamp = await readStamp();
    S.mode = S.tool && LINE_MODES.includes(S.tool) ? S.tool : null; S.picked = null; $("cr-sel").textContent = "";
    if (message !== undefined) status(message, colour);
    updateProgress(); [...$("cr-rows").children].forEach((_, i) => updateRow(i));
    staleTimeline();
    if (S.preview) loadPreview();   // the seed/grow preview against the edited labels
    if (S.fview) await openFollow(S.frame, true); else draw(true);
  }

  // ------------------------------------------------------------------ clicks and edits
  function at(name, e) {
    const g = S.geom[name];
    if (!g) return null;
    const cy = Math.floor((e.offsetY * g.dpr - g.oy) / g.k), cx = Math.floor((e.offsetX * g.dpr - g.ox) / g.k);
    if (!(cy >= 0 && cy < g.crop[0] && cx >= 0 && cx < g.crop[1])) return null;
    return {y: g.origin[0] + cy, x: g.origin[1] + cx};
  }

  const LINE_MODES = ["split", "trim", "fence", "keepin", "draw", "erase"];
  const AREA_MODES = ["fence", "keepin"];   // a click picks the cell, a drawn outline is the area

  async function loadNumber(name, p) {   // Alt+click: this cell's number goes in the number box (Jamie 2026-11-19)
    const hit = await call("/api/correction-review/pick?" + q({...where(), key: fkey(cur()), frame: S.frame, y: p.y, x: p.x, panel: name}));
    if (!hit.now) { status("Alt+click on a cell to load its number", "#ff9a9a"); return; }
    $("cr-give-box").value = hit.now;
    status(`Loaded ${hit.now}: Draw outline, Give and Seed now use it (Alt+click another cell to change it)`, idColour(hit.now));
  }
  async function pickAt(name, e) {
    const p = at(name, e);
    if (!p) return;
    stop();
    if (e.shiftKey && (S.tool === "delete" || S.tool === "track")) return boxStart(name, e, p);
    if (e.altKey) return loadNumber(name, p);
    if (LINE_MODES.includes(S.mode)) { S.line = [[p.y, p.x]]; S.queueLine = e.ctrlKey && !!S.tool; S.splitPreview = null; return; }
    if (e.ctrlKey && S.tool && S.tool !== "swap") return queueItem({y: p.y, x: p.x});
    const it = cur();
    const hit = await call("/api/correction-review/pick?" + q({...where(), key: fkey(it), frame: S.frame, y: p.y, x: p.x, panel: name}));
    if ((S.mode === "swap" || S.mode === "give") && S.picked && S.picked.now) {
      if (!hit.now) { status("Click on a cell", "#ff9a9a"); return; }
      const mode = S.mode; S.mode = null;
      return mode === "swap" ? editCall("swap", {other: hit.now}) : give(hit.now);
    }
    if (S.tool && !LINE_MODES.includes(S.tool)) {
      S.picked = hit; $("cr-to").value = hit.now ? hit.to : S.frame;
      $("cr-sel").textContent = hit.now ? `cell ${hit.now}, frames ${S.frame}–${hit.to}` : `empty spot, frame ${S.frame}`;
      return applyTool(hit);
    }
    S.picked = hit; S.mode = null;
    if (hit.now) {
      $("cr-to").value = hit.to;
      $("cr-sel").textContent = `cell ${hit.now}, frames ${S.frame}–${hit.to}`;
    } else { $("cr-to").value = S.frame; $("cr-sel").textContent = `empty spot, frame ${S.frame}`; }
    const shown = hit.cell && hit.cell !== hit.now
      ? ` (shown as ${hit.cell} in ${name}: ${name === "After" ? "this join is not applied" : "this join is applied"})` : "";
    status(hit.now ? `Picked cell ${hit.now}${shown} at frame ${S.frame}: edit it (S G X C K W, O draw, + grow), note it (1-6 or N)`
      : `Picked an empty spot at frame ${S.frame}: B seeds a new cell here (to the 'to' frame), O draws one`, "#ffe14d");
    sendFocus();
    draw();
  }

  function dragAt(name, e) {
    if (!LINE_MODES.includes(S.mode) || !S.line.length || !(e.buttons & 1)) return;
    const p = at(name, e);
    if (!p) return;
    S.line.push([p.y, p.x]);
    const g = S.geom[name], c = e.target.getContext("2d");
    const [[ya, xa], [yb, xb]] = S.line.slice(-2);
    c.strokeStyle = S.mode === "erase" ? "#ff7b7b" : S.mode === "draw" ? "#7ee08a" : "#ffe14d";
    c.lineWidth = 2 * g.dpr; c.beginPath();
    c.moveTo(g.ox + (xa - g.origin[1] + 0.5) * g.k, g.oy + (ya - g.origin[0] + 0.5) * g.k);
    c.lineTo(g.ox + (xb - g.origin[1] + 0.5) * g.k, g.oy + (yb - g.origin[0] + 0.5) * g.k); c.stroke();
  }

  const fenceNumber = () => { const v = $("cr-snum").value.trim(); return /^\d+$/.test(v) && +v ? +v : null; };
  async function fencePick(y, x) {   // Fence on: a plain click picks the cell to fence (Jamie 2026-11-18)
    const hit = await call("/api/correction-review/pick?" + q({...where(), key: fkey(cur()), frame: S.frame, y, x}));
    if (!hit.now) { status("Fence: that is not a cell. Click the cell to fence, then draw round the area", "#ff9a9a"); return; }
    S.fenceCell = hit.now;
    status(S.mode === "keepin" ? `Keep in: cell ${hit.now}. Now draw round the area it must stay inside (click another cell to change)`
      : `Fence: cell ${hit.now}. Now draw round the area it must never enter (click another cell to change)`, "#ffe14d");
  }
  function releaseAt(name, e) {
    if (AREA_MODES.includes(S.mode) && S.line.length && S.line.length < 4) {   // a click, not an outline
      const [y, x] = S.line[0]; S.line = []; draw();
      if (fenceNumber()) { status(`Fence uses ${fenceNumber()} from the number box: clear it to pick by clicking`, "#ffe14d"); return; }
      return fencePick(y, x).catch(err => status(err.message, "#ff9a9a"));
    }
    if (!LINE_MODES.includes(S.mode) || S.line.length < 2) return;
    const points = S.line, mode = S.mode, flip = !!(e && e.shiftKey); S.line = []; S.mode = S.tool === mode ? mode : null;
    S.splitPreview = null; clearTimeout(S.splitTimer); S.splitTicket++;
    if (S.queueLine) {
      S.queueLine = false;
      const p = S.picked && S.picked.frame === S.frame && S.picked.now ? [S.picked.y, S.picked.x] : null;
      return queueItem({points, pick: p, flip});
    }
    if (AREA_MODES.includes(mode)) {
      const n = fenceNumber() || S.fenceCell || (S.picked && S.picked.now) || S.follow[0];
      status(`${mode === "fence" ? "Fencing" : "Keeping in"} ${n || "?"}...`);
      return editCall(mode === "fence" ? "fence" : "keep_in", {points, cells: n ? [n] : []});
    }
    if (mode === "split" || mode === "trim") {   // own frames (the cell's run, or 'to frame'); not the seed frames
      status(mode === "trim" ? "Trimming..." : "Splitting...");
      return editCall(mode, {points, flip, number: null, range_from: null, range_to: null});
    }
    status(mode === "split" ? "Splitting..." : mode === "draw" ? "Drawing..." : "Erasing...");
    const typed = $("cr-give-box").value.trim();   // Draw gives the typed number when no cell is clicked
    editCall(mode, mode === "draw" && /^\d+$/.test(typed) ? {points, number: +typed} : {points});
  }

  async function editCall(action, extra = {}) {
    const it = cur();
    try {
      const r = await post("/api/correction-review/edit", {action, picked: S.picked, frame: S.frame,
        to_frame: +$("cr-to").value, key: it ? fkey(it) : null, piece_only: $("cr-piece").checked,
        level: +$("cr-level").value || 0.2, one_piece: $("cr-onepiece").checked, ...range(), ...extra});
      if (!r.ok) { if (r.message) status(r.message, "#ff9a9a"); draw(); return; }
      S.undo.push({kind: "edit", stem: S.stem, count: r.count});
      await reload(r.message, "#c9a6ff");
    } catch (err) { status(err.message, "#ff9a9a"); draw(); }
  }

  const cell = () => { if (!(S.picked && S.picked.now)) { status("Click a cell first", "#ff9a9a"); return null; } return S.picked.now; };
  function startSwap() { if (cell()) { S.mode = "swap"; status(`Click the cell to swap ${S.picked.now} with (Esc cancels)`, "#ffe14d"); } }
  function startGive() {
    if (!cell()) return;
    S.mode = "give"; $("cr-give-box").value = ""; $("cr-give-box").focus();
    status(`Click the cell whose number ${S.picked.now} should take, or type it and press Enter`, "#ffe14d");
  }
  function giveTyped() {
    const text = $("cr-give-box").value.trim();
    $("cr-give-box").blur();
    if (S.tool === "give") {   // sticky Give: the number stays for the next clicks
      if (!/^\d+$/.test(text) || !+text) { status("Type a number above 0", "#ff9a9a"); return; }
      if (S.picked && S.picked.now && S.picked.frame === S.frame) give(+text);
      else status(`Give number on: click each cell to give it ${text} (Esc stops)`, "#ffe14d");
      return;
    }
    if (!/^\d+$/.test(text) || !+text || !cell()) { status("Type a number above 0", "#ff9a9a"); return; }
    S.mode = null; $("cr-give-box").value = "";
    give(+text);
  }
  function give(b) { if (S.picked.now !== b) editCall("give", {number: b}); }
  function deleteCell() { if (cell()) editCall("delete"); }
  function deleteTrack() { if (cell()) { status(`Deleting ${S.picked.now} from every frame...`); editCall("delete", {whole: true}); } }
  function keep(whole) {
    if (!cell()) return;
    const typed = $("cr-give-box").value.trim();
    if (!S.tool) $("cr-give-box").value = "";
    status(`Following ${S.picked.now}${whole ? " over the whole recording" : ""}...`);
    editCall("keep", {whole, number: /^\d+$/.test(typed) ? +typed : null});
  }
  function startLine(mode) {
    stop(); S.mode = mode; S.line = [];
    const who = S.picked && S.picked.now;
    const typed = $("cr-give-box").value.trim();
    status({split: "Draw a line across the cell to cut it (Esc cancels)",
      trim: "Draw a line across the cell: the orange side will be erased (Shift swaps sides; Esc cancels)",
      fence: (S.fenceCell = fenceNumber() || who || (S.follow.length === 1 ? S.follow[0] : null))
        ? `Fence: cell ${S.fenceCell}. Draw round the area it must never enter (click another cell, or type a number in the number box, to change)`
        : "Fence: click the cell to fence (or type its number in the number box), then draw round the area it must never enter",
      keepin: (S.fenceCell = fenceNumber() || who || (S.follow.length === 1 ? S.follow[0] : null))
        ? `Keep in: cell ${S.fenceCell}. Draw round the area it must stay inside (click another cell, or type a number in the number box, to change)`
        : "Keep in: click the cell (or type its number in the number box), then draw round the area it must stay inside",
      draw: `Hold the button and draw round the area: empty pixels inside go to ${who ? "cell " + who : /^\d+$/.test(typed) ? "number " + typed : "a new number"} (this frame; Esc cancels)`,
      erase: `Hold the button and draw round the area to erase ${who ? "cell " + who + "'s pixels" : "every cell's pixels"} inside it (this frame; Esc cancels)`}[mode], "#ffe14d");
  }
  function seed() {
    if (!S.picked || S.picked.y == null) { status("Click an empty spot on the cell's signal first, then B", "#ff9a9a"); return; }
    const {range_from: a, range_to: b} = range(), n = numberSet();
    status(`Seeding at frame ${S.frame}${a != null || b != null ? ` over frames ${a ?? "start"}–${b ?? "end"}` : ": following its signal"}${n ? " as " + n : ""}...`);
    editCall("seed", {number: n});
  }
  function island() { if (cell()) { status(`Keeping the clicked piece of ${S.picked.now}, frames ${S.picked.frame}–${$("cr-to").value}...`); editCall("one_island"); } }
  function mgrow() { if (cell()) { status(`Model grow ${S.picked.now} frames ${S.picked.frame}–${$("cr-to").value} (first use on a recording runs the outline model, about 3-4 minutes, unless it already ran in the background)...`); editCall("model_grow"); } }
  function grow() { if (cell()) { status(`Growing ${S.picked.now} frames ${S.picked.frame}–${$("cr-to").value}...`); editCall("grow"); } }
  function cancelEdit() {
    if (!$("cr-traces").hidden) { $("cr-traces").hidden = true; return; } if (!$("cr-tower").hidden) { $("cr-tower").hidden = true; return; }
    if (S.queue.length) { document.activeElement?.blur(); return clearQueue(); }
    if (S.tool) { document.activeElement?.blur(); return toolOff(); }
    if (S.mode) { S.mode = null; S.line = []; document.activeElement?.blur(); status(""); draw(); return; }
    document.activeElement?.blur();
    if (S.fview) { backToList(); status(""); return; }
    status(""); draw();
  }

  // ------------------------------------------------------------------ notes
  async function saveNote(tag = null) {
    const typed = $("cr-note").value.trim();
    if ((!tag && !typed) || !cur()) return;
    const it = cur();
    const r = await post("/api/correction-review/note", {key: it.key, frame: S.frame, text: typed, tag, picked: S.picked});
    $("cr-note").value = ""; $("cr-note").blur(); S.picked = null;
    S.set.notes = (await call("/api/correction-review/set?" + q(where()))).notes;
    S.stamp = await readStamp();
    if (!S.fview) updateRow(S.pos);
    status(r.message);
    draw();
  }

  function renderQuestions() {   // questions from the page's assistant, each pinned to a frame and cell
    const box = $("cr-questions"), open = S.set.notes.filter(n => n.tag === "Question" && !n.answer);
    box.hidden = !open.length;
    box.innerHTML = open.length ? `<h4>Questions from the assistant (${open.length})</h4>` : "";
    for (const n of open) {
      const div = document.createElement("div");
      div.className = "cr-question";
      div.innerHTML = `<b title="Show this frame">Frame ${n.frame}${n.cell ? `, cell ${n.cell}` : ""}</b><span></span>
        <span class="cr-row"><input placeholder="Your answer"><button>Answer</button></span>`;
      div.querySelector("span").textContent = n.text;
      div.querySelector("b").onclick = () => {
        const key = n.y != null ? `place:${n.frame}:${n.y}:${n.x}` : n.item;
        let i = S.shown.findIndex(it => it.key === key);
        if (i < 0) {   // hidden by the list filter: show everything
          document.querySelector('#cr-filter input[value="all"]').checked = true; refill();
          i = S.shown.findIndex(it => it.key === key);
        }
        const show = () => { stop(); S.frame = Math.max(S.lo, Math.min(S.hi, n.frame)); $("cr-slider").value = S.frame;
          if (n.y != null) S.picked = {frame: n.frame, y: n.y, x: n.x, now: n.cell}; draw(); };
        if (i >= 0 && (i !== S.pos || S.fview)) select(i).then(show); else show();
      };
      div.querySelector("button").onclick = async () => {
        const text = div.querySelector("input").value.trim();
        if (!text) return;
        const notes = S.set.notes.map(m => m === n ? {...m, answer: text, answered: new Date().toISOString()} : m);
        await call("/api/correction-review/set", {method: "PUT", body: JSON.stringify({...where(), notes})});
        S.set.notes = notes; S.stamp = await readStamp(); renderQuestions();
        status("Answered: " + text);
      };
      box.append(div);
    }
  }

  // ------------------------------------------------------------------ edit history (Motion's history table)
  const opCells = (op) => [...new Set(["a", "b", "from", "to", "cell", "host", "child", "only"].map(k => op[k]).filter(v => v > 0))];
  function describe(op) {
    const span = op.first === op.last ? `frame ${op.first}` : `frames ${op.first}–${op.last}`;
    const what = {swap: `swap ${op.a} ↔ ${op.b}`, renumber: `${op.from} → ${op.to}`, delete: `delete ${op.cell}`,
      piece: `piece of ${op.host} → ${op.child || "deleted"}`, split: `split ${op.host} → ${op.child}`,
      keep: `keep ${op.cell}${op.to && op.to !== op.cell ? " as " + op.to : ""}`,
      paint: op.island ? `one island ${op.only}` : op.to ? `${op.model ? "model grow" : op.level ? "grow/seed" : "draw"} ${op.to}` : `erase ${op.only || "all"}`}[op.op] || op.op;
    return `${what}, ${span}${op.by && op.by !== "jamie" ? " (" + op.by + ")" : ""}`;
  }
  function renderHistory() {
    const list = ops(), box = $("cr-hist");
    $("cr-history").querySelector("summary").textContent = `Edits (${list.length})${list.length ? ": click one to see it" : ""}`;
    box.innerHTML = "";
    list.map((op, i) => [op, i]).reverse().forEach(([op, i]) => {
      const div = document.createElement("div");
      div.className = "cr-hist-row"; div.textContent = `${i + 1}. ${describe(op)}`;
      div.title = (op.time || "") + (i === list.length - 1 ? "  (Ctrl+Z undoes this one)" : "");
      div.onclick = () => {
        const cells = opCells(op);
        if (!cells.length) { status("This edit names no cell to show", "#ff9a9a"); return; }
        S.follow = cells; saveFollow(); $("cr-ffrom").value = ""; $("cr-fto").value = "";
        openFollow(op.first);
      };
      box.append(div);
    });
  }

  // ------------------------------------------------------------------ timeline (Gantt chart of every cell)
  const ROW = 9, GUT = 44, TOP = 16;
  function staleTimeline() { S.timeline = null; if (!$("cr-gantt-box").hidden) loadTimeline(); }
  async function loadTimeline() {
    $("cr-glegend").textContent = "Reading every frame...";
    try { S.timeline = await call("/api/correction-review/cells?" + q(where())); }
    catch (e) { $("cr-glegend").textContent = e.message; return; }
    renderLegend(); drawGantt(); renderEvents();
  }
  function toggleTimeline() {
    const box = $("cr-gantt-box");
    box.hidden = !box.hidden; setting("timeline", box.hidden ? "" : "1");
    if (!box.hidden && !S.timeline) loadTimeline();
    setTimeout(() => draw(), 0);
  }
  function renderLegend() {
    const counts = {}, box = $("cr-glegend");   // open (unresolved) events
    for (const c of markedCells()) for (const ev of c.events) counts[ev[1]] = (counts[ev[1]] || 0) + 1;
    box.innerHTML = "";
    for (const [kind, [colour, tip]] of Object.entries(FLAGS)) {
      const b = document.createElement("button");
      b.className = "flag" + (S.gkind === kind ? " on" : ""); b.tabIndex = -1;
      b.innerHTML = `<i style="background:${colour}"></i>${kind} ${counts[kind] || 0}`;
      b.title = `${tip}. Click: list only cells with this flag and step through them with , and .`;
      b.onclick = () => setKind(S.gkind === kind ? null : kind);
      box.append(b);
    }
  }
  function ganttRows() {
    let cells = markedCells();
    if ($("cr-gticked").checked) cells = cells.filter(c => !ticked(c.n));
    if (S.gkind) cells = cells.filter(c => c.events.some(e => e[1] === S.gkind));
    else if ($("cr-gevents").checked) cells = cells.filter(c => c.events.length);
    const order = $("cr-gsort").value;
    const key = {first: c => c.runs[0][0] * 1e5 + c.n, flags: c => -c.events.length * 1e5 + c.n, n: c => c.n}[order];
    cells = [...cells].sort((a, b) => key(a) - key(b));
    const followed = cells.filter(c => S.follow.includes(c.n));
    return followed.concat(cells.filter(c => !S.follow.includes(c.n)));
  }
  function drawGantt() {
    const box = $("cr-gantt-box");
    if (box.hidden || !S.timeline) return;
    const c = $("cr-gantt"), dpr = window.devicePixelRatio || 1, T = S.timeline.T;
    const rows = S.rows = ganttRows();
    const W = $("cr-gantt-scroll").clientWidth - 2, H = TOP + rows.length * ROW + 4;
    c.style.width = W + "px"; c.style.height = H + "px"; c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#15171a"; g.fillRect(0, 0, W, H);
    const px = (W - GUT - 6) / T, X = (f) => GUT + f * px;
    g.font = "10px Segoe UI"; g.textBaseline = "middle";
    g.fillStyle = MUTED; g.textAlign = "center";
    const stepT = T > 160 ? 24 : T > 60 ? 12 : 4;
    for (let f = 0; f < T; f += stepT) { g.fillText(f, X(f), 7); g.fillStyle = "#2a2e34"; g.fillRect(X(f), TOP, 1, H - TOP); g.fillStyle = MUTED; }
    const rowOf = new Map(rows.map((r, i) => [r.n, i]));
    rows.forEach((r, i) => {
      const y = TOP + i * ROW, followed = S.follow.includes(r.n), col = followed ? idColour(r.n) : "#59616d";
      if (followed) { g.fillStyle = "#23272d"; g.fillRect(0, y, W, ROW); }
      g.fillStyle = ticked(r.n) ? "#7ee08a" : followed ? col : MUTED; g.textAlign = "right";
      g.fillText((ticked(r.n) ? "\u2713" : "") + r.n, GUT - 4, y + ROW / 2);
      g.fillStyle = col;
      for (const [a, b] of r.runs) g.fillRect(X(a), y + 2, Math.max(1, (b - a + 1) * px), ROW - 4);
      for (const [f, kind] of r.events) {
        if (kind === "gap" || (S.gkind && kind !== S.gkind)) continue;
        g.fillStyle = FLAGS[kind][0]; g.fillRect(X(f) - 1, y + 1, Math.max(3, px), ROW - 2);
      }
      for (const [, b] of r.runs.slice(0, -1)) { g.fillStyle = FLAGS.gap[0]; g.fillRect(X(b + 1), y + ROW / 2, 2, 1); }
    });
    g.globalAlpha = 0.9;
    for (const j of S.timeline.joins) for (const n of j.cells) if (rowOf.has(n)) {   // joins: thin cyan tick on the top edge
      g.fillStyle = j.kind === "unsure" ? "#ffcf5a" : "#00dcff"; g.fillRect(X(j.frame), TOP + rowOf.get(n) * ROW, 1, 3);
    }
    for (const e of S.timeline.edits) for (const n of e.cells) if (rowOf.has(n)) {   // your edits: purple underline
      g.fillStyle = "#c9a6ff"; g.fillRect(X(e.first), TOP + rowOf.get(n) * ROW + ROW - 2, Math.max(2, (e.last - e.first + 1) * px), 2);
    }
    for (const qn of S.timeline.questions) if (rowOf.has(qn.cell)) {
      g.strokeStyle = "#ffe14d"; g.beginPath(); g.arc(X(qn.frame), TOP + rowOf.get(qn.cell) * ROW + ROW / 2, 4, 0, 2 * Math.PI); g.stroke();
    }
    g.globalAlpha = 1;
    if (S.fview || S.shown.length) { g.fillStyle = "#ffffff"; g.fillRect(X(S.frame), TOP, 1, H - TOP); }
    if (S.fview) {   // the frames the followed cells are in
      const [a, b] = S.fview.present;
      g.fillStyle = "rgba(255,225,77,0.10)"; g.fillRect(X(a), TOP, (b - a + 1) * px, H - TOP);
    }
  }
  function ganttAt(e) {
    if (!S.timeline) return null;
    const W = $("cr-gantt").clientWidth, px = (W - GUT - 6) / S.timeline.T;
    const i = Math.floor((e.offsetY - TOP) / ROW), f = Math.max(0, Math.min(S.timeline.T - 1, Math.floor((e.offsetX - GUT) / px)));
    const r = S.rows[i];
    return r ? {r, f} : null;
  }
  function ganttHover(e) {
    const hit = ganttAt(e), c = $("cr-gantt");
    if (!hit) { c.title = ""; return; }
    const {r, f} = hit;
    const near = r.events.filter(ev => Math.abs(ev[0] - f) <= 2).map(ev => ev[2]);
    const runs = r.runs.map(([a, b]) => a === b ? a : `${a}–${b}`).join(", ");
    const mine = S.marks.flags.filter(fl => fl.cell === r.n && Math.abs(fl.frame - f) <= 2).map(fl => `\u2691 your flag at frame ${fl.frame}${fl.note ? ": " + fl.note : ""}`);
    c.title = `Cell ${r.n}${ticked(r.n) ? " (ticked as good)" : ""}: frames ${runs}` + (near.length ? "\n" + near.join("\n") : "") +
      (mine.length ? "\n" + mine.join("\n") : "") +
      `\nClick: follow ${r.n} at frame ${f}. Shift+click: add it. Click its number: tick as good. Right-click: flag it here`;
  }
  function ganttClick(e) {
    const hit = ganttAt(e);
    if (!hit) return;
    if (e.offsetX < GUT) return tickCells([hit.r.n]);   // the cell's number: tick / untick it as good
    if (e.shiftKey) { if (!S.follow.includes(hit.r.n)) S.follow = S.follow.concat([hit.r.n]); }
    else S.follow = [hit.r.n];
    saveFollow(); $("cr-ffrom").value = ""; $("cr-fto").value = "";
    openFollow(hit.f);
  }
  function stepFlag(d) {   // , and . : the previous / next open timeline event, in frame order
    if (!S.timeline) { if ($("cr-gantt-box").hidden) toggleTimeline(); return; }
    return goEvent(S.flag + d);
    const all = [];
    for (const c of markedCells()) for (const ev of c.events) if (!S.gkind || ev[1] === S.gkind) all.push({n: c.n, f: ev[0], kind: ev[1], text: ev[2]});
    all.sort((a, b) => a.f - b.f || a.n - b.n);
    if (!all.length) { status("No flagged events" + (S.gkind ? " of this kind" : ""), MUTED); return; }
    S.flag = (S.flag + d + all.length) % all.length;
    const ev = all[S.flag];
    S.follow = [ev.n]; saveFollow(); $("cr-ffrom").value = ""; $("cr-fto").value = "";
    openFollow(ev.f).then(() => status(`Flag ${S.flag + 1}/${all.length}: ${ev.text}`, FLAGS[ev.kind][0]));
  }

  // ------------------------------------------------------------------ tracking tower of the followed cells
  async function showTower() {
    stop();   // the whole field and every cell; followed cells emphasised
    const {lo, hi} = S.follow.length ? followRange() : {lo: "", hi: ""};
    $("cr-tower").hidden = false; $("cr-tower-img").removeAttribute("src");
    $("cr-tower-title").textContent = `Tower of every cell: drawing...`;
    try {
      const t = await call("/api/correction-review/tower?" + q({...where(), cells: S.follow.join(","), lo, hi,
        style: $("cr-tower-style").value, preset: $("cr-tower-preset").value}));
      $("cr-tower-img").src = t.image;
      $("cr-tower-title").innerHTML = `Tower of every cell` + (t.cells.length ? `, emphasising ` +
        t.cells.map(n => `<span style="color:${t.colours[n]}">${n}</span>`).join(", ") : "") +
        `, frames ${t.lo}–${t.hi}, a plane every ${t.snapshot_hours} h`;
      const box = $("cr-tower-frames"); box.innerHTML = "";
      for (const f of t.frames) {
        const b = document.createElement("button"); b.textContent = f; b.title = `Show frame ${f}`;
        b.onclick = () => {
          $("cr-tower").hidden = true;
          if (S.follow.length) return openFollow(f);
          if (f < S.lo || f > S.hi) { status(`Frame ${f} is outside this join's frames: right-click a cell to see the whole recording`, "#ff9a9a"); return; }
          stop(); S.frame = f; $("cr-slider").value = f; draw();
        };
        box.append(b);
      }
    } catch (e) { $("cr-tower-title").textContent = e.message; }
  }

  // ------------------------------------------------------------------ outputs
  async function exportSet() {
    status(`Exporting ${S.stem} and saving it to the ground-truth folder...`);
    try {
      const r = await post("/api/correction-review/save-truth", {});
      status(r.message, r.export_note.startsWith("Motion export skipped") ? "#ffe14d" : "#7ee08a");
    } catch (e) { status(`Export failed: ${e.message}`, "#ff9a9a"); }
  }
  // Render: the whole recording with the corrected outlines, made in the background (correction_outputs.py)
  async function render() {
    try {
      const r = await post("/api/correction-review/render", {});
      status(r.message + " (Movie opens it when done)", "#ffe14d");
      clearInterval(S.renderTimer);
      S.renderTimer = setInterval(async () => {
        const st = await call("/api/correction-review/render-status?" + q(where())).catch(() => null);
        const j = st && st.job;
        if (!j) return;
        if (j.state === "running") { status(`Rendering: frame ${j.done} of ${j.total}`, "#ffe14d"); return; }
        clearInterval(S.renderTimer);
        if (j.state === "done") status(`Corrected movie ready (${j.total} frames, ${j.edits} edits): press Movie`, "#7ee08a");
        else status(`Render failed: ${j.error}`, "#ff9a9a");
      }, 1500);
    } catch (e) { status(`Render failed: ${e.message}`, "#ff9a9a"); }
  }
  async function openMovie() {
    const st = await call("/api/correction-review/render-status?" + q(where())).catch(() => null);
    if (!st || !st.latest) { status(`No corrected movie yet for ${S.stem}: press Render first`, "#ff9a9a"); return; }
    window.open("/api/correction-review/movie?" + q({...where(), v: st.latest}), "_blank");
    status(`Opened ${st.latest}` + (st.job && st.job.state === "running" ? " (a newer one is still rendering)" : ""));
  }

  // trace grid: every cell's red signal from the current outlines, like the Workbench's All cell traces
  async function showTraces(reload = true) {
    stop(); $("cr-traces").hidden = false;
    if (reload || !S.traces) {
      $("cr-traces-title").textContent = "Trace grid: measuring the current outlines...";
      try { S.traces = await call("/api/correction-review/traces?" + q(where())); }
      catch (e) { $("cr-traces-title").textContent = e.message; return; }
    }
    drawTraces();
  }
  function drawTraces() {
    const T = S.traces, light = $("cr-traces-light").value, k = +$("cr-traces-smooth").value;
    const shared = $("cr-traces-scale").value === "recording";
    let cells = T.cells;
    if ($("cr-traces-which").value === "follow") cells = cells.filter(c => S.follow.includes(c.n));
    const smooth = (v) => v.map((_, i) => {   // centred mean of k frames; a frame without the cell stays a gap
      if (v[i] == null) return null;
      let s = 0, n = 0;
      for (let j = i - (k >> 1); j <= i + (k >> 1); j++) if (v[j] != null) { s += v[j]; n++; }
      return s / n;
    });
    const series = cells.map(c => ({n: c.n, v: smooth(c[light])}));
    const all = series.flatMap(s => s.v.filter(x => x != null));
    const glo = Math.min(...all), ghi = Math.max(...all);
    const grid = $("cr-traces-grid"), colSel = $("cr-traces-cols").value;
    const cols = colSel === "auto" ? Math.max(1, Math.floor(grid.clientWidth / 230)) : +colSel;
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`; grid.innerHTML = "";
    $("cr-traces-title").textContent = `Trace grid: ${series.length} cells, red ${light === "total" ? "total in mask" : "mean per pixel"}` +
      `${k > 1 ? `, ${k}-frame mean` : ""} (current outlines; click a cell to follow it)`;
    const x0 = T.hours[0], x1 = T.hours[T.hours.length - 1] || 1;
    for (const s of series) {
      const box = document.createElement("div"); box.className = "cr-trace";
      box.innerHTML = `<b style="color:${idColour(s.n)}">${s.n}</b>`;
      const c = document.createElement("canvas"); box.append(c); grid.append(box);
      box.onclick = () => { if (!S.follow.includes(s.n)) S.follow.push(s.n); saveFollow(); $("cr-traces").hidden = true; refreshFollow(); };
      const dpr = window.devicePixelRatio || 1, w = c.clientWidth * dpr, h = c.clientHeight * dpr;
      c.width = w; c.height = h;
      const g = c.getContext("2d"), vals = s.v.filter(x => x != null);
      const lo = shared ? glo : Math.min(...vals), hi = shared ? ghi : Math.max(...vals);
      const X = t => 4 * dpr + (t - x0) / (x1 - x0 || 1) * (w - 8 * dpr), Y = v => h - 4 * dpr - (v - lo) / (hi - lo || 1) * (h - 8 * dpr);
      g.strokeStyle = idColour(s.n); g.lineWidth = 1.2 * dpr; g.beginPath();
      let on = false;
      s.v.forEach((v, i) => { if (v == null) { on = false; return; } on ? g.lineTo(X(T.hours[i]), Y(v)) : g.moveTo(X(T.hours[i]), Y(v)); on = true; });
      g.stroke();
      g.strokeStyle = "rgba(255,225,77,0.5)"; g.lineWidth = dpr; g.beginPath();   // the frame you are on
      g.moveTo(X(T.hours[S.frame] ?? x0), 0); g.lineTo(X(T.hours[S.frame] ?? x0), h); g.stroke();
    }
  }

  // ------------------------------------------------------------------ the page assistant and other writers
  let focusTimer;
  function sendFocus() {
    clearTimeout(focusTimer);
    focusTimer = setTimeout(() => fetch("/api/correction-review/focus", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({...where(),
        key: cur()?.key, frame: S.frame, cell: S.picked})}).catch(() => {}), 400);
  }
  async function readStamp() { return (await call("/api/correction-review/stamp?" + q(where()))).stamp; }
  async function watch() {
    if (document.hidden || !S.set || S.busy) return;
    const now = await readStamp().catch(() => null);
    if (now === null || now === S.stamp) return;
    const edits = S.set.edits.length, notes = S.set.notes.length;
    await reload();
    const e = S.set.edits.length - edits, n = S.set.notes.length - notes;
    status(`Updated by the assistant: ${e} edit${e === 1 ? "" : "s"}, ${n} note${n === 1 ? "" : "s"}`, "#c9a6ff");
  }

  // ------------------------------------------------------------------ keys and wiring
  function onKey(e) {
    const k = e.key.toLowerCase(), ctrl = e.ctrlKey || e.metaKey;
    const t = e.target;
    if (t.tagName === "INPUT" && (t.type === "number" || !t.type || t.type === "text")) {
      if (t.id === "cr-note") { if (k === "enter") { e.preventDefault(); saveNote(); } else if (k === "escape") { t.value = ""; t.blur(); } }
      if (t.id === "cr-give-box") { if (k === "enter") { e.preventDefault(); giveTyped(); } else if (k === "escape") cancelEdit(); }
      if (["cr-fps", "cr-to", "cr-level", "cr-rfrom", "cr-rto", "cr-snum"].includes(t.id) && k === "enter") { if (t.id === "cr-fps") fpsSet(); t.blur(); }
      if ((t.id === "cr-ffrom" || t.id === "cr-fto") && k === "enter") { t.blur(); openFollow(); }
      return;
    }
    if (t.tagName === "SELECT") return;
    if (k === "n") { stop(); $("cr-note").focus(); e.preventDefault(); return; }
    if (!S.set) return;
    const go = (fn) => { e.preventDefault(); fn(); };
    if (ctrl && k === "z") return go(undoLast);
    if (ctrl || e.altKey) return;
    if (k === "x" && e.shiftKey) return go(() => useTool("track"));
    if (k === "enter" && S.queue.length) return go(runQueue);
    if (k === "v") return go(() => tickCells());
    if (k === "j") return go(() => flagCells());
    if (k === "h") return go(resolveCurrent);
    if (k === "l") return go(trackOn);
    if (k === "q") return go(fillGaps);
    if (k === "-" || k === "_") return go(() => zoomStep(0.8));
    if (k === "0") return go(zoomReset);
    const map = {a: () => decide("approve"), d: () => decide("deny"), u: () => decide(null), enter: nextUndecided,
      arrowdown: () => select(S.pos + 1), arrowup: () => select(S.pos - 1), arrowright: () => step(1),
      arrowleft: () => step(-1), " ": togglePlay, "[": () => fpsSet(-1), "]": () => fpsSet(1), e: exportSet,
      r: render, m: openMovie, s: () => useTool("swap"), g: () => useTool("give"), x: () => useTool("delete"),
      delete: () => useTool("delete"), c: () => useTool("split"), z: () => useTool("trim"), k: () => useTool("keep"), w: () => useTool("whole"),
      escape: cancelEdit, o: () => useTool("draw"), i: () => useTool("erase"), b: () => useTool("seed"),
      "+": () => useTool("grow"), "=": () => useTool("grow"), "*": () => useTool("mgrow"), "!": () => useTool("island"),
      f: () => openFollow(), t: toggleTimeline, y: showTower, ",": () => stepFlag(-1), ".": () => stepFlag(1),
      p: () => { $("cr-piece").checked = !$("cr-piece").checked; status("Piece only: " + ($("cr-piece").checked ? "on" : "off")); }};
    if (/^[1-6]$/.test(k)) return go(() => saveNote(S.set.note_kinds[+k - 1].label));
    if (map[k]) go(map[k]);
  }

  function wire() {
    $("cr-stem").onchange = () => openWell($("cr-stem").value).catch(e => status(e.message, "#ff9a9a"));
    document.querySelectorAll("#cr-filter input").forEach(r => { r.onchange = () => refill(); });
    $("cr-approve").onclick = () => decide("approve"); $("cr-deny").onclick = () => decide("deny");
    $("cr-clear").onclick = () => decide(null); $("cr-undo").onclick = undoLast;
    $("cr-export").onclick = exportSet; $("cr-render").onclick = render; $("cr-traces-open").onclick = () => showTraces(true);
    $("cr-traces-close").onclick = () => { $("cr-traces").hidden = true; };
    for (const id of ["cr-traces-light", "cr-traces-smooth", "cr-traces-scale", "cr-traces-cols", "cr-traces-which"]) $(id).onchange = () => showTraces(false);
    $("cr-movie").onclick = openMovie;
    $("cr-play").onclick = togglePlay;
    $("cr-fps").value = setting("fps") || 3;
    $("cr-fps").onchange = () => fpsSet();
    $("cr-slider").oninput = () => { const f = +$("cr-slider").value; if (f !== S.frame) { stop(); S.frame = f; draw(); } };
    for (const t of Object.keys(TOOLS)) $(TOOLS[t][0]).onclick = () => useTool(t);
    $("cr-qrun").onclick = runQueue; $("cr-qclear").onclick = clearQueue;
    $("cr-trackon").onclick = trackOn; $("cr-fillgaps").onclick = fillGaps;
    for (const type of ["keydown", "keyup"]) document.addEventListener(type, (e) => {   // Shift flips the split sides live
      if (e.key === "Shift" && CUTS.includes(S.mode) && S.line.length >= 2) { S.splitFlip = type === "keydown"; loadSplitPreview(); }
    });
    $("cr-ekind").onchange = () => setKind($("cr-ekind").value);
    $("cr-eshow").onchange = () => { S.flag = -1; renderEvents(); };
    $("cr-events").ontoggle = () => { if ($("cr-events").open && !S.timeline) loadTimeline(); else renderEvents(); };
    $("cr-eresolve").onclick = resolveCurrent; $("cr-eprev").onclick = () => stepFlag(-1); $("cr-enext").onclick = () => stepFlag(1);
    $("cr-sticky").checked = setting("sticky") !== "0";
    $("cr-sticky").onchange = () => { setting("sticky", $("cr-sticky").checked ? "1" : "0"); if (!$("cr-sticky").checked) toolOff("Tools are one-shot again"); };

    $("cr-level").value = setting("level") || 0.2;
    $("cr-level").onchange = () => setting("level", +$("cr-level").value || 0.2);
    $("cr-level").oninput = () => { clearTimeout(S.previewTimer); S.previewTimer = setTimeout(loadPreview, 150); };
    $("cr-fshow").onclick = () => openFollow();
    $("cr-fclear").onclick = () => { S.follow = []; saveFollow(); if (S.fview) backToList(); else draw(); drawGantt(); };
    $("cr-every").checked = setting("every") === "1";
    $("cr-every").onchange = () => { setting("every", $("cr-every").checked ? "1" : ""); draw(); };
    $("cr-timeline").onclick = toggleTimeline; $("cr-tower-open").onclick = showTower;
    $("cr-zin").onclick = () => zoomStep(1.25); $("cr-zout").onclick = () => zoomStep(0.8); $("cr-zfit").onclick = zoomReset;
    $("cr-overview").addEventListener("click", overviewClick);
    window.addEventListener("mouseup", e => { if (e.button === 1 && S.pan) panEnd(e); });
    $("cr-tower-close").onclick = () => { $("cr-tower").hidden = true; };
    $("cr-tower-style").onchange = showTower; $("cr-tower-preset").onchange = showTower;
    $("cr-gevents").onchange = drawGantt; $("cr-gsort").onchange = drawGantt;
    $("cr-gprev").onclick = () => stepFlag(-1); $("cr-gnext").onclick = () => stepFlag(1);
    $("cr-gantt").addEventListener("mousemove", ganttHover);
    $("cr-gantt").addEventListener("click", ganttClick);
    $("cr-gantt").addEventListener("contextmenu", (e) => { e.preventDefault(); const hit = ganttAt(e); if (hit) flagCells([hit.r.n], hit.f); });
    $("cr-gticked").onchange = drawGantt;
    $("cr-tick").onclick = () => tickCells(); $("cr-flag").onclick = () => flagCells();
    $("cr-gantt-box").hidden = setting("timeline") !== "1";
    document.querySelectorAll(".cr-show input").forEach(b => {
      b.checked = (setting("channels") || "").split(",").includes(b.dataset.ch);
      b.onchange = () => { setting("channels", [...document.querySelectorAll(".cr-show input:checked")].map(x => x.dataset.ch).join()); draw(true); };
    });
    document.addEventListener("keydown", onKey);
    let resized; window.addEventListener("resize", () => { clearTimeout(resized); resized = setTimeout(() => draw(), 150); });
    setInterval(() => watch().catch(() => {}), 4000);
  }

  function quickNotes(kinds) {
    $("cr-quick").innerHTML = "";
    kinds.forEach(({label, tip}, i) => {
      const b = document.createElement("button");
      b.textContent = `${i + 1} ${label}`; b.tabIndex = -1;
      b.title = `${tip} (${i + 1}). Click a cell first to attach it; any typed text is added.`;
      b.onclick = () => saveNote(label);
      $("cr-quick").append(b);
    });
  }

  async function start() {
    wire();
    quickNotes([["Swap", "Two numbers swap cells"], ["Same cell", "Two numbers are one cell"],
      ["Split", "One cell split over several numbers, or one number over several cells"],
      ["Not a cell", "No cell here: delete this number"],
      ["Missed join", "The number stops and the cell comes back under another number"],
      ["Wrong number", "This cell should carry another number"]].map(([label, tip]) => ({label, tip})));
    const root = await defaultRoot();
    if (!root) { status("Open this page from the workbench's Motion tracks sidebar (no editing folder known)", "#ff9a9a"); return; }
    openRoot(root).catch(e => status(e.message, "#ff9a9a"));
  }
  start();
})();
