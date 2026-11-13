"use strict";
// Analyse recordings: drop TIFF stacks or a folder, label them, run the full
// automated chain (Auto-Organotypic's stages, then cells, tracking and
// measurement), then open the measured results.
(() => {
  const LABELS = [["name","Name"],["condition","Condition"],["treatment","Treatment"],["sample_id","Sample"],["notes","Notes"]];
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[char]);

  function mount(host, {api, onOpenRun, onJob}) {
    const view = {data:null, message:"", error:"", job:null, dirty:new Set(), saveTimer:null, busy:false, visible:false};
    const native = () => !!window.pywebview?.api?.choose_recordings;

    function rows() { return view.data?.recordings || []; }
    function selected() { return rows().filter(row => row.selected); }

    async function add(paths) {
      paths = (paths || []).map(path => String(path).trim()).filter(Boolean);
      if (!paths.length) return;
      view.busy = true; view.error = ""; view.message = "Reading what was dropped…"; render();
      try {
        await flushLabels();
        view.data = await api("/api/intake/inspect", {method:"POST", body:JSON.stringify({paths})});
        const ready = selected().filter(row => row.valid).length;
        view.message = `${ready} of ${rows().length} recordings ready. Add labels, then run.`;
        try { localStorage.setItem("pm-intake-folder", view.data.folder); } catch { /* optional */ }
      } catch (error) { view.error = error.message; view.message = ""; }
      view.busy = false; render();
    }

    function labelsOf(row) { return {...(row.labels || {}), name: row.name || ""}; }
    function scheduleSave(row) {
      view.dirty.add(row.key); clearTimeout(view.saveTimer);
      view.saveTimer = setTimeout(() => flushLabels().catch(error => { view.error = error.message; render(); }), 700);
    }
    async function flushLabels() {
      clearTimeout(view.saveTimer);
      if (!view.data || !view.dirty.size) return;
      const labels = {};
      for (const key of view.dirty) { const row = rows().find(item => item.key === key); if (row) labels[key] = labelsOf(row); }
      view.dirty.clear();
      await api("/api/intake/labels", {method:"POST", body:JSON.stringify({folder:view.data.folder, labels})});
      const note = host.querySelector("#intake-saved"); if (note) note.textContent = "Labels saved beside the recordings";
    }

    async function run() {
      const chosen = selected();
      if (!chosen.length) throw new Error("Tick at least one recording to analyse.");
      const blocked = chosen.filter(row => !row.valid && !(row.issues || []).every(issue => /frame interval/i.test(issue) && Number(row.interval_s) > 0));
      if (blocked.length) throw new Error(`${blocked[0].stem}: ${(blocked[0].issues || []).join("; ")}`);
      await flushLabels();
      view.job = await api("/api/intake/run", {method:"POST", body:JSON.stringify({folder:view.data.folder, mode:view.data.mode, recordings:rows()})});
      onJob?.(view.job); watch(); render();
    }

    function watch() {
      clearInterval(view.timer);
      view.timer = setInterval(async () => {
        try { view.job = await api(`/api/jobs/${view.job.id}`); } catch (error) { view.error = error.message; clearInterval(view.timer); }
        if (["completed","failed"].includes(view.job?.state)) {
          clearInterval(view.timer);
          if (view.job.state === "completed" && view.data) view.data.results = view.job.results || view.data.results;
        }
        if (view.visible) render();
      }, 2000);
    }

    function statusBlock() {
      const job = view.job;
      if (!job) return "";
      const progress = job.progress || {};
      const done = (progress.stages || []).map(stage => stage.stage);
      const state = job.state === "running" ? `Running · ${esc(progress.stage || "starting")} · chain step ${progress.completed || 0} of ${progress.total || "?"}${(progress.completed || 0) >= (progress.total || Infinity) - 1 ? " · tracking and measuring cells" : ""}`
        : job.state === "queued" ? "Waiting for the previous analysis to finish"
        : job.state === "completed" ? "Finished. Open a recording below." : `Stopped: ${esc(job.error || "unknown error")}`;
      return `<section class="intake-status ${esc(job.state)}" aria-live="polite"><div><strong>${state}</strong>${done.length ? `<small>${done.map(esc).join(" → ")}</small>` : ""}</div>${["queued","running"].includes(job.state) ? `<button class="button secondary" data-intake="stop" ${job.stop ? "disabled" : ""}>${job.stop ? "Stopping after this step…" : "Stop"}</button>` : ""}</section>`;
    }

    function dropZone() {
      const compact = !!view.data;
      return `<section class="intake-drop ${compact ? "compact" : ""}" data-intake-drop><div><strong>Drop a TIFF stack or a folder of recordings</strong><span>${native() ? "Anywhere on this window. Files are read in place; nothing is copied." : "Your browser hides where dropped files are: paste the path here."}</span></div><form class="intake-path" data-intake-form><input id="intake-path" placeholder="File or folder path" aria-label="TIFF file or folder path" value="${esc(compact ? "" : (localStorage.getItem("pm-intake-folder") || ""))}"><button class="button secondary" type="submit">Read</button>${native() ? `<button class="button secondary" type="button" data-intake="files">Choose stacks…</button><button class="button secondary" type="button" data-intake="folder">Choose folder…</button>` : ""}</form></section>`;
    }

    function channelSelect(row) {
      if (view.data.mode !== "local" || (row.channels || []).length < 2) return row.channels?.length ? esc(row.channels[0].name || "1") : "";
      return `<select data-intake-role="${esc(row.key)}" aria-label="Signal channel for ${esc(row.stem)}">${row.channels.map(channel => `<option value="${channel.index}" ${Number(row.roles?.signal) === channel.index ? "selected" : ""}>${channel.index} · ${esc(channel.name || "unnamed")}</option>`).join("")}</select>`;
    }

    function intervalCell(row) {
      const missing = row.interval_s == null || (row.issues || []).some(issue => /frame interval/i.test(issue));
      if (view.data.mode === "local" && missing) return `<input type="number" min="0" step="any" class="intake-interval" data-intake-interval="${esc(row.key)}" value="${esc(row.interval_s ?? "")}" placeholder="seconds" aria-label="Seconds between frames for ${esc(row.stem)}">`;
      return row.interval_s == null ? "—" : `${(Number(row.interval_s) / 60).toFixed(Number(row.interval_s) % 60 ? 1 : 0)} min`;
    }

    function table() {
      const data = view.data;
      if (!data) return "";
      const issueText = row => (row.issues || []).filter(issue => !(/frame interval/i.test(issue) && Number(row.interval_s) > 0)).join("; ");
      return `<section class="intake-table-wrap"><div class="intake-head"><div><strong>${esc(data.folder)}</strong><small>${rows().length} recording${rows().length === 1 ? "" : "s"} · ${data.mode === "manifest" ? "microscope manifest" : "TIFF stacks"} · <span id="intake-saved">Labels are saved beside the recordings</span></small></div><div class="intake-bulk"><input id="intake-bulk-condition" placeholder="Condition" aria-label="Condition for ticked recordings"><button class="button secondary" data-intake="bulk">Set for ticked</button></div></div>
        <div class="intake-scroll"><table class="intake-table"><thead><tr><th><input type="checkbox" data-intake="all" aria-label="Tick all recordings" ${rows().every(row => row.selected) ? "checked" : ""}></th><th>File</th>${LABELS.map(([, label]) => `<th>${label}</th>`).join("")}<th>Signal</th><th>Frames</th><th>Interval</th><th>Ready</th></tr></thead><tbody>${rows().map(row => `<tr class="${row.selected ? "" : "skipped"}"><td><input type="checkbox" data-intake-select="${esc(row.key)}" ${row.selected ? "checked" : ""} aria-label="Analyse ${esc(row.stem)}"></td><td class="intake-file" title="${esc(row.path)}">${esc(row.key)}</td>${LABELS.map(([key, label]) => `<td><input data-intake-label="${key}" data-intake-key="${esc(row.key)}" value="${esc(key === "name" ? row.name : row.labels?.[key] || "")}" placeholder="${key === "name" ? esc(row.stem) : ""}" aria-label="${label} for ${esc(row.stem)}"></td>`).join("")}<td>${channelSelect(row)}</td><td>${row.frames ?? "—"}</td><td>${intervalCell(row)}</td><td>${issueText(row) ? `<span class="intake-issue" title="${esc(issueText(row))}">${esc(issueText(row))}</span>` : `<span class="intake-ok">Ready</span>`}</td></tr>`).join("")}</tbody></table></div>
        <div class="intake-actions"><button class="button primary" data-intake="run" ${view.job && ["queued","running"].includes(view.job.state) ? "disabled" : ""}>Run full analysis on ${selected().length} recording${selected().length === 1 ? "" : "s"}</button><small>Registration, cell masks, tracking, measurement, cell grids and videos, with the saved defaults. Every step and setting is recorded in the run folder.</small></div></section>`;
    }

    function resultsBlock() {
      const results = view.data?.results || [];
      if (!results.length) return "";
      return `<section class="intake-results"><h2>Finished analyses in this folder</h2>${results.map(result => `<div class="intake-result"><div><strong>${esc(result.label)}</strong><small>${esc(new Date(result.modified * 1000).toLocaleString())}</small></div><div class="intake-open">${result.recordings.map(item => `<button class="button secondary" data-intake-open="${esc(item.measurement)}" title="Open ${esc(item.stem)} in the linked views">${esc(nameFor(item.stem))}</button>`).join("") || "<small>No measured recordings</small>"}</div></div>`).join("")}</section>`;
    }

    function nameFor(stem) {
      const row = [...rows()].sort((a, b) => b.stem.length - a.stem.length).find(item => stem === item.stem || stem.startsWith(item.stem + "_"));
      return row?.name || stem;
    }

    function render() {
      if (!view.visible) return;
      const focus = document.activeElement?.closest?.("#intake-host") ? {key:document.activeElement.dataset.intakeKey, label:document.activeElement.dataset.intakeLabel, start:document.activeElement.selectionStart} : null;
      host.innerHTML = `${statusBlock()}${dropZone()}${view.error ? `<div class="intake-error" role="alert">${esc(view.error)}</div>` : view.message ? `<p class="intake-message" role="status">${esc(view.message)}</p>` : ""}${table()}${resultsBlock()}`;
      if (focus?.key && focus.label) {
        const again = host.querySelector(`[data-intake-key="${CSS.escape(focus.key)}"][data-intake-label="${focus.label}"]`);
        if (again) { again.focus(); try { again.setSelectionRange(focus.start, focus.start); } catch { /* not a text field */ } }
      }
    }

    function rowFor(key) { return rows().find(row => row.key === key); }
    host.addEventListener("input", event => {
      const target = event.target;
      if (target.dataset.intakeLabel) {
        const row = rowFor(target.dataset.intakeKey); if (!row) return;
        if (target.dataset.intakeLabel === "name") row.name = target.value;
        else row.labels = {...(row.labels || {}), [target.dataset.intakeLabel]: target.value};
        scheduleSave(row);
      }
      if (target.dataset.intakeInterval) { const row = rowFor(target.dataset.intakeInterval); if (row) row.interval_s = target.value === "" ? null : Number(target.value); }
    });
    host.addEventListener("change", event => {
      const target = event.target;
      if (target.dataset.intakeSelect) { const row = rowFor(target.dataset.intakeSelect); if (row) row.selected = target.checked; render(); }
      if (target.dataset.intake === "all") { rows().forEach(row => { row.selected = target.checked; }); render(); }
      if (target.dataset.intakeRole) { const row = rowFor(target.dataset.intakeRole); if (row) row.roles = {...(row.roles || {}), signal:Number(target.value)}; }
      if (target.dataset.intakeInterval) render();
    });
    host.addEventListener("submit", event => {
      if (!event.target.matches("[data-intake-form]")) return;
      event.preventDefault(); add([host.querySelector("#intake-path").value]);
    });
    host.addEventListener("click", event => {
      const button = event.target.closest("button"); if (!button) return;
      const action = button.dataset.intake, open = button.dataset.intakeOpen;
      const perform = task => { view.error = ""; Promise.resolve().then(task).catch(error => { view.error = error.message; render(); }); };
      if (open) { perform(async () => { await flushLabels(); onOpenRun?.(open); }); return; }
      if (action === "run") perform(run);
      if (action === "stop") perform(async () => { view.job = await api(`/api/jobs/${view.job.id}/stop`, {method:"POST", body:"{}"}); render(); });
      if (action === "files") perform(async () => add(await window.pywebview.api.choose_recordings(view.data?.folder || null)));
      if (action === "folder") perform(async () => { const folder = await window.pywebview.api.choose_folder(view.data?.folder || null); if (folder) await add([folder]); });
      if (action === "bulk") {
        const value = host.querySelector("#intake-bulk-condition").value;
        selected().forEach(row => { row.labels = {...(row.labels || {}), condition:value}; view.dirty.add(row.key); });
        perform(async () => { await flushLabels(); render(); });
      }
    });
    window.addEventListener("pywebviewready", () => render());

    return {
      add,
      open() { view.visible = true; host.hidden = false; render(); },
      close() { view.visible = false; host.hidden = true; flushLabels().catch(() => {}); },
      get visible() { return view.visible; },
      follow(job) { view.job = job; watch(); render(); },
      hint(text) { view.error = text; render(); },
    };
  }
  window.PyMicrogliaIntake = {mount};
})();
