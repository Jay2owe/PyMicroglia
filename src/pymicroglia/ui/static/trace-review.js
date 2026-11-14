// Trace review: one grid of saved rhythm traces, Jamie's include/exclude marks, and a handoff.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const COLOURS = ['--v0', '--v1', '--v2', '--v3', '--v4'].map((v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim());
  const SORTS = [
    ['p', 'p-value'], ['q', 'q-value'], ['period', 'Period'], ['period24', 'Distance from 24 h'], ['amplitude', 'Amplitude'],
    ['phase', 'Phase'], ['cycles', 'Cycles observed'], ['points', 'Observed points'], ['spike', 'Spike score'],
    ['signal', 'Mean raw signal'], ['delta', 'Period change between first two variants'], ['decision', 'Decision'],
    ['recording', 'Recording, then cell'], ['identity', 'Cell number'],
  ];
  const params = new URLSearchParams(location.search);
  const state = {
    home: params.get('home') || '', listing: null, shown: [], overlay: false, metrics: [], series: [], grid: null, decisions: null,
    selected: new Set(), anchor: null, focus: null, sort: 'recording', descending: false, filter: 'all', recording: '',
    rows: 4, cols: 'auto', images: false, fit: true, order: [], frame: 0, playing: null, crops: new Map(), saving: Promise.resolve(),
  };
  const keyStore = 'pymicroglia-trace-review';
  try { Object.assign(state, JSON.parse(localStorage.getItem(keyStore) || '{}')); } catch (_) { /* per-viewer convenience only */ }
  state.selected = new Set();
  state.crops = new Map();
  state.saving = Promise.resolve();
  if (params.get('home')) state.home = params.get('home');
  const remember = () => {
    try { localStorage.setItem(keyStore, JSON.stringify({ rows: state.rows, cols: state.cols, images: state.images, fit: state.fit, sort: state.sort, descending: state.descending })); } catch (_) { /* ignore */ }
  };

  async function api(path, options = {}) {
    const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || response.statusText);
    return body;
  }
  const status = (text) => { $('tr-status').textContent = text || ''; };
  const query = (values) => new URLSearchParams(values).toString();

  // ---------------------------------------------------------------- loading
  async function openHome(path) {
    status('Opening…');
    try {
      let listing;
      try {
        listing = await api('/api/trace-review?' + query({ home: path }));
      } catch (error) {
        if (!/no trace review/i.test(error.message)) throw error;
        const created = await api('/api/trace-review/create', { method: 'POST', body: JSON.stringify({ run: path }) });
        path = created.home;
        listing = await api('/api/trace-review?' + query({ home: path }));
      }
      state.home = path;
      history.replaceState(null, '', '?' + query({ home: path }));
      $('tr-open').hidden = true;
      state.listing = listing;
      $('tr-title').textContent = listing.review.title || 'Trace review';
      document.title = (listing.review.title || 'Trace') + ' review';
      const saved = await api('/api/trace-review/decisions?' + query({ home: path }));
      const preferred = listing.variants.some((v) => v.key === saved.preferred_variant) ? saved.preferred_variant : null;
      state.shown = listing.variants.length ? [preferred || listing.variants[0].key] : [];
      buildVariantChips();
      buildRecordingFilter();
      await loadGrid();
    } catch (error) {
      $('tr-open').hidden = false;
      $('tr-open-path').value = path;
      $('tr-open-message').textContent = error.message;
      status('');
    }
  }

  async function loadGrid() {
    if (!state.shown.length) {
      state.grid = { cells: [], variants: [], series: [] };
      state.series = [];
      state.decisions = await api('/api/trace-review/decisions?' + query({ home: state.home }));
      $('tr-grid').innerHTML = '<div class="tr-empty">No saved rhythm analyses for these recordings yet. Ask an agent to run the rhythm analysis (or import an older trace folder), then reload.</div>';
      updateCounts();
      return;
    }
    status('Loading traces…');
    const grid = await api('/api/trace-review/grid?' + query({ home: state.home, variants: state.shown.join(','), metrics: state.metrics.join(',') }));
    state.grid = grid;
    state.series = grid.series;
    state.decisions = grid.decisions;
    if (!state.decisions.preferred_variant) state.decisions.preferred_variant = null;
    state.metrics = [...new Set(grid.series.map((s) => s.metric).filter(Boolean))];
    buildMetricChips(new Set(grid.variants.flatMap((v) => v.metrics)));
    computeRange();
    render();
    status('');
  }

  function buildVariantChips() {
    const holder = $('tr-variants');
    holder.innerHTML = '';
    state.listing.variants.forEach((variant, index) => {
      const chip = document.createElement('label');
      chip.className = 'tr-chip';
      chip.style.setProperty('--c', COLOURS[index % COLOURS.length]);
      chip.title = `${variant.description}\nCovers ${variant.coverage} of ${state.listing.recordings.length} recordings` + (variant.stale ? `\n${variant.stale} made from measurements that have since changed` : '');
      const name = variant.descriptor.label || variant.description;
      chip.innerHTML = `<input type="checkbox" value="${variant.key}"><b>${variant.short}</b><span>${esc(name)}</span>` + (variant.stale ? `<small>${variant.stale} stale</small>` : '');
      const box = chip.querySelector('input');
      box.checked = state.shown.includes(variant.key);
      chip.classList.toggle('on', box.checked);
      box.addEventListener('change', () => {
        let note = '';
        if (box.checked) {
          // Without overlay a tick switches the view; with overlay it adds,
          // unless the time window differs: different windows never share an axis.
          const sameWindow = state.shown.every((k) => windowKey(k) === windowKey(variant.key));
          state.shown = state.overlay && sameWindow ? [...state.shown, variant.key] : [variant.key];
          if (state.overlay && !sameWindow) note = 'Different time window: switched view (windows are not overlaid).';
        } else {
          state.shown = state.shown.filter((k) => k !== variant.key);
        }
        holder.querySelectorAll('input').forEach((b) => { b.checked = state.shown.includes(b.value); b.parentElement.classList.toggle('on', b.checked); });
        markWindows();
        loadGrid().then(() => note && status(note)).catch((e) => status(e.message));
      });
      holder.appendChild(chip);
    });
  }

  const windowKey = (key) => {
    const d = state.listing.variants.find((v) => v.key === key)?.descriptor || {};
    return `${d.window_start_hours ?? ''}-${d.window_end_hours ?? ''}`;
  };

  function markWindows() {
    // With overlay on, variants on another time window can only be switched to.
    const current = state.shown.length ? windowKey(state.shown[0]) : null;
    $('tr-variants').querySelectorAll('.tr-chip').forEach((chip) => {
      chip.classList.toggle('other-window', state.overlay && current !== null && windowKey(chip.querySelector('input').value) !== current);
    });
  }

  function buildMetricChips(available) {
    const holder = $('tr-metrics');
    holder.innerHTML = '';
    holder.hidden = available.size < 2;
    for (const metric of available) {
      const chip = document.createElement('label');
      chip.className = 'tr-chip';
      chip.style.setProperty('--c', '#c9d6dd');
      chip.innerHTML = `<input type="checkbox" value="${esc(metric)}"><span>${esc(metric)}</span>`;
      const box = chip.querySelector('input');
      box.checked = state.metrics.includes(metric);
      chip.classList.toggle('on', box.checked);
      box.addEventListener('change', () => {
        if (box.checked) state.metrics = state.overlay ? [...state.metrics, metric] : [metric];
        else state.metrics = state.metrics.filter((m) => m !== metric);
        if (!state.metrics.length) state.metrics = [metric];
        loadGrid().catch((e) => status(e.message));
      });
      holder.appendChild(chip);
    }
    markWindows();
  }

  function buildRecordingFilter() {
    $('tr-recording').innerHTML = '<option value="">All</option>' + state.listing.recordings.map((r) => `<option value="${esc(r.key)}">${esc(r.stem)}</option>`).join('');
    $('tr-recording').parentElement.hidden = state.listing.recordings.length < 2;
  }

  // -------------------------------------------------------------- ordering
  // A series is one variant's analysis of one metric; overlays draw several.
  const seriesOf = (id) => state.series.find((s) => s.id === id);
  const variantOf = (id) => state.listing.variants.find((v) => v.key === seriesOf(id)?.variant);
  const colourOf = (id) => COLOURS[Math.max(0, state.series.findIndex((s) => s.id === id)) % COLOURS.length];
  const seriesLabel = (id) => {
    const metrics = new Set(state.series.map((s) => s.metric));
    const variants = new Set(state.series.map((s) => s.variant));
    return [variants.size > 1 ? variantOf(id).short : '', metrics.size > 1 ? seriesOf(id).metric : ''].filter(Boolean).join(' ');
  };
  const seriesIds = () => state.series.map((s) => s.id);
  const primary = () => state.series[0]?.id;
  const mark = (cell) => (state.decisions.cells[cell.key] || { status: 'unreviewed', star: false, note: '' });
  const suggestion = (cell) => state.decisions.suggestions?.[cell.key];
  const stat = (cell, key = primary()) => cell.stats[key] || {};
  const stale = (cell) => state.shown.some((k) => state.listing.variants.find((v) => v.key === k)?.recordings[cell.recording]?.fresh === false);

  function sortValue(cell) {
    const s = stat(cell);
    const q = cell.quality[primary()] || {};
    switch (state.sort) {
      case 'period24': return s.period == null ? null : Math.abs(s.period - 24);
      case 'points': return q.points;
      case 'spike': return q.spike;
      case 'signal': return q.signal;
      case 'delta': {
        const [a, b] = seriesIds();
        const pa = cell.stats[a]?.period; const pb = cell.stats[b]?.period;
        return pa == null || pb == null ? null : Math.abs(pa - pb);
      }
      case 'decision': return { include: 0, unreviewed: 1, exclude: 2 }[mark(cell).status] - (mark(cell).star ? 0.5 : 0);
      case 'recording': return state.listing.recordings.findIndex((r) => r.key === cell.recording) * 1e6 + cell.identity;
      case 'identity': return cell.identity;
      default: return s[state.sort];
    }
  }

  function visibleCells() {
    const cells = state.grid.cells.filter((cell) => {
      if (state.recording && cell.recording !== state.recording) return false;
      const m = mark(cell);
      switch (state.filter) {
        case 'unreviewed': return m.status === 'unreviewed';
        case 'include': case 'exclude': return m.status === state.filter;
        case 'star': return m.star;
        case 'suggested': return !!suggestion(cell);
        case 'rhythmic': return stat(cell).status === 'rhythmic';
        case 'stale': return stale(cell);
        default: return true;
      }
    });
    const direction = state.descending ? -1 : 1;
    // Missing values always sort last, whichever the direction.
    return cells.sort((a, b) => {
      const x = sortValue(a); const y = sortValue(b);
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return (x - y) * direction || a.identity - b.identity;
    });
  }

  // --------------------------------------------------------------- drawing
  let range = [0, 1]; let rangeSeen = false;
  function computeRange() {
    // The axis only widens during a session, so switching to a windowed
    // variant keeps its traces where they sit in the whole recording.
    let [lo, hi] = rangeSeen ? range : [Infinity, -Infinity];
    for (const cell of state.grid.cells) for (const trace of Object.values(cell.traces)) for (const h of trace.hours) if (h != null) { lo = Math.min(lo, h); hi = Math.max(hi, h); }
    rangeSeen = Number.isFinite(lo) && hi > lo;
    range = rangeSeen ? [lo, hi] : [0, 1];
  }

  function drawTrace(canvas, cell, { cursorHours = null, axes = false } = {}) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth; const height = canvas.clientHeight;
    if (!width || !height) return;
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    const g = canvas.getContext('2d');
    g.setTransform(ratio, 0, 0, ratio, 0, 0);
    g.clearRect(0, 0, width, height);
    const pad = axes ? { l: 4, r: 4, t: 6, b: 16 } : { l: 2, r: 2, t: 3, b: 3 };
    const keys = seriesIds();
    const x = (h) => pad.l + ((h - range[0]) / (range[1] - range[0])) * (width - pad.l - pad.r);
    if (axes) {
      g.fillStyle = '#89a5b3'; g.font = '10px Segoe UI, sans-serif';
      const step = (range[1] - range[0]) > 96 ? 24 : 12;
      for (let h = Math.ceil(range[0] / step) * step; h <= range[1]; h += step) {
        g.fillRect(x(h), height - pad.b, 1, 3); g.fillText(`${h} h`, x(h) + 2, height - 4);
      }
    } else {
      // Faint 24 h guides help judge period by eye.
      g.fillStyle = '#ffffff10';
      for (let h = Math.ceil(range[0] / 24) * 24; h <= range[1]; h += 24) g.fillRect(x(h), 0, 1, height);
    }
    keys.forEach((key) => {
      const trace = cell.traces[key];
      if (!trace) return;
      const values = trace.value.filter((v) => v != null).concat((state.fit && trace.fit) ? trace.fit.filter((v) => v != null) : []);
      if (!values.length) return;
      let lo = Math.min(...values); let hi = Math.max(...values);
      if (hi === lo) { hi += 1; lo -= 1; }
      const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (height - pad.t - pad.b);
      const line = (series, dashed) => {
        g.beginPath(); let open = false;
        series.forEach((v, i) => {
          const h = trace.hours[i];
          if (v == null || h == null) { open = false; return; }
          if (open) g.lineTo(x(h), y(v)); else { g.moveTo(x(h), y(v)); open = true; }
        });
        g.setLineDash(dashed ? [4, 3] : []); g.stroke();
      };
      const colour = colourOf(key);
      g.lineWidth = axes ? 1.4 : 1.1; g.strokeStyle = colour; line(trace.value, false);
      if (state.fit && trace.fit) { g.lineWidth = 1; g.strokeStyle = keys.length > 1 ? colour : '#ffffffaa'; line(trace.fit, true); }
    });
    g.setLineDash([]);
    if (cursorHours != null) { g.fillStyle = '#ffffffcc'; g.fillRect(x(cursorHours) - 0.5, 0, 1.5, height - (axes ? pad.b : 0)); }
  }

  function fmt(value, digits = 3) {
    if (value == null) return '–';
    return Math.abs(value) < 0.001 && value !== 0 ? value.toExponential(1) : Number(value.toPrecision(digits)).toString();
  }

  function tileHeader(cell) {
    const s = stat(cell); const m = mark(cell); const sug = suggestion(cell);
    const parts = [`<b>${esc(cell.stem)} #${cell.identity}</b>`];
    if (m.star) parts.push('<span class="star">★</span>');
    const shownStats = seriesIds().map((key) => {
      const v = cell.stats[key]; if (!v) return '';
      const name = seriesLabel(key);
      const label = name ? `<b style="color:${colourOf(key)}">${esc(name)}</b> ` : '';
      return `${label}${v.status === 'rhythmic' ? '' : (v.status === 'not rhythmic' ? 'n.s. ' : '? ')}${fmt(v.period)} h p ${fmt(v.p, 2)}`;
    }).filter(Boolean).join(' · ');
    parts.push(`<span class="stat" title="${esc(s.reason || '')}">${shownStats || 'not in this variant'}</span>`);
    if (stale(cell)) parts.push('<span class="stale" title="Measurements changed after this analysis">stale</span>');
    if (sug && m.status === 'unreviewed') parts.push(`<span class="sugg ${sug.status}" title="${esc(sug.agent)}: ${esc(sug.reason)}">→${sug.status === 'include' ? 'in' : 'out'}</span>`);
    return parts.join(' ');
  }

  let observer = null;
  function render() {
    const grid = $('tr-grid');
    state.order = visibleCells();
    const width = grid.clientWidth - 8;
    const cols = state.cols === 'auto' ? Math.max(1, Math.floor(width / 240)) : Number(state.cols);
    const height = Math.max(70, Math.floor((grid.clientHeight - 8 - (state.rows - 1) * 4) / state.rows));
    grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    grid.style.gridAutoRows = `${height}px`;
    if (observer) observer.disconnect();
    observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
      if (entry.isIntersecting) paintTile(entry.target);
    }), { root: grid, rootMargin: '200px' });
    grid.innerHTML = '';
    if (!state.order.length) { grid.innerHTML = '<div class="tr-empty">No cells match this view.</div>'; }
    const fragment = document.createDocumentFragment();
    for (const cell of state.order) {
      const tile = document.createElement('article');
      tile.className = 'tr-tile';
      tile.dataset.key = cell.key;
      tile.innerHTML = `<header></header><canvas class="trace"></canvas>${state.images ? '<div class="thumbs"></div>' : ''}`;
      decorate(tile, cell);
      fragment.appendChild(tile);
      observer.observe(tile);
    }
    grid.appendChild(fragment);
    updateCounts();
    if (state.order.length) inspect(state.order.some((c) => c.key === state.focus) ? state.focus : state.order[0].key);
  }

  function decorate(tile, cell) {
    const m = mark(cell);
    tile.classList.toggle('include', m.status === 'include');
    tile.classList.toggle('exclude', m.status === 'exclude');
    tile.classList.toggle('selected', state.selected.has(cell.key));
    tile.classList.toggle('focused', state.focus === cell.key);
    tile.querySelector('header').innerHTML = tileHeader(cell);
  }

  const cellByKey = (key) => state.grid.cells.find((c) => c.key === key);
  function paintTile(tile) {
    const cell = cellByKey(tile.dataset.key);
    if (!cell) return;
    drawTrace(tile.querySelector('canvas'), cell, { cursorHours: state.focus === cell.key ? hoursAt(cell, state.frame) : null });
    const thumbs = tile.querySelector('.thumbs');
    if (thumbs && !thumbs.childElementCount) {
      sampleFrames(cell, 5).forEach((frame) => {
        const img = document.createElement('img');
        img.alt = `frame ${frame}`;
        thumbs.appendChild(img);
        crop(cell, frame, 48).then((r) => { if (r.available) img.src = r.image; }).catch(() => {});
      });
    }
  }

  function refreshTile(key) {
    const tile = $('tr-grid').querySelector(`[data-key="${CSS.escape(key)}"]`);
    const cell = cellByKey(key);
    if (tile && cell) { decorate(tile, cell); paintTile(tile); }
  }

  function updateCounts() {
    const cells = state.grid?.cells || [];
    const counts = { include: 0, exclude: 0, unreviewed: 0, star: 0 };
    for (const cell of cells) { const m = mark(cell); counts[m.status] += 1; if (m.star) counts.star += 1; }
    const suggested = cells.filter((c) => suggestion(c) && mark(c).status === 'unreviewed').length;
    $('tr-counts').textContent = `${cells.length} cells · ${counts.include} included · ${counts.exclude} excluded · ${counts.star} starred · ${counts.unreviewed} unreviewed` +
      (state.order.length !== cells.length ? ` · ${state.order.length} shown` : '');
    $('tr-selection').textContent = `${state.selected.size} selected`;
    $('tr-accept').hidden = !suggested;
    $('tr-accept').textContent = `Accept suggestions (${suggested})`;
  }

  // ------------------------------------------------------------- decisions
  function applyMarks(keys, change) {
    if (!keys.length) { status('Select cells first.'); return; }
    const changes = keys.map((key) => {
      const current = state.decisions.cells[key] || { status: 'unreviewed', star: false, note: '' };
      const next = typeof change === 'function' ? change(current, key) : change;
      return { cell: key, ...next };
    });
    for (const row of changes) {
      const current = { status: 'unreviewed', star: false, note: '', ...(state.decisions.cells[row.cell] || {}) };
      state.decisions.cells[row.cell] = { ...current, ...row };
    }
    keys.forEach(refreshTile);
    updateCounts();
    if (keys.includes(state.focus)) showInspectorMark();
    save(changes);
  }

  function save(changes, extra = {}) {
    state.saving = state.saving.then(async () => {
      try {
        const saved = await api('/api/trace-review/decisions', { method: 'PUT', body: JSON.stringify({ home: state.home, revision: state.decisions.revision, changes, ...extra }) });
        state.decisions.revision = saved.revision;
        state.decisions.updated_at = saved.updated_at;
        status(`Saved ${new Date().toLocaleTimeString()}`);
      } catch (error) {
        status(`Not saved: ${error.message}`);
        state.decisions = await api('/api/trace-review/decisions?' + query({ home: state.home }));
        render();
      }
    });
    return state.saving;
  }

  const targets = () => (state.selected.size ? [...state.selected] : (state.focus ? [state.focus] : []));
  const setStatus = (value) => applyMarks(targets(), { status: value });
  const toggleStar = () => {
    const keys = targets();
    const allStarred = keys.every((k) => state.decisions.cells[k]?.star);
    applyMarks(keys, { star: !allStarred });
  };

  // ------------------------------------------------------------- selection
  function select(key, event) {
    const order = state.order.map((c) => c.key);
    if (event.shiftKey && state.anchor) {
      const [a, b] = [order.indexOf(state.anchor), order.indexOf(key)].sort((p, q) => p - q);
      if (!(event.ctrlKey || event.metaKey)) state.selected.clear();
      order.slice(a, b + 1).forEach((k) => state.selected.add(k));
    } else if (event.ctrlKey || event.metaKey) {
      if (state.selected.has(key)) state.selected.delete(key); else state.selected.add(key);
      state.anchor = key;
    } else {
      state.selected = new Set([key]);
      state.anchor = key;
    }
    $('tr-grid').querySelectorAll('.tr-tile').forEach((tile) => tile.classList.toggle('selected', state.selected.has(tile.dataset.key)));
    inspect(key);
    updateCounts();
  }

  // ------------------------------------------------------------- inspector
  const sampleFrames = (cell, count) => {
    const frames = framesOf(cell);
    if (!frames.length) return [];
    return [...new Set(Array.from({ length: count }, (_, i) => frames[Math.round((i * (frames.length - 1)) / Math.max(1, count - 1))]))];
  };
  function framesOf(cell) {
    const trace = cell.traces[primary()] || Object.values(cell.traces)[0];
    return trace ? trace.frame.filter((f, i) => trace.value[i] != null) : [];
  }
  function hoursAt(cell, frame) {
    const trace = cell.traces[primary()] || Object.values(cell.traces)[0];
    if (!trace || !trace.frame.length) return null;
    let best = 0;
    trace.frame.forEach((f, i) => { if (Math.abs(f - frame) < Math.abs(trace.frame[best] - frame)) best = i; });
    const h = trace.hours[best];
    if (h == null) return null;
    const step = trace.frame.length > 1 ? (trace.hours[trace.hours.length - 1] - trace.hours[0]) / (trace.frame[trace.frame.length - 1] - trace.frame[0] || 1) : 0;
    return h + (frame - trace.frame[best]) * step;
  }

  function crop(cell, frame, size) {
    const id = `${cell.key}|${frame}|${size}`;
    if (!state.crops.has(id)) {
      if (state.crops.size > 1500) state.crops.delete(state.crops.keys().next().value);
      state.crops.set(id, api('/api/trace-review/crop?' + query({ run: cell.run, stem: cell.stem, identity: cell.identity, frame, size })));
    }
    return state.crops.get(id);
  }

  function inspect(key) {
    const previous = state.focus;
    state.focus = key;
    if (previous && previous !== key) refreshTile(previous);
    refreshTile(key);
    const cell = cellByKey(key);
    if (!cell) return;
    $('tr-cell-name').textContent = `${cell.stem} #${cell.identity}`;
    const frames = framesOf(cell);
    $('tr-frame').min = frames[0] ?? 0; $('tr-frame').max = frames[frames.length - 1] ?? 0;
    if (state.frame < frames[0] || state.frame > frames[frames.length - 1]) state.frame = frames[0] ?? 0;
    showInspectorMark();
    renderThumbs(cell);
    renderProvenance(cell);
    showFrame(state.frame);
  }

  function showInspectorMark() {
    const cell = cellByKey(state.focus); if (!cell) return;
    const m = mark(cell);
    $('tr-cell-mark').textContent = (m.star ? '★ ' : '') + m.status;
    $('tr-cell-mark').style.color = m.status === 'include' ? 'var(--include)' : m.status === 'exclude' ? 'var(--exclude)' : 'var(--muted)';
    if (document.activeElement !== $('tr-note')) $('tr-note').value = m.note || '';
    const sug = suggestion(cell);
    $('tr-suggestion').hidden = !sug;
    if (sug) $('tr-suggestion').textContent = `Agent suggests ${sug.status} (${sug.agent}): ${sug.reason}`;
  }

  function renderThumbs(cell) {
    const holder = $('tr-thumbs');
    holder.innerHTML = '';
    sampleFrames(cell, 5).forEach((frame) => {
      const figure = document.createElement('figure');
      const hours = hoursAt(cell, frame);
      figure.innerHTML = `<img alt="frame ${frame}"><figcaption>${hours == null ? frame : fmt(hours, 3) + ' h'}</figcaption>`;
      figure.addEventListener('click', () => { stop(); showFrame(frame); });
      holder.appendChild(figure);
      crop(cell, frame, Number($('tr-crop').value)).then((r) => { if (r.available) figure.querySelector('img').src = r.image; }).catch(() => {});
    });
  }

  function renderProvenance(cell) {
    const rows = seriesIds().map((key) => {
      const variant = variantOf(key);
      const analysis = variant.recordings[cell.recording];
      const s = cell.stats[key] || {};
      const d = variant.descriptor;
      const items = [
        ['Rhythm', s.status ? `${s.status}${s.reason ? ' — ' + s.reason : ''}` : 'not analysed in this variant'],
        ['Period', s.period == null ? '–' : `${fmt(s.period)} h (${fmt(s.cycles, 2)} cycles${s.supported ? '' : ', unsupported'})`],
        ['p / q', `${fmt(s.p, 2)} / ${fmt(s.q, 2)}`], ['Phase', s.phase == null ? '–' : `${fmt(s.phase)} h`],
        ['Method', `${d.fit_method || '?'}${d.fft_component_test ? ' + component test' : d.significance_method ? ' + ' + d.significance_method : ''}`],
        ['Detrend', d.detrend || '–'], ['Normalise', d.trace_normalization || '–'],
        ['Periods', d.period_min_hours != null ? `${d.period_min_hours}–${d.period_max_hours} h` : '–'],
        ['Window', d.window_start_hours == null && d.window_end_hours == null ? 'whole recording' : `${d.window_start_hours ?? 'start'}–${d.window_end_hours ?? 'end'} h`],
        ['Made', analysis ? `${analysis.origin}${analysis.fresh ? '' : ' · STALE'} · ${(analysis.created_at || '').slice(0, 16)}` : '–'],
        ['Analysis', analysis ? analysis.analysis_id : '–'], ['Reproduce', analysis?.reproduce || analysis?.imported_from || '–'],
      ];
      return `<tr><td colspan="2" class="vhead" style="--c:${colourOf(key)}">${variant.short} · ${esc(seriesOf(key).metric || '')} · ${esc(variant.description)}</td></tr>` +
        items.map(([k, v]) => `<tr><th>${k}</th><td>${esc(String(v))}</td></tr>`).join('');
    });
    $('tr-provenance').innerHTML = `<table>${rows.join('')}</table>`;
  }

  let frameRequest = 0;
  async function showFrame(frame) {
    const cell = cellByKey(state.focus); if (!cell) return;
    state.frame = Number(frame);
    $('tr-frame').value = state.frame;
    const hours = hoursAt(cell, state.frame);
    $('tr-time').textContent = `frame ${state.frame}${hours == null ? '' : ' · ' + fmt(hours, 4) + ' h'}`;
    drawTrace($('tr-trace'), cell, { cursorHours: hours, axes: true });
    const tile = $('tr-grid').querySelector(`[data-key="${CSS.escape(cell.key)}"]`);
    if (tile) drawTrace(tile.querySelector('canvas'), cell, { cursorHours: hours });
    const request = ++frameRequest;
    const size = Number($('tr-crop').value);
    try {
      const result = await crop(cell, state.frame, size);
      if (request !== frameRequest) return;
      const canvas = $('tr-video'); const g = canvas.getContext('2d');
      $('tr-video-message').textContent = result.available ? (result.present ? '' : 'not segmented in this frame') : result.message;
      if (!result.available) { g.clearRect(0, 0, canvas.width, canvas.height); return; }
      const image = new Image();
      image.onload = () => { if (request === frameRequest) { g.imageSmoothingEnabled = false; g.drawImage(image, 0, 0, canvas.width, canvas.height); } };
      image.src = result.image;
      // Prefetch the next few frames so playback stays smooth.
      const frames = framesOf(cell); const at = frames.indexOf(state.frame);
      frames.slice(at + 1, at + 6).forEach((f) => crop(cell, f, size).catch(() => {}));
    } catch (error) {
      $('tr-video-message').textContent = error.message;
    }
  }

  function stop() { if (state.playing) { clearInterval(state.playing); state.playing = null; $('tr-play').textContent = 'Play'; } }
  function togglePlay() {
    if (state.playing) { stop(); return; }
    const cell = cellByKey(state.focus); if (!cell) return;
    $('tr-play').textContent = 'Pause';
    state.playing = setInterval(() => {
      const frames = framesOf(cell);
      const at = frames.indexOf(state.frame);
      showFrame(frames[(at + 1) % frames.length]);
    }, 1000 / Number($('tr-fps').value));
  }

  // --------------------------------------------------------------- handoff
  function openHandoff() {
    const counts = $('tr-counts').textContent;
    $('tr-handoff-summary').textContent = counts + '. Unreviewed cells are handed off as unreviewed, not excluded.';
    $('tr-handoff-variant').innerHTML = state.listing.variants.map((v) => `<option value="${v.key}">${v.short} · ${esc(v.description)}</option>`).join('');
    $('tr-handoff-variant').value = state.decisions.preferred_variant || state.series[0]?.variant || '';
    $('tr-handoff-result').textContent = '';
    $('tr-handoff-dialog').showModal();
  }
  async function saveHandoff(event) {
    event.preventDefault();
    const variant = $('tr-handoff-variant').value || null;
    try {
      await state.saving;
      if (variant !== state.decisions.preferred_variant) {
        state.decisions.preferred_variant = variant;
        await save([], { preferred_variant: variant });
      }
      const metric = state.series.find((s) => s.variant === variant)?.metric || null;
      const result = await api('/api/trace-review/handoff', { method: 'POST', body: JSON.stringify({ home: state.home, note: $('tr-handoff-note').value, variant, metric }) });
      $('tr-handoff-result').textContent = `Saved ${result.created_at}: ${result.counts.include} included, ${result.counts.exclude} excluded, ${result.counts.representative} representative → handoff.json and included_cells.csv in ${state.home}`;
    } catch (error) {
      $('tr-handoff-result').textContent = error.message;
    }
  }

  // --------------------------------------------------------------- wiring
  const esc = (text) => String(text ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function wire() {
    $('tr-sort').innerHTML = SORTS.map(([k, l]) => `<option value="${k}">${l}</option>`).join('');
    $('tr-rows').innerHTML = [1, 2, 3, 4, 5, 6, 8, 10, 12].map((n) => `<option>${n}</option>`).join('');
    $('tr-cols').innerHTML = ['auto', 1, 2, 3, 4, 5, 6, 8, 10, 12].map((n) => `<option>${n}</option>`).join('');
    $('tr-sort').value = state.sort; $('tr-rows').value = state.rows; $('tr-cols').value = state.cols;
    $('tr-images').checked = state.images; $('tr-fit').checked = state.fit;
    $('tr-sort-dir').textContent = state.descending ? '↑' : '↓';
    const rerender = (fn) => (event) => { fn(event); remember(); if (state.grid) render(); };
    $('tr-sort').addEventListener('change', rerender((e) => { state.sort = e.target.value; }));
    $('tr-sort-dir').addEventListener('click', rerender(() => { state.descending = !state.descending; $('tr-sort-dir').textContent = state.descending ? '↑' : '↓'; }));
    $('tr-filter').addEventListener('change', rerender((e) => { state.filter = e.target.value; }));
    $('tr-recording').addEventListener('change', rerender((e) => { state.recording = e.target.value; }));
    $('tr-rows').addEventListener('change', rerender((e) => { state.rows = Number(e.target.value); }));
    $('tr-cols').addEventListener('change', rerender((e) => { state.cols = e.target.value; }));
    $('tr-images').addEventListener('change', rerender((e) => { state.images = e.target.checked; }));
    $('tr-fit').addEventListener('change', rerender((e) => { state.fit = e.target.checked; if (state.focus) showFrame(state.frame); }));
    $('tr-overlay').addEventListener('change', (e) => {
      state.overlay = e.target.checked;
      markWindows();
      if (!state.overlay && (state.shown.length > 1 || state.metrics.length > 1)) {
        state.shown = state.shown.slice(0, 1); state.metrics = state.metrics.slice(0, 1); buildVariantChips(); loadGrid();
      } else if (state.grid) render();
    });
    $('tr-inspector-toggle').addEventListener('change', (e) => { document.querySelector('.tr-main').classList.toggle('no-inspector', !e.target.checked); if (state.grid) render(); });
    $('tr-include').addEventListener('click', () => setStatus('include'));
    $('tr-exclude').addEventListener('click', () => setStatus('exclude'));
    $('tr-clear').addEventListener('click', () => setStatus('unreviewed'));
    $('tr-star').addEventListener('click', toggleStar);
    $('tr-cell-include').addEventListener('click', () => state.focus && applyMarks([state.focus], { status: 'include' }));
    $('tr-cell-exclude').addEventListener('click', () => state.focus && applyMarks([state.focus], { status: 'exclude' }));
    $('tr-cell-clear').addEventListener('click', () => state.focus && applyMarks([state.focus], { status: 'unreviewed' }));
    $('tr-cell-star').addEventListener('click', () => state.focus && applyMarks([state.focus], (m) => ({ star: !m.star })));
    $('tr-accept').addEventListener('click', () => {
      const pool = state.selected.size ? [...state.selected] : state.order.map((c) => c.key);
      const keys = pool.filter((k) => state.decisions.suggestions?.[k] && (state.decisions.cells[k]?.status || 'unreviewed') === 'unreviewed');
      applyMarks(keys, (_, key) => ({ status: state.decisions.suggestions[key].status }));
    });
    let noteTimer = null;
    $('tr-note').addEventListener('input', () => {
      clearTimeout(noteTimer);
      const key = state.focus; const note = $('tr-note').value;
      noteTimer = setTimeout(() => key && applyMarks([key], { note }), 600);
    });
    $('tr-handoff').addEventListener('click', openHandoff);
    $('tr-handoff-save').addEventListener('click', saveHandoff);
    $('tr-play').addEventListener('click', togglePlay);
    $('tr-frame').addEventListener('input', (e) => { stop(); showFrame(e.target.value); });
    $('tr-fps').addEventListener('change', () => { if (state.playing) { stop(); togglePlay(); } });
    $('tr-crop').addEventListener('change', () => { const cell = cellByKey(state.focus); if (cell) { renderThumbs(cell); showFrame(state.frame); } });
    $('tr-trace').addEventListener('click', (e) => {
      const cell = cellByKey(state.focus); if (!cell) return;
      const box = e.target.getBoundingClientRect();
      const hours = range[0] + ((e.clientX - box.left - 4) / (box.width - 8)) * (range[1] - range[0]);
      const trace = cell.traces[primary()] || Object.values(cell.traces)[0];
      let best = 0; trace.hours.forEach((h, i) => { if (h != null && Math.abs(h - hours) < Math.abs((trace.hours[best] ?? Infinity) - hours)) best = i; });
      stop(); showFrame(trace.frame[best]);
    });
    $('tr-grid').addEventListener('click', (e) => { const tile = e.target.closest('.tr-tile'); if (tile) select(tile.dataset.key, e); });
    $('tr-open').addEventListener('submit', (e) => { e.preventDefault(); openHome($('tr-open-path').value.trim()); });
    document.addEventListener('keydown', (e) => {
      if (e.target.matches('input[type=text], textarea, select') || $('tr-handoff-dialog').open) return;
      const key = e.key.toLowerCase();
      if (key === 'i') setStatus('include');
      else if (key === 'x') setStatus('exclude');
      else if (key === 'u') setStatus('unreviewed');
      else if (key === 's') toggleStar();
      else if (key === 'p') togglePlay();
      else if (key === 'escape') { state.selected.clear(); render(); }
      else if (key === 'a' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); state.order.forEach((c) => state.selected.add(c.key)); render(); }
      else if (key === ' ' && state.selected.size) { e.preventDefault(); inspect([...state.selected][0]); }
      else if (['arrowright', 'arrowleft', 'arrowdown', 'arrowup'].includes(key) && state.order.length) {
        e.preventDefault();
        const cols = getComputedStyle($('tr-grid')).gridTemplateColumns.split(' ').length;
        const order = state.order.map((c) => c.key);
        const at = Math.max(0, order.indexOf(state.focus));
        const step = { arrowright: 1, arrowleft: -1, arrowdown: cols, arrowup: -cols }[key];
        const next = order[Math.min(order.length - 1, Math.max(0, at + step))];
        select(next, e);
        $('tr-grid').querySelector(`[data-key="${CSS.escape(next)}"]`)?.scrollIntoView({ block: 'nearest' });
      }
    });
    let resizeTimer = null;
    window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => state.grid && render(), 150); });
  }

  wire();
  if (state.home) openHome(state.home); else $('tr-open').hidden = false;
})();
