/* Copyright (c) Meta Platforms, Inc. and affiliates. */

/* ---------------------------------------------------------------------------
 * Page builder.
 *
 * Static prose (title, abstract, method text, BibTeX) lives in index.html and is
 * edited by hand.  Everything data-heavy — the comparison sliders and the result
 * galleries — is built here from `window.PAGE_DATA` (see data.js, which
 * collect_assets.py regenerates).  Adding a dataset or a baseline therefore
 * never requires touching index.html.
 * ------------------------------------------------------------------------- */

(function () {
  'use strict';

  var DATA = window.PAGE_DATA || null;

  /* Media is not in the repo: the published page loads it from Google Cloud
   * Storage (`assetBase` in data.js, uploaded by upload_assets.py). Every media
   * path in data.js is a relative `assets/...`, so they are all rewritten here,
   * once, before anything reads them. A local dev server (localhost /
   * 127.0.0.1) or `?local` in the URL keeps the local assets/ folder instead,
   * so freshly generated media shows before it is uploaded. */
  (function useAssetBase() {
    var base = DATA && DATA.assetBase;
    var local = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname) ||
                /(^|[?&])local(=|&|$)/.test(location.search.slice(1));
    if (!base || local) return;
    base = base.replace(/\/+$/, '') + '/';
    (function walk(o) {
      Object.keys(o).forEach(function (k) {
        var v = o[k];
        if (typeof v === 'string' && v.indexOf('assets/') === 0) o[k] = base + v;
        else if (v && typeof v === 'object') walk(v);
      });
    })(DATA);
  })();

  function el(tag, className, text) {
    var n = document.createElement(tag);
    if (className) n.className = className;
    if (text != null) n.textContent = text;
    return n;
  }

  function appendBlurb(host, blurb, extraClass) {
    if (!blurb) return;
    var b = el('p', 'section-blurb' + (extraClass ? ' ' + extraClass : ''));
    b.innerHTML = blurb;          // blurbs are authored, not user input
    host.appendChild(b);
  }

  /* "this half of the page has no manifest yet" box. */
  function setupNote(host, html) {
    var box = el('div', 'setup-note center');
    box.innerHTML = html;
    host.appendChild(box);
  }

  /* A row of mutually exclusive chips, from `{id, display|label, available}`
   * items.  `available: false` is a configuration the page declares but has no
   * orbit for — greyed out, so the row is the full space of options rather than
   * only the finished ones.  `onPick` fires only on a real change: re-selecting
   * the active chip would reload every viewer for nothing.
   *
   * `set` is idempotent — memoized on the item-id key — so `render()` can call
   * it every pass and rows whose contents depend on other state (the "Show" row
   * follows the view count) rebuild exactly when that state changes. */
  function chipRow(host, label, getActive, onPick) {
    var row = el('div', 'chip-row');
    var group = el('div', 'chip-group');
    row.appendChild(el('span', 'chip-label', label));
    row.appendChild(group);
    host.appendChild(row);
    var made = [];
    var madeKey = null;

    return {
      set: function (items) {
        items = items || [];
        var key = items.map(function (i) { return i.id; }).join('|');
        if (key === madeKey) return;
        madeKey = key;
        group.innerHTML = '';
        made = items.map(function (item) {
          var c = el('button', 'chip', item.display || item.label);
          c.type = 'button';
          if (item.available === false) {
            c.disabled = true;
            c.title = 'no orbit render yet';
          } else {
            c.addEventListener('click', function () {
              if (item.id === getActive()) return;
              onPick(item.id);
            });
          }
          group.appendChild(c);
          return { item: item, el: c };
        });
        // A lone chip is not a selector, but it still names what is on screen.
        row.hidden = !made.length;
      },
      sync: function () {
        made.forEach(function (m) {
          m.el.classList.toggle('active', m.item.id === getActive());
        });
      }
    };
  }

  /* Gallery videos are fetched on first approach and played only while on
   * screen. <img> gets this natively via loading="lazy"; <video> has no
   * equivalent, so the src is parked in a data attribute until the observer
   * says the tile is near the viewport. */
  /* One play/pause per comparison HOST (the section-top button), keyed by
   * host element id -- the interactive comparisons and the limitations
   * section each pause independently. The observer stays the play
   * authority: while paused it still loads a video's src on approach, it
   * just doesn't start it. */
  var pausedHosts = {};

  /* The three comparison panels (ours w/o TTR, ours w/ TTR, baseline) are
   * same-length turntable loops, but switching one chip (e.g. the baseline)
   * only rebuilds the panel that changed -- the untouched siblings keep
   * playing from wherever they already were. A freshly (re)built video
   * starts at t=0 instead, so it visibly spins out of phase with them. Every
   * comparison video is instead phased off one shared clock: whenever a new
   * one loads, it seeks to how far into its own duration that clock has
   * gotten, landing back in step with any sibling built off the same clock. */
  var orbitEpoch = Date.now();
  function syncOrbitPhase(v) {
    function seek() {
      var d = v.duration;
      if (isFinite(d) && d > 0) v.currentTime = ((Date.now() - orbitEpoch) / 1000) % d;
    }
    if (v.readyState >= 1) seek();
    else v.addEventListener('loadedmetadata', seek);
  }

  var lazyObserver = ('IntersectionObserver' in window)
    ? new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          var v = e.target;
          if (!e.isIntersecting) { v.pause(); return; }
          if (!v.src && v.dataset.src) { v.src = v.dataset.src; }
          var pHost = v.closest('#comparisons, #limitations');
          if (pHost && pausedHosts[pHost.id]) return;
          var p = v.play();
          if (p && p.catch) p.catch(function () {});
        });
      }, { rootMargin: '200px' })
    : null;

  function makeVideo(src, className) {
    var v = document.createElement('video');
    v.className = className || '';
    v.muted = true;
    v.defaultMuted = true;
    v.loop = true;
    v.playsInline = true;
    v.setAttribute('muted', '');
    v.setAttribute('playsinline', '');
    v.setAttribute('webkit-playsinline', '');
    v.preload = 'metadata';
    v.controls = false;
    if (lazyObserver) { v.dataset.src = src; lazyObserver.observe(v); }
    else { v.src = src; v.autoplay = true; }
    return v;
  }

  var IMAGE_RE = /\.(png|jpe?g|gif|webp|avif|bmp)(\?.*)?$/i;

  function makeMediaNode(src, className) {
    if (!IMAGE_RE.test(src)) return makeVideo(src, className);
    var img = document.createElement('img');
    img.src = src;
    img.loading = 'lazy';
    img.alt = '';
    if (className) img.className = className;
    return img;
  }

  /* Full-size view of one plate, for the grids whose tiles are small enough
   * that the hover magnifier only goes so far. One overlay for the page,
   * rebuilt per open; a click anywhere on it or Escape closes it. The media
   * is built here rather than by makeMediaNode: the lazy observer parks a
   * video's src until it scrolls into view, and this node is already in view. */
  var lightbox = null;

  function closeLightbox() {
    if (lightbox) { lightbox.hidden = true; lightbox.innerHTML = ''; }
  }

  function openLightbox(src, caption) {
    if (!lightbox) {
      lightbox = el('div', 'lightbox');
      lightbox.addEventListener('click', closeLightbox);
      document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') closeLightbox();
      });
      document.body.appendChild(lightbox);
    }
    lightbox.innerHTML = '';
    var fig = el('figure', 'lightbox-figure');
    if (IMAGE_RE.test(src)) {
      var img = document.createElement('img');
      img.src = src;
      img.alt = caption || '';
      fig.appendChild(img);
    } else {
      var v = document.createElement('video');
      v.src = src;
      v.muted = true;
      v.defaultMuted = true;
      v.loop = true;
      v.autoplay = true;
      v.playsInline = true;
      v.setAttribute('muted', '');
      v.setAttribute('playsinline', '');
      fig.appendChild(v);
    }
    if (caption) fig.appendChild(el('figcaption', 'lightbox-caption', caption));
    lightbox.appendChild(fig);
    lightbox.hidden = false;
  }

  /* `count` independent orbits shown side by side (ours w/o TTR, ours w/
   * TTR, a baseline), each labelled with the same badge style the teaser
   * grid uses. Hovering any panel magnifies the same source point across
   * all three -- the same interactive zoom the teaser wires per column,
   * scoped to the whole row here since there is no other axis to split on.
   * `set([{src, label}, ...])` skips rebuilding a panel whose (src, label)
   * did not change -- switching the baseline chip should not re-fetch the
   * two Ours panels beside it. */
  function buildPanelRow(container, count) {
    container.classList.add('pair-row');
    var panels = [], sig = [], medias = [];
    for (var i = 0; i < count; i++) {
      panels.push(el('div', 'pair-panel'));
      container.appendChild(panels[i]);
      sig.push(null);
      medias.push(null);
      wireZoom(i);
    }

    function wireZoom(i) {
      panels[i].addEventListener('pointermove', function (e) {
        var b = panels[i].getBoundingClientRect();
        var ox = ((e.clientX - b.left) / b.width * 100).toFixed(1) + '%';
        var oy = ((e.clientY - b.top) / b.height * 100).toFixed(1) + '%';
        medias.forEach(function (m) {
          if (!m) return;
          m.style.transformOrigin = ox + ' ' + oy;
          m.style.transform = 'scale(2)';
        });
      });
      panels[i].addEventListener('pointerleave', function () {
        medias.forEach(function (m) { if (m) m.style.transform = ''; });
      });
    }

    function fill(i, src, label) {
      var key = src + '|' + label;
      if (sig[i] === key) return;
      sig[i] = key;
      var panel = panels[i];
      panel.innerHTML = '';
      medias[i] = null;
      panel.classList.toggle('pair-panel-missing', !src);
      panel.classList.remove('enlargeable');
      panel.onclick = null;
      if (!src) {
        panel.appendChild(el('span', 'pair-missing-text', 'no render yet'));
        return;
      }
      var media = makeMediaNode(src);
      if (media.tagName === 'VIDEO') syncOrbitPhase(media);
      medias[i] = media;
      panel.appendChild(media);
      if (label) panel.appendChild(el('span', 'vc-label teaser-badge', label));
      // Same click-to-open-full-size as the teaser; drop the zoom first, same
      // reason it does -- the lightbox covers the panel, so pointerleave is
      // not guaranteed to fire and clear it.
      panel.classList.add('enlargeable');
      panel.onclick = function () {
        medias.forEach(function (m) { if (m) m.style.transform = ''; });
        openLightbox(src, label);
      };
    }

    return {
      set: function (entries) {
        entries.forEach(function (e, i) { fill(i, e.src, e.label); });
      }
    };
  }

  /* A fourth tile beside the orbit panels: the scene's input photos, laid out
   * the same small grid `collect_qualitative_figures` draws as its own "Input
   * views" row on qualitative.html -- here compacted into one panel-sized
   * tile since there is no wide row to spread it across. Each thumbnail opens
   * full size on its own click; unlike the orbit panels there is nothing to
   * hover-zoom (a handful of stills, not one continuous render). */
  function buildInputsPanel(container) {
    var panel = el('div', 'pair-panel pair-panel-inputs');
    container.appendChild(panel);
    var sig = null;

    return {
      set: function (pool, label) {
        var key = (pool || []).join('|');
        if (key === sig) return;
        sig = key;
        panel.innerHTML = '';
        panel.classList.toggle('pair-panel-missing', !pool || !pool.length);
        if (!pool || !pool.length) {
          panel.appendChild(el('span', 'pair-missing-text', 'no input views'));
          return;
        }
        var grid = el('div', 'pair-inputs-grid');
        grid.classList.toggle('single-input', pool.length === 1);
        pool.forEach(function (src, i) {
          var thumb = el('div', 'pair-input-thumb enlargeable');
          var img = document.createElement('img');
          img.src = src;
          img.loading = 'lazy';
          img.alt = '';
          thumb.appendChild(img);
          thumb.addEventListener('click', function () {
            openLightbox(src, label + ' — view ' + (i + 1));
          });
          grid.appendChild(thumb);
        });
        panel.appendChild(grid);
        if (label) panel.appendChild(el('span', 'vc-label teaser-badge', label));
      }
    };
  }

  /* --- comparison block --------------------------------------------------- */

  function buildComparison(block) {
    var wrap = el('div', 'comparison-block');
    if (block.title) {
      var h = el('h3', 'sub-title center', block.title);
      wrap.appendChild(h);
    }
    appendBlurb(wrap, block.blurb, 'small');

    // "Ours" is a family, not a setting: each reference pins a geometry source,
    // an input view count and whether TTR ran. Both TTR settings show at once
    // (there is no "Show" selector any more), so Views picks which pair of
    // them is on screen and vs picks the baseline beside them. Which ones a
    // dataset offers is declared in collect_assets.py's DATASETS, and each
    // was rendered for its own set of scenes.
    var scenes = block.scenes || [];
    var references = block.references || [];
    if (!scenes.length || !block.defaultReference) {
      wrap.appendChild(el('p', 'section-blurb small', 'No assets found for this block.'));
      return wrap;
    }

    var refById = {}, baseById = {};
    references.forEach(function (r) { refById[r.id] = r; });
    var baselines = block.baselines || [];
    baselines.forEach(function (b) { baseById[b.id] = b; });

    // The view axis is derived, not shipped: references are the declared
    // (count x TTR) cross-product, so the counts they span ARE the axis. A
    // monocular dataset spans one count and gets no Views row.
    var viewCounts = [];
    references.forEach(function (r) {
      if (viewCounts.indexOf(r.views) < 0) viewCounts.push(r.views);
    });
    var hasViewAxis = viewCounts.length > 1;

    var state = {
      sceneId: null,   // resolved to a real scene the first time render() runs
      views: (refById[block.defaultReference] || references[0]).views,
      baseline: block.defaultBaseline
    };

    /* The baseline selector is scoped to the view control: a baseline given a
     * different number of input views is not a like-for-like comparison. */
    function atViewOf(list, n) {
      return list.filter(function (e) { return e.views === n; });
    }
    // The scene is tracked by ID, not position: switching Views re-scopes
    // `shownScenes()`, and an index would silently land on whatever scene
    // happens to fall at the old slot rather than the one the reader was
    // looking at. -1 when the id is gone from the new list (falls back to
    // scene 0 in render()).
    function indexOfScene(list, id) {
      for (var i = 0; i < list.length; i++) {
        if (list[i].id === id) return i;
      }
      return -1;
    }
    function atView(list) { return atViewOf(list, state.views); }
    function firstLive(list) {
      var live = list.filter(function (e) { return e.available; });
      return (live[0] || list[0] || {}).id || null;
    }
    // Both TTR settings at the current view count -- the two collect_assets.py
    // always declares via _ttr_variants, told apart by the one thing that
    // never changes shape across datasets: their `display` text.
    function ttrPair() {
      var atV = atView(references);
      return {
        wo: atV.filter(function (r) { return r.display === 'Ours w/o TTR'; })[0],
        w: atV.filter(function (r) { return r.display === 'Ours w/ TTR'; })[0]
      };
    }
    function shownScenes() {
      var pair = ttrPair();
      return scenes.filter(function (s) {
        return (pair.wo && s.media[pair.wo.id]) || (pair.w && s.media[pair.w.id]);
      });
    }

    var controls = el('div', 'controls');
    wrap.appendChild(controls);

    var viewRow = hasViewAxis ? chipRow(controls, 'Views',
      function () { return state.views; },
      function (n) {
        state.views = n;
        state.baseline = firstLive(atView(baselines));
        // state.sceneId is left as-is: every view count renders the same
        // scene set, so the reader stays on the scene they were looking at.
        render();                      // render() re-scopes the row below
      }) : null;
    // One baseline selector for the whole dataset; every scene switches together
    var baseRow = chipRow(controls, 'Baseline',
      function () { return state.baseline; },
      function (bid) { state.baseline = bid; render(); });
    if (viewRow) {
      viewRow.set(viewCounts.map(function (n) {
        return { id: n, label: String(n),
                 available: atViewOf(references, n).some(function (r) { return r.available; }) };
      }));
    }

    // One scene on screen at a time: three panels (ours w/o TTR, ours w/ TTR,
    // baseline) built once and re-pointed as the reader pages, so switching
    // scenes or the baseline chip costs a source swap, not a fresh widget.
    var cell = el('div', 'scene-cell');
    var stage = el('div', '');
    var caption = el('p', 'vc-caption scene-caption');
    cell.appendChild(stage);
    cell.appendChild(caption);
    wrap.appendChild(cell);
    var panelRow = buildPanelRow(stage, 3);
    var inputsPanel = buildInputsPanel(stage);

    var emptyNote = el('p', 'section-blurb small',
      'No orbit rendered for this configuration yet.');
    wrap.appendChild(emptyNote);

    // One dot per scene; the current one lights up.
    var pager = el('div', 'scene-pager');
    var arrows = { prev: el('button', 'pager-arrow', '‹'),
                   next: el('button', 'pager-arrow', '›') };
    [['prev', -1, 'Previous scene'], ['next', 1, 'Next scene']].forEach(function (a) {
      var btn = arrows[a[0]];
      btn.type = 'button';
      btn.setAttribute('aria-label', a[2]);
      btn.addEventListener('click', function () {
        goToScene(indexOfScene(shownScenes(), state.sceneId) + a[1]);
      });
    });
    var dotRow = el('div', 'pager-dots');
    var dots = [];
    var dotsKey = null;
    pager.appendChild(arrows.prev);
    pager.appendChild(dotRow);
    pager.appendChild(arrows.next);
    wrap.appendChild(pager);

    function goToScene(i) {
      var cur = shownScenes();
      i = Math.min(Math.max(i, 0), cur.length - 1);
      var id = (cur[i] || {}).id || null;
      if (id === state.sceneId) return;
      state.sceneId = id;
      render();
    }

    /* The scene list follows the chosen view count, so the dots are rebuilt
     * when it changes -- but only then, since each carries its scene's name. */
    function syncDots(cur) {
      var key = cur.map(function (s) { return s.id; }).join('|');
      if (key === dotsKey) return;
      dotsKey = key;
      dotRow.innerHTML = '';
      dots = cur.map(function (s, si) {
        var d = el('button', 'pager-dot');
        d.type = 'button';
        d.title = s.label || s.id;
        d.setAttribute('aria-label', d.title);
        d.addEventListener('click', function () { goToScene(si); });
        dotRow.appendChild(d);
        return d;
      });
    }

    function render() {
      var cur = shownScenes();
      // Re-resolve the scene by id rather than trusting an old index: a
      // Views switch re-scopes `cur`, and the id this block was showing
      // may now sit at a different position, or (if it were ever dropped
      // from a view count) not exist at all.
      var idx = indexOfScene(cur, state.sceneId);
      if (idx < 0) idx = 0;
      state.sceneId = (cur[idx] || {}).id || null;
      // set() is memoized on content, so re-scoping here is free unless the
      // view count actually changed
      baseRow.set(atView(baselines));
      syncDots(cur);

      var pair = ttrPair();
      var baseLabel = (baseById[state.baseline] || {}).label || 'baseline';
      var scene = cur[idx];
      if (scene) {
        var has = state.baseline && scene.media[state.baseline];
        panelRow.set([
          { src: pair.wo && scene.media[pair.wo.id], label: pair.wo && pair.wo.label },
          { src: pair.w && scene.media[pair.w.id], label: pair.w && pair.w.label },
          { src: has ? scene.media[state.baseline] : null, label: has ? baseLabel : '' }
        ]);
        // Static multi-view datasets (CO3D, GSO) pool one thumbnail per input
        // view, in view order -- so the tile should track the Views chip
        // rather than always showing every thumbnail the pool has. Dynamic
        // datasets (DAVIS, ActionBench) have no view axis: their pool is
        // temporal frames, not input views, and stays shown in full.
        inputsPanel.set(hasViewAxis ? (scene.pool || []).slice(0, state.views) : scene.pool,
          'Input views');
        caption.textContent = (scene.label || scene.id) +
          (has ? '' : ' — no ' + baseLabel + ' render');
      }

      if (viewRow) viewRow.sync();
      baseRow.sync();
      dots.forEach(function (d, si) {
        d.classList.toggle('active', si === idx);
      });
      arrows.prev.disabled = idx <= 0;
      arrows.next.disabled = idx >= cur.length - 1;
      // nothing rendered for the chosen configuration: say so instead of leaving
      // a stuck empty stage
      cell.hidden = !cur.length;
      emptyNote.hidden = cur.length > 0;
      pager.hidden = cur.length < 2;
    }

    render();
    return wrap;
  }

  /* --- gallery block ------------------------------------------------------ */

  function buildGallery(block) {
    var wrap = el('div', 'gallery-block');
    if (block.title) wrap.appendChild(el('h3', 'sub-title center', block.title));
    appendBlurb(wrap, block.blurb, 'small');
    var grid = el('div', 'grid' + (block.columns === 2 ? ' cols-2' : block.columns === 4 ? ' cols-4' : ''));
    (block.items || []).forEach(function (item) {
      var fig = el('figure', 'tile');
      fig.appendChild(makeMediaNode(item.src));
      if (item.caption) fig.appendChild(el('figcaption', null, item.caption));
      grid.appendChild(fig);
    });
    wrap.appendChild(grid);
    return wrap;
  }

  /* --- quantitative tables ------------------------------------------------- */

  /* One table, rendered from build_tables.py's output. The shape mirrors the
   * paper's baked tabular: an optional band row over the title row, rowspan'd
   * group column, and first/second/third cell shading. */
  function buildTable(t) {
    var fig = el('figure', 'table-block');
    if (t.title) fig.appendChild(el('figcaption', 'table-title', t.title));

    // Raw column indices where a new DATASET starts, so a vertical rule can
    // separate e.g. GSO-30 from CO3D all the way down the table -- the
    // outermost band level carries that split (`superbands` when present,
    // else `bands` on the tables with only one banded header row). Index 0
    // is skipped: nothing precedes it to separate from.
    var datasetBands = t.superbands || t.bands || [];
    var boundaryCols = {};
    (function () {
      var col = 0;
      datasetBands.forEach(function (b) {
        if (col > 0 && b.rule) boundaryCols[col] = true;
        col += b.span;
      });
    })();

    // Row-label ("stub") columns: everything left of the first dataset (#V +
    // Method, Variant, or the Attn/RG/TTR switches). On a phone they stay
    // pinned while the numbers scroll under them (sticky, see app.css).
    var firstBoundary = Math.min.apply(null,
      Object.keys(boundaryCols).map(Number).concat([t.head.length]));
    if (firstBoundary >= t.head.length) firstBoundary = 1;
    function markStub(cell, col, span) {
      if (col >= firstBoundary) return;
      cell.classList.add('stub');
      cell.dataset.col = col;
      if (col + (span || 1) >= firstBoundary) cell.classList.add('stub-last');
    }

    var table = el('table', 'results');
    var thead = el('thead');
    /* Up to two banded header rows: the superband row (dataset names on the merged
     * benchmark tables) above the band row (Train/Test), mirroring the paper's
     * three-row header. Either may be absent. */
    [t.superbands, t.bands].forEach(function (bands) {
      if (!bands) return;
      var bandRow = el('tr', 'band-row');
      var col = 0;
      bands.forEach(function (b) {
        var th = el('th', b.rule ? 'band' : null);
        if (boundaryCols[col]) th.classList.add('col-group-start');
        markStub(th, col, b.span);
        th.colSpan = b.span;
        th.innerHTML = b.label;
        bandRow.appendChild(th);
        col += b.span;
      });
      thead.appendChild(bandRow);
    });
    var headRow = el('tr');
    t.head.forEach(function (h, i) {
      var th = el('th');
      if (boundaryCols[i]) th.classList.add('col-group-start');
      markStub(th, i);
      th.innerHTML = h.label;
      th.style.textAlign = h.align === 'l' ? 'left' : h.align === 'r' ? 'right' : 'center';
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    // Body rows walk RAW columns, not `r.cells` array positions: a rowspan
    // cell (e.g. the "#V" stub) is omitted from `r.cells` on the rows it
    // covers, which the browser re-aligns onto the grid automatically but
    // would otherwise throw off a plain per-cell array index by one column
    // for exactly those rows.
    var totalCols = t.head.length;
    var activeSpans = {};
    var tbody = el('tbody');
    t.rows.forEach(function (r) {
      var tr = el('tr', r.newGroup ? 'group-start' : null);
      var ci = 0;
      for (var col = 0; col < totalCols; col++) {
        if (activeSpans[col] > 0) { activeSpans[col]--; continue; }
        var c = r.cells[ci++];
        var cls = [c.rank ? 'rank' + c.rank : '', c.bold ? 'bold' : '',
                  c.muted ? 'muted' : '', boundaryCols[col] ? 'col-group-start' : '']
          .join(' ').trim();
        var td = el('td', cls || null);
        markStub(td, col);
        if (c.rowspan) { td.rowSpan = c.rowspan; activeSpans[col] = c.rowspan - 1; }
        td.style.textAlign = c.align === 'l' ? 'left' : c.align === 'r' ? 'right' : 'center';
        td.innerHTML = c.html;
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    // Sticky offsets need layout, so they are measured once the table is on
    // the page (and again whenever it resizes): each stub cell sticks at its
    // own column's left edge, so several stub columns stay side by side.
    function pinStub() {
      var lefts = Array.prototype.map.call(headRow.children, function (th) { return th.offsetLeft; });
      Array.prototype.forEach.call(table.querySelectorAll('.stub'), function (cell) {
        cell.style.left = (lefts[+cell.dataset.col] - lefts[0]) + 'px';
      });
    }
    if (window.ResizeObserver) new ResizeObserver(pinStub).observe(table);

    var scroll = el('div', 'table-scroll');
    scroll.appendChild(table);
    fig.appendChild(scroll);
    return fig;
  }

  function buildTableSection(block) {
    var wrap = el('div', 'table-section');
    var host = wrap;

    if (block.collapsed) {
      // <details> gives the disclosure behaviour, the keyboard handling and the
      // in-page-find expansion for free — no JS, no ARIA to get wrong
      var det = el('details', 'table-details');
      det.appendChild(el('summary', 'table-summary', block.title));
      wrap.appendChild(det);
      host = det;
    } else {
      wrap.appendChild(el('h3', 'sub-title center', block.title));
    }

    appendBlurb(host, block.blurb);
    var grid = el('div', 'table-stack');
    (block.tables || []).forEach(function (t) { grid.appendChild(buildTable(t)); });
    host.appendChild(grid);
    return wrap;
  }

  /* --- mount -------------------------------------------------------------- */

  /* Icons for the header link pills, inlined as SVG so the page still loads
   * no icon font and makes no CDN request. Same icons the VolSurfs project
   * page uses: Font Awesome Free 5.15.1 (CC BY 4.0) for PDF (fas file-pdf),
   * Code (fab github) and Data (far images), Academicons (SIL OFL) for arXiv,
   * plus the Hugging Face logo for the demo. The single-colour ones fill with
   * currentColor, so they follow the pill's text colour, greyed-out included;
   * the multicolour HF logo is greyed out in css/app.css instead. */
  var LINK_ICONS = {
    pdf: '<svg viewBox="0 0 384 512" aria-hidden="true" fill="currentColor"><path d="M181.9 256.1c-5-16-4.9-46.9-2-46.9 8.4 0 7.6 36.9 2 46.9zm-1.7 47.2c-7.7 20.2-17.3 43.3-28.4 62.7 18.3-7 39-17.2 62.9-21.9-12.7-9.6-24.9-23.4-34.5-40.8zM86.1 428.1c0 .8 13.2-5.4 34.9-40.2-6.7 6.3-29.1 24.5-34.9 40.2zM248 160h136v328c0 13.3-10.7 24-24 24H24c-13.3 0-24-10.7-24-24V24C0 10.7 10.7 0 24 0h200v136c0 13.2 10.8 24 24 24zm-8 171.8c-20-12.2-33.3-29-42.7-53.8 4.5-18.5 11.6-46.6 6.2-64.2-4.7-29.4-42.4-26.5-47.8-6.8-5 18.3-.4 44.1 8.1 77-11.6 27.6-28.7 64.6-40.8 85.8-.1 0-.1.1-.2.1-27.1 13.9-73.6 44.5-54.5 68 5.6 6.9 16 10 21.5 10 17.9 0 35.7-18 61.1-61.8 25.8-8.5 54.1-19.1 79-23.2 21.7 11.8 47.1 19.5 64 19.5 29.2 0 31.2-32 19.7-43.4-13.9-13.6-54.3-9.7-73.6-7.2zM377 105L279 7c-4.5-4.5-10.6-7-17-7h-6v128h128v-6.1c0-6.3-2.5-12.4-7-16.9zm-74.1 255.3c4.1-2.7-2.5-11.9-42.8-9 37.1 15.8 42.8 9 42.8 9z"/></svg>',
    arxiv: '<svg viewBox="0 0 448 512" aria-hidden="true" fill="currentColor"><path transform="translate(0 448) scale(1 -1)" d="M62.2578 439.994v0.000976563c8.7373 -0.208008 17.0049 -10.248 17.0049 -10.248l139.339 -133.672l101.071 91.6465l0.19043 0.172852l0.203125 0.166016c4.90625 4.81934 11.3877 7.70312 18.252 8.12207l0.00585938 0.0185547 c1.05957 -0.00195312 2.11816 -0.0712891 3.16699 -0.211914c10.166 -2.4209 18.7891 -9.11816 23.6465 -18.3701c4.01758 -9.88574 0.291016 -17.6826 -7.50195 -27.8057l-0.102539 -0.139648l-0.115234 -0.132812l-87.1963 -102.987l25.2568 -24.2285 c17.0928 -15.6211 17.002 -42.5723 -0.197266 -58.0762l-23.0615 -21.4434l128.783 -155.019c6.05273 -6.27637 8.49609 -15.2041 6.48145 -23.6885c-2.51465 -9.04297 -9.44238 -16.1914 -18.4004 -18.9902c-2.37793 -0.728516 -4.84863 -1.10254 -7.33496 -1.1084 c-6.88184 0.109375 -13.4678 2.83105 -18.4189 7.61035l-146.593 139.609l-95.1221 -88.4375c-3.72852 -4.63086 -9.30664 -7.38672 -15.249 -7.53516c-9.27832 -0.0908203 -17.6777 5.47754 -21.209 14.0586c-3.46777 8.33105 0.375 19.9385 6.58398 26.5527 l79.9766 98.2344l-28.6631 27.3008c-19.292 19.2861 -18.4004 47.1162 2.33984 67.8516l24.6191 23.0391s-111.243 133.583 -122.178 149.671c-7.46582 10.6934 -9.91016 16.4219 -6.50586 24.5918c3.57715 8.33398 11.8613 13.6562 20.9277 13.4473zM338.319 388.78 v-0.0214844c-5.18945 -0.473633 -10.0342 -2.79199 -13.6592 -6.53516l-100.681 -91.3047l40.8809 -39.2197l86.8916 102.64c8.05957 10.4746 8.75195 14.9609 6.50488 20.4824c-3.83691 6.77344 -10.2119 11.7451 -17.7188 13.8105 c-0.735352 0.0976562 -1.47754 0.145508 -2.21875 0.148438zM175.076 246.873l-24.623 -23.041c-16.8564 -16.8564 -19.4756 -39.8701 -2.15918 -57.1865l220.151 -209.691c3.5752 -3.45898 8.32031 -5.44336 13.293 -5.56152 c1.75293 0.00292969 3.49512 0.266602 5.1709 0.78418c6.58887 2.0752 11.6631 7.36914 13.4629 14.0381c1.83984 6.0957 -0.860352 11.5049 -5.18066 16.9141z"/></svg>',
    github: '<svg viewBox="0 0 496 512" aria-hidden="true" fill="currentColor"><path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg>',
    data: '<svg viewBox="0 0 576 512" aria-hidden="true" fill="currentColor"><path d="M480 416v16c0 26.51-21.49 48-48 48H48c-26.51 0-48-21.49-48-48V176c0-26.51 21.49-48 48-48h16v48H54a6 6 0 0 0-6 6v244a6 6 0 0 0 6 6h372a6 6 0 0 0 6-6v-10h48zm42-336H150a6 6 0 0 0-6 6v244a6 6 0 0 0 6 6h372a6 6 0 0 0 6-6V86a6 6 0 0 0-6-6zm6-48c26.51 0 48 21.49 48 48v256c0 26.51-21.49 48-48 48H144c-26.51 0-48-21.49-48-48V80c0-26.51 21.49-48 48-48h384zM264 144c0 22.091-17.909 40-40 40s-40-17.909-40-40 17.909-40 40-40 40 17.909 40 40zm-72 96l39.515-39.515c4.686-4.686 12.284-4.686 16.971 0L288 240l103.515-103.515c4.686-4.686 12.284-4.686 16.971 0L480 208v80H192v-48z"/></svg>',
    huggingface: '<svg class="multicolor" viewBox="0 0 95 88" aria-hidden="true" fill="none"><path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z"/><path fill="#FF9D0B" d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z"/><path fill="#3A3B45" d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z"/><path fill="#FF323D" d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z"/><path fill="#3A3B45" fill-rule="evenodd" d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z" clip-rule="evenodd"/><path fill="#FF9D0B" d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z"/><path fill="#FFD21E" d="M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z"/><path fill="#FF9D0B" d="M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z"/><path fill="#FFD21E" d="M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z"/></svg>'
  };

  function mountAll() {
    if (!DATA) {
      var host = document.getElementById('comparisons');
      if (host) {
        setupNote(host, '<strong>No <code>data.js</code> found.</strong> Generate the manifest ' +
          'and copy the render assets with:<br><code>python paper/webpage/collect_assets.py</code>');
      }
      return;
    }

    var linkHost = document.getElementById('links');
    var links = DATA.links || [];
    if (linkHost) {
      if (!links.length) {
        linkHost.hidden = true;
      } else {
        links.forEach(function (l) {
          var li = document.createElement('li');
          var a = document.createElement('a');
          if (l.icon && LINK_ICONS[l.icon]) a.innerHTML = LINK_ICONS[l.icon];
          a.appendChild(document.createTextNode(l.label));
          a.href = l.href || '#';
          if (l.disabled || !l.href) a.className = 'disabled';
          if (l.href && /^https?:/.test(l.href)) { a.target = '_blank'; a.rel = 'noopener'; }
          li.appendChild(a);
          linkHost.appendChild(li);
        });
      }
    }

    // Shared by the interactive comparisons and the limitations section --
    // both are a stack of `buildComparison` blocks in one host, with one
    // shared play/pause toggle for every orbit in that host at once.
    // Pausing acts directly; resuming re-observes each video so the lazy
    // observer (the play authority) restarts only the ones actually near
    // the viewport. Drops the section (and its nav link) when there is
    // nothing to show, rather than leaving an empty heading behind.
    function mountComparisonHost(hostId, sectionId, blocks) {
      var host = document.getElementById(hostId);
      if (!host) return;
      if (!blocks.length) {
        dropSection(sectionId);
        return;
      }
      var toggleRow = el('div', 'controls');
      var toggle = el('button', 'chip', 'Pause rotation');
      toggle.type = 'button';
      toggle.addEventListener('click', function () {
        var paused = pausedHosts[hostId] = !pausedHosts[hostId];
        toggle.textContent = paused ? 'Play rotation' : 'Pause rotation';
        var vids = host.querySelectorAll('video');
        for (var i = 0; i < vids.length; i++) {
          if (paused) vids[i].pause();
          else if (lazyObserver) {
            lazyObserver.unobserve(vids[i]);
            lazyObserver.observe(vids[i]);   // re-delivers the current intersection
          } else {
            var p = vids[i].play();
            if (p && p.catch) p.catch(function () {});
          }
        }
      });
      toggleRow.appendChild(el('span', 'small muted', 'Click on images to open at full resolution.'));
      toggleRow.appendChild(toggle);
      host.appendChild(toggleRow);
      // All datasets stack vertically, in manifest order -- no picker.
      blocks.forEach(function (block) {
        host.appendChild(buildComparison(block));
      });
    }

    mountComparisonHost('comparisons', 'comparisons-sec', DATA.comparisons || []);
    mountComparisonHost('limitations', 'limitations-sec', DATA.limitations || []);

    var galHost = document.getElementById('galleries');
    if (galHost) {
      var galleries = DATA.galleries || [];
      // Once every scene has a baseline orbit, the comparisons cover them all
      // and there is nothing left over — drop the section rather than leave an
      // empty heading with a live nav link pointing at it.
      if (!galleries.length) dropSection('results');
      else galleries.forEach(function (b) { galHost.appendChild(buildGallery(b)); });
    }
  }

  function dropSection(id) {
    var sec = document.getElementById(id);
    if (sec) {
      var rule = sec.nextElementSibling;
      if (rule && rule.tagName === 'HR') rule.remove();
      sec.remove();
    }
    var link = document.querySelector('.navbar-links a[href="#' + id + '"]');
    if (link && link.parentNode) link.parentNode.remove();
  }

  /* Tables come from their own generated file (tables.js), because they are
   * refreshed by a different pipeline than the renders: the paper's CSVs. Each
   * section carries the page section it belongs to — cross-method results, or
   * ablations of our own components. */
  function mountTables() {
    var hosts = {
      results: document.getElementById('tables-results'),
      ablations: document.getElementById('tables-ablations')
    };
    if (!hosts.results && !hosts.ablations) return;
    if (!window.PAGE_TABLES) {
      setupNote(hosts.results || hosts.ablations,
        '<strong>No <code>tables.js</code> found.</strong> Generate it with:' +
        '<br><code>python paper/webpage/build_tables.py</code>');
      return;
    }
    window.PAGE_TABLES.forEach(function (block) {
      var host = hosts[block.group] || hosts.results || hosts.ablations;
      host.appendChild(buildTableSection(block));
    });
  }

  /* --- the paper's own figures --------------------------------------------- */

  function missingNote(host, what) {
    setupNote(host, what + ' not collected yet — run ' +
      '<code>python paper/webpage/collect_assets.py</code>.');
  }

  /* The teaser on wide screens is the paper's one grid; on a phone it would
   * need five plates across, so there it is split into one grid per setting
   * (a column with a header starts a setting; a header-less one belongs to
   * the setting before it) and those are stacked. Each slice keeps the full
   * grid's plate shapes via `unitWidth`, so the wide dynamic plate is not
   * cropped square. Rebuilt whenever the viewport crosses the breakpoint,
   * which must match the `max-width` of the mobile block in app.css. */
  function teaserSlices(t) {
    var groups = [];
    (t.columns || []).forEach(function (c, i) {
      if (c.header != null || !groups.length) groups.push([]);
      groups[groups.length - 1].push(i);
    });
    var unit = Math.min.apply(null, t.columns.map(function (c) { return c.width || 1; }));
    return groups.map(function (idx) {
      var slice = {};
      Object.keys(t).forEach(function (k) { slice[k] = t[k]; });
      slice.unitWidth = unit;
      slice.columns = idx.map(function (i) { return t.columns[i]; });
      slice.rows = (t.rows || []).map(function (r) {
        var row = {};
        Object.keys(r).forEach(function (k) { row[k] = r[k]; });
        row.cells = idx.map(function (i) { return (r.cells || [])[i]; });
        return row;
      });
      return slice;
    });
  }

  function mountTeaser(host, t) {
    var narrow = window.matchMedia('(max-width: 720px)');
    function render() {
      host.innerHTML = '';
      if (!narrow.matches) { host.appendChild(buildTeaserGrid(t)); return; }
      var stack = el('div', 'teaser-stack');
      teaserSlices(t).forEach(function (slice) { stack.appendChild(buildTeaserGrid(slice)); });
      host.appendChild(stack);
    }
    render();
    if (narrow.addEventListener) narrow.addEventListener('change', render);
    else if (narrow.addListener) narrow.addListener(render);   // older Safari
  }

  /* The teaser, laid out the way figures/teaser.tex lays it out: a row-label column
   * plus one column per setting, a header row where a column with no header of its
   * own extends its neighbour's (the tracks panel shares "Monocular-Dynamic"), and a
   * method badge in the bottom row's cells. The column widths are the paper's own
   * measured fractions, carried in the manifest. */
  function buildTeaserGrid(t) {
    var cols = t.columns || [];
    var grid = el('div', 'teaser-grid');
    // A label column (fixed) then one fr-weighted column per panel, so the row keeps
    // the paper's proportions at any page width.
    grid.style.gridTemplateColumns = 'auto ' + cols.map(function (c) {
      return (c.width || 1) + 'fr';
    }).join(' ');

    /* One header row, spanning: a cell covers its own column plus every
     * following column whose key is null. Called twice, so a grid can carry
     * the paper's two levels — a band (`superheader`, e.g. "5 views (0–4)")
     * over the method names — which is what lets one grid hold several groups
     * side by side under one set of row labels instead of stacking them.
     * `null` merges leftward, `""` draws an empty cell that still BREAKS the
     * run (how a trailing GT column stays outside the last band). */
    // Where a new superheader band starts (view-count group, or a trailing
    // GT column) -- a stronger vertical rule goes here, on every row, so the
    // groups read as separate blocks rather than one run of columns with
    // only the band row's text telling them apart.
    function groupStart(i) {
      return i > 0 && cols[i].superheader != null;
    }
    // One representative element per boundary column, first one built wins --
    // used after the grid is complete to draw ONE continuous line per
    // boundary (a border-left on every row's cell instead breaks at each
    // row's grid-gap, reading as a dashed line rather than a straight one).
    var groupStartEl = {};
    function markGroupStart(el2, i) {
      if (!groupStart(i)) return;
      el2.classList.add('teaser-group-start');
      if (!groupStartEl[i]) groupStartEl[i] = el2;
    }
    function headerRow(key, cls) {
      if (!cols.some(function (c) { return c[key] != null; })) return;
      grid.appendChild(el('div', 'teaser-corner'));     // above the row labels
      cols.forEach(function (c, i) {
        if (c[key] == null) return;
        var span = 1;
        while (i + span < cols.length && cols[i + span][key] == null) span++;
        var h = el('div', cls, c[key]);
        markGroupStart(h, i);
        h.style.gridColumn = 'span ' + span;
        grid.appendChild(h);
      });
    }
    headerRow('superheader', 'teaser-superhead');
    headerRow('header', 'teaser-head');

    // The paper's box model: every plate in a row shares ONE height (\tplateH =
    // 0.9 x the narrow plate's width) and the image is cover-cropped to fill it,
    // so a wide render and a tall one still line up. Reproduced by giving each
    // cell an aspect ratio proportional to its width weight over the narrowest.
    // `plateHeight` is that 0.9; a grid of square renders passes 1 so nothing
    // is cropped away.
    // `unitWidth` lets a slice of a larger grid (the teaser's per-setting
    // mobile grids) keep the full grid's plate shapes rather than re-deriving
    // them from its own narrowest column.
    var minW = t.unitWidth ||
      Math.min.apply(null, cols.map(function (c) { return c.width || 1; }));
    var plateH = t.plateHeight || 0.9;

    // Hovering a tile magnifies the SAME source point in every tile it is
    // compared against — the interactive stand-in for the paper's baked-in zoom
    // insets. The scope follows what the grid's axes MEAN, since magnifying one
    // point across two tiles only says something when they show the same thing:
    //   default — one column (the teaser: ours, its baseline and the input they
    //             share, one scene per column);
    //   "all"   — the whole grid (the appearance ladder: one scene, and a ladder
    //             is read across its steps);
    //   "row"   — one row (the supplementary's figures: rows are scenes, columns
    //             are methods, so the comparable set is one scene on one view.
    //             "all" there would magnify the same point on a different
    //             object).
    var mode = t.zoom;
    var colImgs = cols.map(function () { return []; });
    var rowImgs = [];
    function bucket(r) {
      return (rowImgs[r] || (rowImgs[r] = []));
    }
    function zoomGroup(i, r) {
      if (mode === 'all') return colImgs.reduce(function (a, b) { return a.concat(b); }, []);
      if (mode === 'row') return bucket(r);
      return colImgs[i];
    }
    function wireZoom(cell, i, ri) {
      cell.addEventListener('pointermove', function (e) {
        var b = cell.getBoundingClientRect();
        var ox = ((e.clientX - b.left) / b.width * 100).toFixed(1) + '%';
        var oy = ((e.clientY - b.top) / b.height * 100).toFixed(1) + '%';
        zoomGroup(i, ri).forEach(function (m) {
          m.style.transformOrigin = ox + ' ' + oy;
          m.style.transform = 'scale(2)';
        });
      });
      cell.addEventListener('pointerleave', function () {
        zoomGroup(i, ri).forEach(function (m) { m.style.transform = ''; });
      });
    }

    // Clicking a plate opens it full size. The magnifier is dropped first: the
    // overlay covers the cell, so the pointerleave that would clear it is not
    // guaranteed to arrive, and the tile would stay stuck at scale(2) behind.
    // `headerOf` walks back to the column the header spans from, so a
    // header-less column captions as the setting it belongs to.
    function headerOf(i) {
      for (var j = i; j >= 0; j--) { if (cols[j] && cols[j].header) return cols[j].header; }
      return null;
    }
    function wireOpen(host, i, ri, src, caption) {
      host.classList.add('enlargeable');
      host.addEventListener('click', function () {
        zoomGroup(i, ri).forEach(function (m) { m.style.transform = ''; });
        openLightbox(src, caption);
      });
    }
    function captionFor(label, i, badge) {
      var head = headerOf(i);
      return (badge ? label + ' (' + badge + ')' : label) + (head ? ' — ' + head : '');
    }

    (t.rows || []).forEach(function (row, ri) {
      grid.appendChild(el('div', 'teaser-rowlabel', row.label));
      (row.cells || []).forEach(function (entry, i) {
        var cell = el('div', 'teaser-cell');
        markGroupStart(cell, i);
        cell.style.aspectRatio = ((cols[i] || {}).width || 1) / minW + ' / ' + plateH;
        // A cell is either a bare src (the teaser) or `{src, badge}` -- a
        // number decoupled from the render, drawn as an HTML overlay rather
        // than baked into the pixels, so it reads straight from the manifest
        // rather than from whatever the image happens to show.
        var metric = (entry && typeof entry === 'object') ? entry.badge : null;
        var src = (entry && typeof entry === 'object') ? entry.src : entry;
        if (src) {
          var media = makeMediaNode(src);
          cell.appendChild(media);
          colImgs[i].push(media);
          bucket(ri).push(media);
          wireZoom(cell, i, ri);
          // Method badge only where the manifest names one for this column, and
          // only in rows other than the first -- the paper badges the
          // baselines, not "Ours".
          var methodBadge = (cols[i] || {}).badge;
          var badged = methodBadge && row !== t.rows[0];
          if (badged) cell.appendChild(el('span', 'teaser-badge', methodBadge));
          if (metric) cell.appendChild(el('span', 'metric-badge', metric));
          wireOpen(cell, i, ri, src, captionFor(row.label, i, badged ? methodBadge : null));
        } else {
          cell.classList.add('teaser-cell-missing');
        }
        grid.appendChild(cell);
      });
    });

    // The input strip: what each column was reconstructed FROM, on its own row
    // under the plates (the paper's \tinputsoverlayfalse layout). The thumbnails
    // join their column's zoom group in both directions — hovering one magnifies
    // the plates above it, and hovering a plate magnifies them — so the observed
    // detail can be checked against what each method made of it.
    if (cols.some(function (c) { return (c.inputs || []).length; })) {
      grid.appendChild(el('div', 'teaser-rowlabel', 'Input'));
      var stripRow = (t.rows || []).length;   // the strip is a row of its own
      cols.forEach(function (c, i) {
        var strip = el('div', 'teaser-inputs');
        markGroupStart(strip, i);
        (c.inputs || []).forEach(function (src) {
          if (!src) { strip.appendChild(el('span', 'teaser-ellipsis', '…')); return; }
          var thumb = el('div', 'teaser-thumb');
          var media = makeMediaNode(src);
          thumb.appendChild(media);
          colImgs[i].push(media);
          bucket(stripRow).push(media);
          wireZoom(thumb, i, stripRow);
          wireOpen(thumb, i, stripRow, src, captionFor('Input', i, null));
          strip.appendChild(thumb);
        });
        grid.appendChild(strip);
      });
    }

    // Draw one continuous divider per boundary, top to bottom -- an
    // absolutely-positioned overlay rather than a per-row border, since a
    // border only covers its own cell and breaks at every row's grid-gap.
    // Repositioned on resize because the columns are `fr`-sized (fluid).
    var boundaryCols = Object.keys(groupStartEl);
    if (boundaryCols.length) {
      var lines = boundaryCols.map(function (i) {
        var line = el('div', 'teaser-group-line');
        grid.appendChild(line);
        return { i: i, line: line };
      });
      function positionLines() {
        // Centered in the grid's column-gap (half the gap, minus half the
        // line's own 1px width), then backed off the left margin
        // `.teaser-group-start` adds on its own side -- so the line stays put
        // and that margin reads as spacing after the line, not a shift of the
        // line itself. Both are read from the computed style, so retuning
        // either in app.css keeps the line centred.
        var gap = parseFloat(getComputedStyle(grid).columnGap) || 0;
        lines.forEach(function (p) {
          var el0 = groupStartEl[p.i];
          var margin = parseFloat(getComputedStyle(el0).marginLeft) || 0;
          p.line.style.left = (el0.offsetLeft - margin - gap / 2 - 0.5) + 'px';
        });
      }
      positionLines();
      if (window.ResizeObserver) {
        new ResizeObserver(positionLines).observe(grid);
      } else {
        window.addEventListener('resize', positionLines);
      }
    }
    return grid;
  }

  /* The interactive completion comparison: the input photo beside one orbit
   * render, with a chip per method. Dragging across the orbit scrubs it, so a
   * horizontal drag reads as rotating the object; switching methods keeps the
   * current angle, so the same unobserved side stays in view across methods. */
  function buildCompletionViewer(cv) {
    var host = document.getElementById('completion');
    if (!host) return;

    var byId = {};
    cv.methods.forEach(function (m) { byId[m.id] = m; });
    // Ours is the default, found by id rather than assumed from array
    // position -- collect_assets.py's COMPLETION_VIEWER happens to list it
    // last today, but nothing should break if that list is ever reordered.
    var active = (byId.ours || cv.methods[cv.methods.length - 1]).id;
    // The first clip stays with the page's lazy-loader (makeVideo arms it): the
    // reader must scroll here to use the viewer, so fetching at page load would
    // only compete with the sections above.
    var video = makeVideo(byId[active].src, 'figure-wide');

    var stage = el('div', 'orbit-stage');
    stage.appendChild(video);
    stage.appendChild(el('span', 'orbit-hint', 'drag to rotate'));

    var orbitPane = el('div', 'orbit-pane');
    orbitPane.appendChild(el('p', 'pipeline-title', 'Reconstruction'));
    orbitPane.appendChild(stage);

    /* Drag = rotate: one stage width sweeps one full turn. Playback pauses
     * while dragging and resumes on release from wherever the drag left it. */
    var drag = null;
    stage.addEventListener('pointerdown', function (e) {
      if (!isFinite(video.duration) || !video.duration) return;
      drag = {x: e.clientX, t: video.currentTime};
      video.pause();
      stage.classList.add('dragging');
      stage.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    stage.addEventListener('pointermove', function (e) {
      if (!drag) return;
      var d = video.duration;
      var t = (drag.t + ((e.clientX - drag.x) / stage.clientWidth) * d) % d;
      video.currentTime = t < 0 ? t + d : t;
    });
    function endDrag(e) {
      if (!drag) return;
      drag = null;
      stage.classList.remove('dragging');
      var p = video.play();
      if (p && p.catch) p.catch(function () {});
    }
    stage.addEventListener('pointerup', endDrag);
    stage.addEventListener('pointercancel', endDrag);

    var inputFig = el('figure', 'orbit-input');
    inputFig.appendChild(el('figcaption', 'pipeline-title', 'Input'));
    inputFig.appendChild(makeMediaNode(cv.input, 'figure-wide'));

    // Input and orbit are equal-width panels side by side (.orbit-images
    // gives both the same flex sizing); the method buttons sit in their own
    // row underneath both, not squeezed above the orbit alone.
    var imagesRow = el('div', 'orbit-images');
    imagesRow.appendChild(inputFig);
    imagesRow.appendChild(orbitPane);

    var controlsRow = el('div', 'orbit-controls');
    var chips = chipRow(controlsRow, 'Method', function () { return active; }, function (id) {
      // Carry the current angle across the swap, so the same unobserved side
      // stays in view: remember it as a fraction, since clips may differ in
      // length, and re-seek once the new clip's duration is known.
      var d = video.duration;
      var frac = (isFinite(d) && d > 0) ? (video.currentTime % d) / d : 0;
      active = id;
      chips.sync();
      video.dataset.src = '';          // bypass the lazy-loader: src changes directly
      video.src = byId[id].src;
      video.addEventListener('loadedmetadata', function seek() {
        video.removeEventListener('loadedmetadata', seek);
        video.currentTime = frac * (video.duration || 0);
      });
      var p = video.play();
      if (p && p.catch) p.catch(function () {});
    });
    chips.set(cv.methods);
    chips.sync();

    var pane = el('div', 'orbit-compare');
    pane.appendChild(imagesRow);
    pane.appendChild(controlsRow);
    host.appendChild(pane);
  }

  /* The supplementary's qualitative figures (qualitative.html only — index.html
   * has no #qualitative host, so this is a no-op there). Each figure is a
   * section of captioned grids drawn by the SAME buildTeaserGrid the teaser and
   * the appearance ladder use; the paper's two-level header (a view-count band
   * over the method names) becomes one grid per band, the band as its caption.
   *
   * The sections and their nav entries are generated rather than written out in
   * the HTML, so which figures exist stays a property of the manifest — and it
   * runs BEFORE wireNav(), whose scroll-spy only observes sections present when
   * it is called. */
  function mountQualitative() {
    var host = document.getElementById('qualitative');
    if (!host) return;
    // One manifest, several pages: the mount div names the group it shows
    // (`data-figures`), so a figure moves between pages by changing its `page`
    // in collect_assets.py and nothing here.
    var want = host.dataset.figures;
    var figures = ((DATA && DATA.qualitative) || []).filter(function (f) {
      return !want || f.page === want;
    });
    if (!figures.length) {
      missingNote(host, 'The supplementary qualitative figures');
      return;
    }
    var nav = document.querySelector('.navbar-links');
    figures.forEach(function (fig) {
      if (!(fig.grids || []).length) return;   // nothing synced for it yet
      var sec = el('section', null);
      sec.id = fig.id;
      sec.appendChild(el('h2', 'section-title', fig.title));
      appendBlurb(sec, fig.blurb);
      // The input pool sits to the LEFT of the comparison, where the paper's
      // own `P` column puts it, so the grids share a row rather than stacking.
      var row = el('div', 'qual-row');
      sec.appendChild(row);
      (fig.grids || []).forEach(function (g) {
        var block = el('figure', 'qual-block' + (g.kind ? ' qual-block-' + g.kind : ''));
        if (g.caption) block.appendChild(el('figcaption', 'table-title', g.caption));
        // These run to 19 columns of detail tiles; rather than shrink them past
        // legibility (the paper's \adjustbox does that, on paper) each grid
        // scrolls inside its own box, the fix the wide result tables already
        // use. The floor is per grid, so a 4-column one never scrolls.
        var scroll = el('div', 'grid-scroll');
        var grid = buildTeaserGrid(g);
        grid.classList.add('qual-grid');
        grid.style.minWidth = (74 * (g.columns || []).length + 110) + 'px';
        scroll.appendChild(grid);
        block.appendChild(scroll);
        row.appendChild(block);
      });
      host.appendChild(sec);
      host.appendChild(el('hr', 'rule'));
      if (nav) {
        var li = document.createElement('li');
        var a = document.createElement('a');
        a.href = '#' + fig.id;
        a.textContent = fig.title;
        li.appendChild(a);
        nav.appendChild(li);
      }
    });
  }

  /* Teaser and pipeline come from paper/figures/ via the manifest rather than
   * fixed paths in index.html, because their extension follows --still-format. */
  function mountStatic() {
    var s = (DATA && DATA.static) || {};

    // The appearance ladder is the same headed grid as the teaser, so it is
    // drawn by the same builder rather than a second near-copy of it.
    var appHost = document.getElementById('appearance');
    if (appHost) {
      if (s.appearance) appHost.appendChild(buildTeaserGrid(s.appearance));
      else missingNote(appHost, 'Appearance ablation figure');
    }

    var teaserHost = document.getElementById('teaser');
    if (teaserHost) {
      // Object = the paper's 2x4 grid (see buildTeaserGrid); string = a single
      // flattened figure, which is what the manifest held before the grid existed.
      if (s.teaser && typeof s.teaser === 'object') mountTeaser(teaserHost, s.teaser);
      else if (s.teaser) teaserHost.appendChild(makeMediaNode(s.teaser, 'figure-wide'));
      else missingNote(teaserHost, 'Teaser');
    }

    /* Pipeline and completion share one shape: titled panels whose widths make
     * the rows (narrow/wide diagram pairs; five equal completion tiles). */
    function mountPanels(hostId, panels, label) {
      var host = document.getElementById(hostId);
      if (!host) return;
      if (!panels) { missingNote(host, label); return; }
      var grid = el('div', 'pipeline-grid');
      panels.forEach(function (p) {
        var panel = el('figure', 'pipeline-panel');
        // same narrow/wide split the paper's pipeline.tex uses; <100% total, so
        // it wraps into rows on its own
        panel.style.flexBasis = 'calc(' + p.width + '% - 12px)';
        var caption = el('figcaption', 'pipeline-title');
        caption.innerHTML = p.title;  // authored, not user input
        panel.appendChild(caption);
        panel.appendChild(makeMediaNode(p.src));
        grid.appendChild(panel);
      });
      host.appendChild(grid);
    }
    mountPanels('pipeline', s.pipeline, 'Pipeline diagrams');
    // The completion section prefers its interactive twin (method tabs over a
    // drag-to-rotate orbit); the still row from the paper's figure is the
    // fallback while a method's orbit run is missing.
    if (s.completion_viewer) buildCompletionViewer(s.completion_viewer);
    else mountPanels('completion', s.completion, 'Completion figure');

    var runtimeHost = document.getElementById('runtime');
    if (runtimeHost) {
      if (s.runtime) runtimeHost.appendChild(makeMediaNode(s.runtime, 'figure-wide'));
      else missingNote(runtimeHost, 'Runtime plot');
    }
  }

  /* --- nav scroll-spy ------------------------------------------------------ */

  /* Smooth scrolling itself is CSS (scroll-behavior on <html>); this only marks
   * which section you are currently in. The observer band starts just below the
   * sticky bar and ends 60% down, so "current" means "at the top of the view". */
  function wireNav() {
    var nav = document.getElementById('navbar');
    if (!nav || !('IntersectionObserver' in window)) return;

    var links = {}, sections = [];
    Array.prototype.forEach.call(nav.querySelectorAll('a[href^="#"]'), function (a) {
      var sec = document.getElementById(a.getAttribute('href').slice(1));
      if (!sec || sec.tagName !== 'SECTION') return;
      links[sec.id] = a;
      sections.push(sec);
    });
    if (!sections.length) return;

    var onScreen = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { onScreen[e.target.id] = e.isIntersecting; });
      var current = sections.filter(function (s) { return onScreen[s.id]; })[0];
      if (!current) return;   // between bands: keep the last highlight
      Object.keys(links).forEach(function (id) {
        links[id].classList.toggle('active', id === current.id);
      });
    }, { rootMargin: '-56px 0px -60% 0px' });

    sections.forEach(function (s) { io.observe(s); });
  }

  /* --- BibTeX copy button -------------------------------------------------- */

  function wireCopy() {
    var btn = document.getElementById('copy-bibtex');
    var pre = document.getElementById('bibtex');
    if (!btn || !pre) return;
    btn.addEventListener('click', function () {
      var text = pre.textContent;
      var done = function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      } else {
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch (e) { /* ignore */ }
        document.body.removeChild(ta);
      }
    });
  }

  /* --- light / dark theme ------------------------------------------------- */

  /* With no saved choice the theme follows the system (the CSS media query).
   * The navbar button saves an explicit choice as data-theme on <html>, which
   * the one-line script in each page's <head> restores before first paint, so
   * a reload never flashes the other theme first. */
  var SUN_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2' +
    'M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  var MOON_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

  // Not called for now: the page is pinned to light (see the <head> of each
  // page). Call it before wireNavMenu() in the init at the bottom to bring
  // the toggle back.
  function wireThemeToggle() {
    var inner = document.querySelector('#navbar .navbar-inner');
    if (!inner) return;
    var root = document.documentElement;
    var system = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
    function current() {
      return root.getAttribute('data-theme') || (system && system.matches ? 'dark' : 'light');
    }
    var btn = el('button', 'theme-toggle');
    btn.type = 'button';
    function render() {
      var dark = current() === 'dark';
      btn.innerHTML = dark ? SUN_ICON : MOON_ICON;   // the icon shows what a click switches TO
      var label = dark ? 'Switch to light mode' : 'Switch to dark mode';
      btn.setAttribute('aria-label', label);
      btn.title = label;
    }
    btn.addEventListener('click', function () {
      var next = current() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('genia-theme', next); } catch (e) { /* private mode */ }
      render();
    });
    if (system && system.addEventListener) system.addEventListener('change', render);
    else if (system && system.addListener) system.addListener(render);   // older Safari
    render();
    inner.appendChild(btn);
  }

  /* --- hamburger menu ------------------------------------------------------ */

  /* Collapse the link row into a toggle whenever it does not fit beside the
   * brand. Measured, not a fixed breakpoint: the pages carry different numbers
   * of links (the subpages append one per figure), and the fonts change size
   * at the mobile breakpoint. Runs after every mount, so all links exist. */
  function wireNavMenu() {
    var nav = document.getElementById('navbar');
    if (!nav) return;
    var inner = nav.querySelector('.navbar-inner');
    var list = nav.querySelector('.navbar-links');
    var brand = nav.querySelector('.navbar-brand');
    if (!inner || !list) return;
    list.id = list.id || 'navbar-links';

    var btn = el('button', 'navbar-toggle');
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Menu');
    btn.setAttribute('aria-controls', list.id);
    btn.setAttribute('aria-expanded', 'false');
    btn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M3 12h18M3 18h18" ' +
                    'stroke="currentColor" stroke-width="2" stroke-linecap="round" fill="none"/></svg>';
    inner.appendChild(btn);

    function setOpen(open) {
      nav.classList.toggle('nav-open', open);
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    function fit() {
      // Measure the row laid out as it would be when expanded, in the same
      // frame, so the page never paints the intermediate state.
      nav.classList.remove('nav-collapsed');
      var gap = parseFloat(getComputedStyle(inner).columnGap) || 0;
      var theme = inner.querySelector('.theme-toggle');
      var need = list.scrollWidth + (brand ? brand.offsetWidth + gap : 0) +
                 (theme ? theme.offsetWidth + gap : 0);
      var collapsed = need > inner.clientWidth;
      nav.classList.toggle('nav-collapsed', collapsed);
      if (!collapsed) setOpen(false);
    }

    btn.addEventListener('click', function () { setOpen(!nav.classList.contains('nav-open')); });
    list.addEventListener('click', function (e) { if (e.target.closest('a')) setOpen(false); });
    document.addEventListener('click', function (e) { if (!nav.contains(e.target)) setOpen(false); });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && nav.classList.contains('nav-open')) { setOpen(false); btn.focus(); }
    });

    fit();
    window.addEventListener('resize', fit);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);
  }

  /* --- 3D gallery --------------------------------------------------------- */
  /* A strip of the scenes' input views (each tile built like the comparisons'
   * "Input views" tile, buildInputsPanel), the comparisons' pager, and ONE
   * viewer: viewer3d.html (3DView) in an iframe. Nothing 3D loads until "Load
   * 3D" is clicked; after that, picking a scene re-points the same iframe,
   * which drops the previous scene's WebGL context. Entries: DATA.gallery3d
   * (GALLERY3D in collect_assets.py), paths already resolved by useAssetBase. */
  function mountGallery3d() {
    var host = document.getElementById('gallery3d');
    var gallery = DATA.gallery3d || [];
    if (!host || !gallery.length) return;

    function viewerUrl(g) {
      var q = new URLSearchParams();
      g.assets.forEach(function (a) { q.append('asset', a); });
      if (g.assets.length > 1) q.set('group', g.temporal ? 'temporal' : 'separate');
      q.set('label', g.title);
      if (g.tracks) q.set('tracks', g.tracks);
      if (g.colmap) q.set('colmap', g.colmap);
      if (g.view) q.set('view', g.view.position.concat(g.view.target).join(','));
      return 'viewer3d.html?' + q.toString();
    }

    var strip = el('div', 'g3d-strip');
    var tiles = gallery.map(function (g, i) {
      var tile = el('button', 'g3d-scene');
      tile.type = 'button';
      tile.setAttribute('aria-label', g.title);
      var panel = el('div', 'pair-panel pair-panel-inputs');
      var grid = el('div', 'pair-inputs-grid');
      grid.classList.toggle('single-input', g.views.length === 1);
      g.views.forEach(function (src) {
        var thumb = el('div', 'pair-input-thumb');
        var img = el('img');
        img.src = src;
        img.loading = 'lazy';
        img.alt = '';
        thumb.appendChild(img);
        grid.appendChild(thumb);
      });
      panel.appendChild(grid);
      panel.appendChild(el('span', 'vc-label teaser-badge', g.title));
      tile.appendChild(panel);
      tile.addEventListener('click', function () { select(i); });
      strip.appendChild(tile);
      return tile;
    });

    var pager = el('div', 'scene-pager');
    var prev = el('button', 'pager-arrow', '‹');
    var next = el('button', 'pager-arrow', '›');
    var dotRow = el('div', 'pager-dots');
    prev.type = next.type = 'button';
    prev.setAttribute('aria-label', 'Previous scene');
    next.setAttribute('aria-label', 'Next scene');
    prev.addEventListener('click', function () { select(cur - 1); });
    next.addEventListener('click', function () { select(cur + 1); });
    var dots = gallery.map(function (g, i) {
      var d = el('button', 'pager-dot');
      d.type = 'button';
      d.title = g.title;
      d.setAttribute('aria-label', g.title);
      d.addEventListener('click', function () { select(i); });
      dotRow.appendChild(d);
      return d;
    });
    pager.appendChild(prev);
    pager.appendChild(dotRow);
    pager.appendChild(next);
    pager.hidden = gallery.length < 2;

    // Until the first load: the scene's first view, washed out, behind a button.
    var cur = -1, frame = null;
    var view = el('div', 'g3d-view');
    var poster = el('img', 'g3d-poster');
    poster.alt = '';
    var load = el('button', 'g3d-load');
    load.type = 'button';
    load.appendChild(el('span', '', '▶ Load 3D'));
    load.addEventListener('click', function () {
      frame = el('iframe');
      view.innerHTML = '';
      view.appendChild(frame);
      show();
    });
    view.appendChild(poster);
    view.appendChild(load);

    host.appendChild(strip);
    host.appendChild(pager);
    host.appendChild(view);

    function show() {
      var g = gallery[cur];
      if (frame) {
        frame.src = viewerUrl(g);
        frame.title = g.title;
      } else {
        poster.src = g.views[0];
      }
    }
    function select(i) {
      i = Math.min(Math.max(i, 0), gallery.length - 1);
      if (i === cur) return;
      cur = i;
      tiles.forEach(function (t, k) { t.classList.toggle('active', k === i); });
      dots.forEach(function (d, k) { d.classList.toggle('active', k === i); });
      // The strip shows a few tiles at a time: bring the picked one into it
      // (by hand, since scrollIntoView could scroll the page too).
      var t = tiles[i], left = t.offsetLeft - strip.offsetLeft;
      if (left < strip.scrollLeft) strip.scrollTo({ left: left, behavior: 'smooth' });
      else if (left + t.offsetWidth > strip.scrollLeft + strip.clientWidth) {
        strip.scrollTo({ left: left + t.offsetWidth - strip.clientWidth, behavior: 'smooth' });
      }
      prev.disabled = i <= 0;
      next.disabled = i >= gallery.length - 1;
      show();
    }
    select(0);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      mountAll(); mountStatic(); mountGallery3d(); mountQualitative(); mountTables(); wireCopy(); wireNav(); wireNavMenu();
    });
  } else {
    mountAll();
    mountStatic();
    mountGallery3d();
    mountQualitative();
    mountTables();
    wireCopy();
    wireNav();
    wireNavMenu();
  }
})();
