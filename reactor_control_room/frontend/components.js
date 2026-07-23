/* Reactor Control Room - shared visual component library (FUTURE.md item 1).
 *
 * Four reusable building blocks, styled to match the approved mockup, composed
 * into all four pages: the particle/glow CHAMBER, an animated GAUGE/dial, a
 * value-vs-reference-vs-tolerance COMPARISON BAR, and a trend SPARKLINE (plus a
 * couple of small primitives: an arc gauge and a meter bar).
 *
 * IMPORTANT: these components only RENDER numbers they are handed. They never
 * compute a physics or AI result - the caller always fetches those from the
 * backend (see api.js) and passes them in. What IS computed here is purely
 * VISUAL: how a value maps to an arc length, a bar position, a glow colour, or a
 * particle's motion. That is presentation, not the model. */

// ---------------------------------------------------------------------------
// Small helpers (formatting + view math only)
// ---------------------------------------------------------------------------
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

const SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
  '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻', '+': '⁺' };
const superscript = (n) => String(n).split('').map((c) => SUP[c] || c).join('');

/** Scientific notation with a superscript exponent, e.g. 9.07e16 -> "9.07×10¹⁶". */
export function fmtSci(v) {
  if (!isFinite(v) || v <= 0) return '0';
  const e = Math.floor(Math.log10(v));
  return (v / Math.pow(10, e)).toFixed(2) + '×10' + superscript(e);
}

const fmtNum = (v, dec = 2) => (isFinite(v) ? v.toFixed(dec) : '—');

const ARC_PATH = 'M23.13 80.87 A38 38 0 1 1 76.87 80.87'; // 270° arc, pathLength 1000
const arcOffset = (frac) => 1000 * (1 - clamp(frac, 0, 1));
const GAUGE_TRANSITION = 'stroke-dashoffset 550ms cubic-bezier(0.22,1,0.36,1), stroke 300ms linear';

/** Defect-probability status colour (matches the mockup's thresholds). */
export const defectStatusColor = (v) => (v <= 0.5 ? 'var(--green)' : v <= 0.55 ? 'var(--amber)' : 'var(--red)');

function el(tag, cssText, html) {
  const node = document.createElement(tag);
  if (cssText) node.style.cssText = cssText;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

// ---------------------------------------------------------------------------
// Arc gauge primitive: the 270° dial shared by every gauge/donut on the app
// ---------------------------------------------------------------------------
export function createArcGauge({ strokeWidth = 8, band = null } = {}) {
  const root = el('div', 'position:relative;width:100%;height:100%');
  root.innerHTML = `
    <svg viewBox="0 0 100 100" style="width:100%;height:100%;overflow:visible">
      <path d="${ARC_PATH}" fill="none" stroke="#161C26" stroke-width="${strokeWidth}" stroke-linecap="round" pathLength="1000"></path>
      ${band ? `<path d="${ARC_PATH}" fill="none" stroke="${band.color}" stroke-width="${strokeWidth}" stroke-linecap="butt" pathLength="1000" stroke-dasharray="${(band.hi - band.lo) * 1000} 1000" stroke-dashoffset="${-band.lo * 1000}"></path>` : ''}
      <path class="arc" d="${ARC_PATH}" fill="none" stroke="#38BDF8" stroke-width="${strokeWidth}" stroke-linecap="round" pathLength="1000" stroke-dasharray="1000" stroke-dashoffset="1000" style="transition:${GAUGE_TRANSITION}"></path>
    </svg>
    <div class="center" style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;text-align:center;padding:0 10px"></div>`;
  const arc = root.querySelector('.arc');
  const center = root.querySelector('.center');
  const glowPx = strokeWidth >= 8 ? 7 : 5;
  function update({ frac, color, centerHTML }) {
    if (color !== undefined) {
      arc.setAttribute('stroke', color);
      arc.style.filter = `drop-shadow(0 0 ${glowPx}px ${color})`;
    }
    if (frac !== undefined) arc.style.strokeDashoffset = String(arcOffset(frac));
    if (centerHTML !== undefined) center.innerHTML = centerHTML;
  }
  return { el: root, update, center };
}

// ---------------------------------------------------------------------------
// Meter: a thin horizontal fill bar (footer CPU/RAM, class-probability rows, etc.)
// ---------------------------------------------------------------------------
export function createMeter({ height = 6, gradient = 'var(--cyan)', bg = 'var(--border)' } = {}) {
  const track = el('div', `position:relative;height:${height}px;border-radius:${height / 2}px;background:${bg};overflow:hidden`);
  const fill = el('div', `height:100%;border-radius:${height / 2}px;background:${gradient};width:0%;transition:width 550ms cubic-bezier(0.22,1,0.36,1)`);
  track.appendChild(fill);
  return { el: track, update: (pct) => { fill.style.width = clamp(pct, 0, 100) + '%'; } };
}

// ---------------------------------------------------------------------------
// GAUGE (tile): label + 270° dial + numeric readout + min/max, for the
// Reactor View and Session Replay output panels.
// ---------------------------------------------------------------------------
export function createGauge(cfg) {
  // cfg: {label, min, max, log?, color, unit, decimals?, sci?, band?[lo,hi], status?, maxWidth?}
  const wrap = el('div', 'position:relative;background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 10px 12px;display:flex;flex-direction:column;align-items:center;gap:8px');

  const statusDot = el('span', 'position:absolute;top:10px;right:10px;width:7px;height:7px;border-radius:50%;display:none');
  wrap.appendChild(statusDot);

  wrap.appendChild(el('div', 'font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--text-dim);font-weight:600;text-align:center', cfg.label));

  const box = el('div', `position:relative;width:100%;max-width:${cfg.maxWidth || 128}px;aspect-ratio:1/1`);
  const arc = createArcGauge({ strokeWidth: 7, band: cfg.band ? { lo: cfg.band[0], hi: cfg.band[1], color: 'rgba(239,68,68,0.28)' } : null });
  box.appendChild(arc.el);
  wrap.appendChild(box);

  const sciLabel = (x) => '10' + superscript(Math.round(Math.log10(x)));
  const numLabel = (x) => (x >= 1000 ? x / 1000 + 'k' : '' + x);
  const minLabel = cfg.sci ? sciLabel(cfg.min) : numLabel(cfg.min);
  const maxLabel = cfg.sci ? sciLabel(cfg.max) : numLabel(cfg.max);
  wrap.appendChild(el('div',
    `display:flex;justify-content:space-between;width:100%;max-width:${cfg.maxWidth || 128}px;font-family:var(--mono);font-size:8.5px;color:var(--text-mute-3)`,
    `<span>${minLabel}</span><span>${maxLabel}</span>`));

  const fracOf = (v) => (cfg.log
    ? (Math.log10(v) - Math.log10(cfg.min)) / (Math.log10(cfg.max) - Math.log10(cfg.min))
    : (v - cfg.min) / (cfg.max - cfg.min));

  function update(value) {
    const color = cfg.status ? defectStatusColor(value) : cfg.color;
    const readout = cfg.sci ? fmtSci(value) : fmtNum(value, cfg.decimals ?? 2);
    arc.update({
      frac: clamp(fracOf(value), 0, 1),
      color,
      centerHTML: `<div style="font-family:var(--mono);font-size:16px;font-weight:600;color:var(--text);line-height:1;letter-spacing:-0.3px">${readout}</div>`
        + `<div style="font-size:9px;letter-spacing:0.4px;color:var(--text-dim)">${cfg.unit}</div>`,
    });
    if (cfg.status) { statusDot.style.display = 'block'; statusDot.style.background = color; statusDot.style.boxShadow = '0 0 8px ' + color; }
  }
  return { el: wrap, update };
}

// ---------------------------------------------------------------------------
// COMPARISON BAR: one physics-validation check - computed deviation plotted
// against its ±tolerance band, centred on the reference, pass/fail coloured.
// ---------------------------------------------------------------------------
const DEV_DOMAIN = 60; // % deviation mapped to the full bar width (matches the mockup)

const prettifyName = (name) => {
  const s = name.replace(/_/g, ' ').replace(/\bvs\b/, 'vs').trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
};

export function createComparisonBar() {
  const card = el('div', 'background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 16px;display:flex;flex-direction:column;gap:11px');
  card.innerHTML = `
    <div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px">
      <span class="c-title" style="font-size:13px;font-weight:600;color:var(--text)"></span>
      <span class="c-badge" style="font-family:var(--mono);font-size:9px;letter-spacing:1px;padding:3px 8px;border-radius:5px"></span>
    </div>
    <div style="position:relative;height:34px">
      <div style="position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);height:8px;border-radius:4px;background:var(--border-3);border:1px solid var(--border)"></div>
      <div class="c-band"></div>
      <div style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:2px;height:16px;background:var(--text-mute)"></div>
      <div style="position:absolute;left:50%;bottom:-2px;transform:translateX(-50%);font-family:var(--mono);font-size:8px;color:var(--text-mute)">ref</div>
      <div class="c-marker"></div>
    </div>
    <div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap">
      <span style="font-family:var(--mono);font-size:11px;color:var(--text-dim)">computed <span class="c-computed" style="color:var(--text)"></span> · ref <span class="c-ref" style="color:var(--text)"></span> <span class="c-unit"></span></span>
      <span class="c-dev" style="font-family:var(--mono);font-size:11px"></span>
    </div>
    <div style="display:flex;align-items:center;gap:7px;border-top:1px solid var(--border-3);padding-top:9px">
      <span style="width:5px;height:5px;border-radius:50%;background:var(--cyan);flex:none;box-shadow:0 0 6px var(--cyan)"></span>
      <span style="font-size:10px;color:var(--text-mute-2)">Source · <span class="c-source" style="color:var(--text-dim)"></span></span>
    </div>`;

  const $ = (s) => card.querySelector(s);
  function update(check) {
    const pass = check.passed;
    const passColor = pass ? 'var(--green)' : 'var(--red)';
    $('.c-title').textContent = prettifyName(check.name);
    const badge = $('.c-badge');
    badge.textContent = pass ? 'PASS' : 'FAIL';
    badge.style.color = passColor;
    badge.style.background = pass ? 'rgba(52,211,153,0.1)' : 'rgba(239,68,68,0.1)';
    badge.style.border = `1px solid ${pass ? 'rgba(52,211,153,0.35)' : 'rgba(239,68,68,0.35)'}`;

    const devFrac = clamp(check.deviation_pct / DEV_DOMAIN, -1, 1);
    const markerLeft = 50 + devFrac * 50;
    const bandHalf = Math.min(50, (check.tolerance_pct / DEV_DOMAIN) * 50);
    $('.c-band').style.cssText = `position:absolute;top:50%;left:${50 - bandHalf}%;width:${2 * bandHalf}%;transform:translateY(-50%);height:16px;border-radius:4px;background:rgba(52,211,153,0.12);border:1px solid rgba(52,211,153,0.3)`;
    $('.c-marker').style.cssText = `position:absolute;top:50%;left:${markerLeft}%;transform:translate(-50%,-50%);width:12px;height:20px;border-radius:3px;background:${passColor};box-shadow:0 0 10px ${passColor};border:2px solid var(--panel);transition:left 450ms cubic-bezier(0.22,1,0.36,1)`;

    const isSci = Math.abs(check.computed_value) < 1e-3 || Math.abs(check.computed_value) >= 1e4;
    $('.c-computed').textContent = isSci ? fmtSci(check.computed_value) : fmtNum(check.computed_value, 3);
    $('.c-ref').textContent = isSci ? fmtSci(check.reference_value) : fmtNum(check.reference_value, 3);
    $('.c-unit').textContent = check.unit && check.unit !== 'dimensionless' ? check.unit : '';
    const dev = $('.c-dev');
    dev.innerHTML = `Δ ${check.deviation_pct >= 0 ? '+' : ''}${check.deviation_pct.toFixed(1)}% <span style="color:var(--text-mute)">/ ±${check.tolerance_pct.toFixed(0)}%</span>`;
    dev.style.color = passColor;
    // The source citation is long; show a trimmed lead so the card stays compact.
    $('.c-source').textContent = check.source.split('(')[0].trim().replace(/,\s*$/, '');
  }
  return { el: card, update };
}

// ---------------------------------------------------------------------------
// SPARKLINE: a trend line across the session record (Session Replay).
// ---------------------------------------------------------------------------
export function createSparkline({ label, unit, color, decimals = 2 }) {
  const card = el('div', 'background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 16px;display:flex;align-items:center;gap:16px');
  const gradId = 'sp-' + Math.random().toString(36).slice(2, 9);
  card.innerHTML = `
    <div style="width:130px;flex:none;display:flex;flex-direction:column;gap:3px">
      <span style="font-size:11px;letter-spacing:0.5px;color:var(--text-dim);font-weight:600">${label}</span>
      <span class="sp-cur" style="font-family:var(--mono);font-size:18px;font-weight:600;color:${color}">—</span>
      <span style="font-size:9px;color:var(--text-mute)">selected run · ${unit}</span>
    </div>
    <svg viewBox="0 0 240 60" preserveAspectRatio="none" style="flex:1;height:58px;overflow:visible">
      <defs><linearGradient id="${gradId}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${color}" stop-opacity="0.35"></stop><stop offset="100%" stop-color="${color}" stop-opacity="0"></stop></linearGradient></defs>
      <path class="sp-area" fill="url(#${gradId})"></path>
      <path class="sp-line" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter:drop-shadow(0 0 4px ${color})"></path>
      <circle class="sp-dot" r="3.5" fill="${color}" stroke="var(--panel)" stroke-width="1.5"></circle>
    </svg>`;
  const $ = (s) => card.querySelector(s);
  function update({ values, currentIndex }) {
    if (!values || values.length === 0) return;
    const mn = Math.min(...values), mx = Math.max(...values), rng = mx - mn || 1;
    const n = values.length;
    const X = (i) => (n === 1 ? 120 : (i / (n - 1)) * 240);
    const Y = (v) => 60 - ((v - mn) / rng) * 50 - 5;
    const pts = values.map((v, i) => [X(i), Y(v)]);
    const line = 'M' + pts.map((p) => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' L');
    $('.sp-line').setAttribute('d', line);
    $('.sp-area').setAttribute('d', line + ' L240 60 L0 60 Z');
    const sel = clamp(currentIndex, 0, n - 1);
    $('.sp-dot').setAttribute('cx', X(sel).toFixed(1));
    $('.sp-dot').setAttribute('cy', Y(values[sel]).toFixed(1));
    $('.sp-cur').textContent = fmtNum(values[sel], decimals);
  }
  return { el: card, update };
}

// ---------------------------------------------------------------------------
// CHAMBER: the animated particle/glow plasma-chamber cross-section.
// The physics VALUES (n_e, T_e, etc.) are fetched from the backend and passed to
// setOutputs(); everything drawn here - glow, particles, electrodes, wafer - is
// pure visualisation of those values, computed only for rendering.
// ---------------------------------------------------------------------------
export function createChamber({ minHeight = 432 } = {}) {
  const wrap = el('div', `position:relative;background:radial-gradient(circle at 50% 46%, #0b1119 0%, #0D1117 78%);border:1px solid var(--border);border-radius:16px;overflow:hidden;min-height:${minHeight}px`);
  const canvas = el('canvas', 'position:absolute;inset:0;width:100%;height:100%;display:block');
  wrap.appendChild(canvas);
  wrap.insertAdjacentHTML('beforeend', `
    <div style="position:absolute;top:16px;left:18px;display:flex;align-items:center;gap:9px">
      <span style="width:7px;height:7px;border-radius:50%;background:var(--cyan);box-shadow:0 0 8px var(--cyan)"></span>
      <span style="font-size:11px;letter-spacing:1.8px;text-transform:uppercase;color:var(--text-dim);font-weight:600">Plasma Chamber</span>
    </div>
    <span style="position:absolute;top:17px;right:16px;font-size:9px;letter-spacing:0.5px;color:var(--text-mute)">hover to stir plasma</span>
    <div style="position:absolute;left:0;right:0;bottom:16px;display:flex;justify-content:center;gap:12px;flex-wrap:wrap;padding:0 14px">
      <div style="background:rgba(5,7,10,0.55);backdrop-filter:blur(6px);border:1px solid var(--border);border-radius:9px;padding:7px 11px;display:flex;flex-direction:column;gap:2px;min-width:96px">
        <span style="font-size:9px;letter-spacing:1px;color:var(--text-dim);text-transform:uppercase">Density nₑ</span>
        <span class="ch-ne" style="font-family:var(--mono);font-size:13px;font-weight:600;color:var(--purple)">— <span style="color:var(--text-dim);font-size:9px">m⁻³</span></span>
      </div>
      <div style="background:rgba(5,7,10,0.55);backdrop-filter:blur(6px);border:1px solid var(--border);border-radius:9px;padding:7px 11px;display:flex;flex-direction:column;gap:2px;min-width:76px">
        <span style="font-size:9px;letter-spacing:1px;color:var(--text-dim);text-transform:uppercase">Temp Tₑ</span>
        <span class="ch-te" style="font-family:var(--mono);font-size:13px;font-weight:600;color:var(--cyan)">— <span style="color:var(--text-dim);font-size:9px">eV</span></span>
      </div>
      <div style="background:rgba(5,7,10,0.55);backdrop-filter:blur(6px);border:1px solid var(--border);border-radius:9px;padding:7px 11px;display:flex;flex-direction:column;gap:4px;min-width:118px">
        <div style="display:flex;justify-content:space-between;align-items:baseline"><span style="font-size:9px;letter-spacing:1px;color:var(--text-dim);text-transform:uppercase">Intensity</span><span class="ch-int" style="font-family:var(--mono);font-size:11px;color:var(--text)">—</span></div>
        <div class="ch-intbar-track" style="height:5px;border-radius:3px;background:var(--border);overflow:hidden"><div class="ch-intbar" style="height:100%;border-radius:3px;background:linear-gradient(90deg,var(--purple),var(--cyan),var(--orange));width:0%;transition:width 550ms cubic-bezier(0.22,1,0.36,1)"></div></div>
        <span style="font-size:8px;color:var(--text-mute)">nₑ · reactivity · ion energy</span>
      </div>
    </div>`);

  const st = { targets: null, disp: null, particles: [], mouse: null, raf: null, lastT: 0, lastDraw: 0 };

  // --- VIEW mapping: fetched outputs -> a 0..1 glow intensity (not a physics value) ---
  function intensityOf(o) {
    const norm = (k, v) => k === 'plasma_density_m3' ? clamp((Math.log10(v) - 16) / (17.4 - 16), 0, 1)
      : k === 'reactivity_index' ? clamp(v / 1.7, 0, 1)
      : k === 'ion_energy_ev' ? clamp(v / 80, 0, 1) : clamp(v, 0, 1);
    return clamp(0.1 + 0.5 * norm('plasma_density_m3', o.plasma_density_m3)
      + 0.25 * norm('reactivity_index', o.reactivity_index)
      + 0.15 * norm('ion_energy_ev', o.ion_energy_ev), 0, 1);
  }
  function rampColor(I) {
    const s = [[167, 139, 250], [56, 189, 248], [251, 146, 60]];
    const t = clamp(I, 0, 1) * 2, i = Math.min(1, Math.floor(t)), f = t - i;
    const a = s[i], b = s[i + 1];
    return { r: Math.round(a[0] + (b[0] - a[0]) * f), g: Math.round(a[1] + (b[1] - a[1]) * f), b: Math.round(a[2] + (b[2] - a[2]) * f) };
  }
  function ensureParticles(I) {
    const N = Math.min(58, Math.round(14 + I * 30));
    while (st.particles.length < N) st.particles.push({ x: Math.random(), y: Math.random(), vx: (Math.random() * 2 - 1) * 0.05, spd: 0.5 + Math.random() * 1.2, size: 0.6 + Math.random() * 1.9, tp: Math.random() * 7, thermal: Math.random() < 0.34 });
    if (st.particles.length > N) st.particles.length = N;
  }

  function draw(ts, dt) {
    if (!st.targets) return;
    const rect = canvas.getBoundingClientRect(), w = rect.width, h = rect.height;
    if (!w || !h) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    const I = clamp(st.targets.intensity || 0, 0, 1);
    const col = rampColor(I);
    const rgba = (a) => `rgba(${col.r},${col.g},${col.b},${a})`;
    const flick = 0.035 * Math.sin(ts / 90) + 0.025 * Math.sin(ts / 47);
    const pulse = 0.045 * Math.sin(ts / 850);
    const gI = clamp(I + pulse + flick, 0, 1);

    const padX = Math.max(28, w * 0.14);
    const vTop = 48, vBot = h - 76, vx0 = padX, vx1 = w - padX, vW = vx1 - vx0, vH = vBot - vTop, cx = (vx0 + vx1) / 2;
    const elecH = Math.max(9, vH * 0.045);
    const topE = vTop + vH * 0.11, botE = vTop + vH * 0.89 - elecH;
    const pTop = topE + elecH, pBot = botE, pH = pBot - pTop;
    const sheath = Math.max(8, pH * 0.13), sf = Math.min(0.4, sheath / pH);
    const bulkTop = pTop + sheath, bulkBot = pBot - sheath, bulkH = bulkBot - bulkTop;
    const rr = (x, y, ww, hh, r) => { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + ww, y, x + ww, y + hh, r); ctx.arcTo(x + ww, y + hh, x, y + hh, r); ctx.arcTo(x, y + hh, x, y, r); ctx.arcTo(x, y, x + ww, y, r); ctx.closePath(); };

    // vessel wall
    ctx.fillStyle = '#0b1017'; ctx.strokeStyle = 'rgba(139,148,158,0.22)'; ctx.lineWidth = 1;
    for (const py of [vTop + vH * 0.3, vTop + vH * 0.7]) { ctx.fillRect(vx0 - 18, py - 6, 10, 12); ctx.strokeRect(vx0 - 18, py - 6, 10, 12); ctx.fillRect(vx1 + 8, py - 6, 10, 12); ctx.strokeRect(vx1 + 8, py - 6, 10, 12); }
    rr(vx0 - 11, vTop - 15, vW + 22, vH + 30, 17);
    const wallG = ctx.createLinearGradient(0, vTop, 0, vBot); wallG.addColorStop(0, '#0d131c'); wallG.addColorStop(0.5, '#090e15'); wallG.addColorStop(1, '#0d131c');
    ctx.fillStyle = wallG; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(139,148,158,0.30)'; ctx.stroke();

    // glow, clipped to interior
    ctx.save(); rr(vx0 - 8, vTop - 12, vW + 16, vH + 24, 14); ctx.clip();
    ctx.globalCompositeOperation = 'lighter';
    const a0 = 0.10 + 0.72 * gI;
    const vg = ctx.createLinearGradient(0, pTop, 0, pBot);
    vg.addColorStop(0, 'rgba(5,7,10,0)'); vg.addColorStop(sf, rgba(a0 * 0.30)); vg.addColorStop(0.5, rgba(a0)); vg.addColorStop(1 - sf, rgba(a0 * 0.16)); vg.addColorStop(1, 'rgba(5,7,10,0)');
    ctx.fillStyle = vg; ctx.fillRect(vx0, pTop, vW, pH);
    const midY = (bulkTop + bulkBot) / 2;
    const rg = ctx.createRadialGradient(cx, midY, 0, cx, midY, Math.max(vW, bulkH) * 0.62);
    rg.addColorStop(0, rgba(0.44 * gI + 0.10)); rg.addColorStop(0.5, rgba(0.20 * gI)); rg.addColorStop(1, 'rgba(5,7,10,0)');
    ctx.fillStyle = rg; ctx.fillRect(vx0, pTop, vW, pH);
    const coreR = Math.max(vW, bulkH) * (0.16 + 0.20 * gI); const rgC = ctx.createRadialGradient(cx, midY, 0, cx, midY, coreR); const cw = 0.26 + 0.6 * gI;
    rgC.addColorStop(0, `rgba(255,255,255,${cw.toFixed(3)})`); rgC.addColorStop(0.42, rgba(cw * 0.55)); rgC.addColorStop(1, 'rgba(5,7,10,0)');
    ctx.fillStyle = rgC; ctx.fillRect(vx0, pTop, vW, pH);
    // plasma-sheath boundary lines
    ctx.lineWidth = 1.6; ctx.strokeStyle = rgba(0.32 + 0.5 * gI); ctx.shadowBlur = 10; ctx.shadowColor = rgba(0.9);
    ctx.beginPath(); ctx.moveTo(vx0 + 7, bulkTop); ctx.lineTo(vx1 - 7, bulkTop); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(vx0 + 7, bulkBot); ctx.lineTo(vx1 - 7, bulkBot); ctx.stroke(); ctx.shadowBlur = 0;

    // particles
    ensureParticles(I);
    const mo = (st.mouse && st.mouse.active) ? st.mouse : null;
    const blur = Math.min(9, 3 + 9 * gI);
    for (const pt of st.particles) {
      const dir = pt.y < 0.5 ? -1 : 1;
      const accel = 0.5 + Math.abs(pt.y - 0.5) * 1.6;
      let boost = 1, mf = 0, mdx = 0, mdy = 0;
      if (mo) { const dx = (vx0 + pt.x * vW) - mo.mx, dy = (bulkTop + pt.y * bulkH) - mo.my, Rm = 106, d2 = dx * dx + dy * dy; if (d2 < Rm * Rm) { const d = Math.sqrt(d2) || 1; mf = 1 - d / Rm; boost = 1 + mf * 6.5; mdx = (-dy / d * mf * 6.5) / vW * 0.62; mdy = (dx / d * mf * 6.5) / bulkH * 0.62; } }
      pt.y += dir * (0.011 + I * 0.15) * pt.spd * dt * accel * (pt.thermal ? 1.4 : 1) * 0.62 * boost + mdy;
      pt.x += ((pt.vx * (pt.thermal ? 1.4 : 0.6) + Math.sin(ts / 860 + pt.tp) * 0.014) * dt) * 0.62 * boost + mdx;
      if (pt.y < 0.02 || pt.y > 0.98 || pt.x < 0.02 || pt.x > 0.98) { pt.y = 0.5 + (Math.random() * 2 - 1) * 0.18; pt.x = 0.12 + Math.random() * 0.76; pt.vx = (Math.random() * 2 - 1) * 0.05; }
      const px = vx0 + pt.x * vW, py = bulkTop + pt.y * bulkH, edge = Math.min(pt.y, 1 - pt.y) * 2, s = pt.size * (0.7 + 0.7 * I) * (pt.thermal ? 0.65 : 1);
      const alpha = (0.3 + 0.5 * I) * (0.35 + 0.65 * edge);
      if (!pt.thermal && (I > 0.45 || mf > 0.1)) { const tl = (5 + 20 * I) * pt.spd * (0.4 + Math.abs(pt.y - 0.5)) * 0.62 * boost; ctx.strokeStyle = rgba((alpha * 0.45).toFixed(3)); ctx.lineWidth = s * 0.85; ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, py - dir * tl); ctx.stroke(); }
      ctx.shadowBlur = blur; ctx.shadowColor = rgba(0.9);
      ctx.fillStyle = `rgba(255,255,255,${alpha.toFixed(3)})`;
      ctx.beginPath(); ctx.arc(px, py, s, 0, Math.PI * 2); ctx.fill();
    }
    ctx.shadowBlur = 0; ctx.globalCompositeOperation = 'source-over'; ctx.restore();

    // electrodes + wafer
    const drawElec = (y, powered) => { const g = ctx.createLinearGradient(0, y, 0, y + elecH); g.addColorStop(0, '#3d4653'); g.addColorStop(0.5, '#20272f'); g.addColorStop(1, '#131920'); ctx.fillStyle = g; rr(vx0 + 8, y, vW - 16, elecH, 3); ctx.fill(); ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 1; ctx.stroke(); ctx.fillStyle = 'rgba(230,237,243,0.5)'; ctx.fillRect(vx0 + 9, y + 1, vW - 18, 1); const rG = ctx.createLinearGradient(0, powered ? y + elecH : y, 0, powered ? y : y + elecH); rG.addColorStop(0, rgba(0.28 * gI)); rG.addColorStop(1, 'rgba(0,0,0,0)'); ctx.globalCompositeOperation = 'lighter'; ctx.fillStyle = rG; rr(vx0 + 8, y, vW - 16, elecH, 3); ctx.fill(); ctx.globalCompositeOperation = 'source-over'; };
    drawElec(topE, true); drawElec(botE, false);
    const wafW = (vW - 16) * 0.52, wafX = cx - wafW / 2, wafY = botE - 3.5;
    const wg = ctx.createLinearGradient(0, wafY, 0, wafY + 3.5); wg.addColorStop(0, '#5b6675'); wg.addColorStop(1, '#2b333d'); ctx.fillStyle = wg; rr(wafX, wafY, wafW, 3.5, 1.5); ctx.fill();
    ctx.fillStyle = 'rgba(139,148,158,0.55)'; ctx.font = '600 7.5px ui-monospace,monospace'; ctx.textAlign = 'center'; ctx.fillText('WAFER', cx, wafY - 4);
    // RF feed + ground labels
    ctx.strokeStyle = 'rgba(139,148,158,0.5)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(cx, vTop - 15); ctx.lineTo(cx, topE); ctx.stroke();
    ctx.fillStyle = 'rgba(167,139,250,0.9)'; ctx.font = '600 9.5px ui-monospace,monospace'; ctx.textAlign = 'left'; ctx.fillText('∼ RF 13.56 MHz', cx + 7, vTop - 6);
    ctx.strokeStyle = 'rgba(139,148,158,0.5)'; ctx.beginPath(); ctx.moveTo(cx, botE + elecH); ctx.lineTo(cx, vBot + 7); ctx.stroke();
    const gy = vBot + 7; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(cx - 9, gy); ctx.lineTo(cx + 9, gy); ctx.moveTo(cx - 6, gy + 3); ctx.lineTo(cx + 6, gy + 3); ctx.moveTo(cx - 3, gy + 6); ctx.lineTo(cx + 3, gy + 6); ctx.stroke();
    ctx.fillStyle = 'rgba(139,148,158,0.6)'; ctx.font = '600 8.5px ui-monospace,monospace'; ctx.textAlign = 'right'; ctx.fillText('POWERED', vx1 - 11, topE - 4); ctx.fillText('GROUND', vx1 - 11, botE + elecH + 11); ctx.textAlign = 'left';
    st.lastDraw = Date.now();
  }

  canvas.addEventListener('pointermove', (e) => { const r = canvas.getBoundingClientRect(); st.mouse = { mx: e.clientX - r.left, my: e.clientY - r.top, active: true }; });
  canvas.addEventListener('pointerleave', () => { if (st.mouse) st.mouse.active = false; });

  function setOutputs(sim) {
    st.targets = Object.assign({}, sim, { intensity: intensityOf(sim) });
    wrap.querySelector('.ch-ne').innerHTML = `${fmtSci(sim.plasma_density_m3)} <span style="color:var(--text-dim);font-size:9px">m⁻³</span>`;
    wrap.querySelector('.ch-te').innerHTML = `${fmtNum(sim.electron_temperature_ev, 2)} <span style="color:var(--text-dim);font-size:9px">eV</span>`;
    wrap.querySelector('.ch-int').textContent = st.targets.intensity.toFixed(2);
    wrap.querySelector('.ch-intbar').style.width = (st.targets.intensity * 100).toFixed(1) + '%';
    if (st.lastDraw === 0) draw(performance.now(), 0.033); // paint once immediately
  }
  function start() {
    if (st.raf) return;
    const loop = (ts) => { st.raf = requestAnimationFrame(loop); const dt = Math.min(0.05, (ts - (st.lastT || ts)) / 1000); st.lastT = ts; draw(ts, dt); };
    st.raf = requestAnimationFrame(loop);
  }
  function stop() { if (st.raf) { cancelAnimationFrame(st.raf); st.raf = null; } }

  return { el: wrap, canvas, setOutputs, start, stop };
}
