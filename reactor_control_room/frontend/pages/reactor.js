/* Reactor View - the centerpiece "turn a knob, watch it react" page.
 *
 * Sliders/presets set the operating point; each change re-fetches /api/simulate
 * and the chamber + 10 gauges re-render from the returned values. No physics runs
 * here - moving a slider triggers a backend solve, never a JS calculation. */
import { createChamber, createGauge, clamp, fmtSci } from '../components.js';

// The 10 solved twin outputs, with the display scale each gauge uses.
export const GAUGE_CFG = [
  { key: 'electron_temperature_ev', label: 'Electron Temp', unit: 'eV', min: 0, max: 8, decimals: 2, color: 'var(--cyan)' },
  { key: 'plasma_density_m3', label: 'Plasma Density', unit: 'm⁻³', min: 1e15, max: 1e18, log: true, sci: true, color: 'var(--purple)' },
  { key: 'ion_flux_m2s', label: 'Ion Flux', unit: 'm⁻²s⁻¹', min: 1e19, max: 1e21, log: true, sci: true, color: 'var(--cyan)' },
  { key: 'sheath_voltage_v', label: 'Sheath Voltage', unit: 'V', min: 0, max: 90, decimals: 0, color: 'var(--orange)' },
  { key: 'ion_energy_ev', label: 'Ion Energy', unit: 'eV', min: 0, max: 100, decimals: 0, color: 'var(--orange)' },
  { key: 'reactivity_index', label: 'Reactivity', unit: 'index', min: 0, max: 2, decimals: 3, color: 'var(--purple)' },
  { key: 'uniformity_index', label: 'Uniformity', unit: 'index', min: 0, max: 1, decimals: 3, color: 'var(--cyan)' },
  { key: 'etch_rate_nm_min', label: 'Etch Rate', unit: 'nm/min', min: 0, max: 150, decimals: 1, color: 'var(--orange)' },
  { key: 'process_quality', label: 'Process Quality', unit: '0–1', min: 0, max: 1, decimals: 3, color: 'var(--purple)' },
  { key: 'defect_probability', label: 'Defect Prob.', unit: '0–1', min: 0, max: 1, decimals: 3, status: true, band: [0.55, 1] },
];

const MODES = [
  { label: 'Stable', rf: 100, p: 10, desc: 'Stable Plasma · 100 W / 10 mTorr' },
  { label: 'Explore', rf: 150, p: 10, desc: 'Exploratory Sweep · 150 W / 10 mTorr' },
  { label: 'Stress', rf: 280, p: 1.5, desc: 'Stress Test · 280 W / 1.5 mTorr' },
];

function ele(tag, cssText, html) {
  const n = document.createElement(tag);
  if (cssText) n.style.cssText = cssText;
  if (html !== undefined) n.innerHTML = html;
  return n;
}

// Custom pointer-drag slider matching the mockup (big readout + glowing thumb).
function createSlider({ label, min, max, step, value, color, glow, unit, onInput }) {
  const wrap = ele('div', 'display:flex;flex-direction:column;gap:11px');
  wrap.innerHTML = `
    <div style="display:flex;align-items:baseline;justify-content:space-between">
      <span style="font-size:11px;letter-spacing:1.2px;text-transform:uppercase;color:var(--text-dim);font-weight:600">${label}</span>
      <span class="sl-val" style="font-family:var(--mono);font-size:27px;font-weight:600;color:${color};line-height:1;text-shadow:0 0 16px ${glow}"></span>
    </div>
    <div class="sl-track" style="position:relative;height:26px;display:flex;align-items:center;cursor:pointer;touch-action:none">
      <div style="position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);height:6px;border-radius:3px;background:var(--border)"></div>
      <div class="sl-fill"></div>
      <div class="sl-thumb"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:9px;color:var(--text-mute)"><span>${min}</span><span>${max} ${unit}</span></div>`;
  const track = wrap.querySelector('.sl-track');
  const fill = wrap.querySelector('.sl-fill');
  const thumb = wrap.querySelector('.sl-thumb');
  const valEl = wrap.querySelector('.sl-val');
  let v = value;
  const fmtV = (x) => (step < 1 ? +x.toFixed(1) : Math.round(x));
  function paint() {
    const f = clamp((v - min) / (max - min), 0, 1) * 100;
    fill.style.cssText = `position:absolute;left:0;top:50%;transform:translateY(-50%);height:6px;border-radius:3px;background:${color};box-shadow:0 0 12px ${glow};width:${f}%;pointer-events:none`;
    thumb.style.cssText = `position:absolute;left:${f}%;top:50%;transform:translate(-50%,-50%);width:16px;height:16px;border-radius:50%;background:${color};box-shadow:0 0 14px ${glow};border:2px solid var(--bg);pointer-events:none`;
    valEl.innerHTML = `${fmtV(v)}<span style="font-size:13px;color:var(--text-dim);margin-left:4px">${unit}</span>`;
  }
  function fromClientX(clientX) {
    const r = track.getBoundingClientRect();
    let nv = min + clamp((clientX - r.left) / r.width, 0, 1) * (max - min);
    nv = +(Math.round(nv / step) * step).toFixed(2);
    const changed = nv !== v;
    v = nv; paint();
    if (changed) onInput(v);
  }
  let dragging = false;
  track.addEventListener('pointerdown', (e) => { e.preventDefault(); dragging = true; track.setPointerCapture(e.pointerId); fromClientX(e.clientX); });
  track.addEventListener('pointermove', (e) => { if (dragging) fromClientX(e.clientX); });
  track.addEventListener('pointerup', () => { dragging = false; });
  paint();
  return { el: wrap, setValue: (nv) => { v = nv; paint(); } };
}

export async function renderReactor(main, { state, api }) {
  main.innerHTML = '';

  // ---- header ----
  const head = ele('div', 'display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:22px');
  head.innerHTML = `
    <div>
      <div style="display:flex;align-items:center;gap:12px">
        <h1 style="margin:0;font-size:25px;font-weight:600;letter-spacing:0.3px;color:var(--text)">Reactor View</h1>
        <span class="live-pill"><span class="dot"></span>LIVE SOLVE</span>
      </div>
      <p style="margin:5px 0 0;font-size:13px;color:var(--text-dim)">Move an input — particle balance re-solves T<span style="font-size:9px">e</span>, power balance re-solves n<span style="font-size:9px">e</span>, and every output responds.</p>
    </div>`;
  const modeWrap = ele('div', 'display:flex;gap:8px');
  head.appendChild(modeWrap);
  main.appendChild(head);

  // ---- chamber + controls grid ----
  const grid = ele('div', 'display:grid;grid-template-columns:minmax(300px,1.12fr) minmax(280px,0.88fr);gap:20px;margin-bottom:22px');
  const chamber = createChamber({ minHeight: 432 });
  grid.appendChild(chamber.el);

  const controls = ele('div', 'background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:20px 20px 22px;display:flex;flex-direction:column;gap:22px');
  controls.appendChild(ele('span', 'font-size:11px;letter-spacing:1.8px;text-transform:uppercase;color:var(--text-dim);font-weight:600', 'Control Inputs'));

  const rfSlider = createSlider({
    label: 'RF Power', min: 50, max: 300, step: 5, value: state.rf_power_w,
    color: 'var(--purple)', glow: 'rgba(167,139,250,0.4)', unit: 'W',
    onInput: (v) => { state.rf_power_w = v; refresh(); },
  });
  const pSlider = createSlider({
    label: 'Chamber Pressure', min: 1, max: 20, step: 0.5, value: state.pressure_mtorr,
    color: 'var(--cyan)', glow: 'rgba(56,189,248,0.4)', unit: 'mTorr',
    onInput: (v) => { state.pressure_mtorr = v; refresh(); },
  });
  controls.appendChild(rfSlider.el);
  controls.appendChild(pSlider.el);
  controls.appendChild(ele('div', 'margin-top:auto;padding-top:15px;border-top:1px solid var(--border);display:flex;flex-direction:column;gap:6px',
    `<span style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--text-mute)">Model note</span>
     <span style="font-size:12px;line-height:1.5;color:var(--text-dim)">T<span style="font-size:9px">e</span> is set by pressure &amp; geometry alone (a real global-model prediction); n<span style="font-size:9px">e</span> scales with absorbed power. Sheath uses the labelled fallback estimate — no RF-voltage drive supplied.</span>`));
  grid.appendChild(controls);
  main.appendChild(grid);

  // ---- mode preset buttons ----
  function paintModes() {
    modeWrap.innerHTML = '';
    for (const m of MODES) {
      const on = Math.abs(state.rf_power_w - m.rf) < 1 && Math.abs(state.pressure_mtorr - m.p) < 0.6;
      const b = ele('button',
        `font-family:var(--mono);font-size:10px;letter-spacing:0.5px;padding:7px 11px;border-radius:8px;cursor:pointer;border:1px solid ${on ? 'var(--purple)' : 'var(--border)'};background:${on ? 'rgba(167,139,250,0.12)' : 'var(--panel)'};color:${on ? 'var(--purple-2)' : 'var(--text-dim)'}`,
        m.label);
      b.title = m.desc;
      b.onclick = () => {
        state.rf_power_w = m.rf; state.pressure_mtorr = m.p;
        rfSlider.setValue(m.rf); pSlider.setValue(m.p);
        refresh();
      };
      modeWrap.appendChild(b);
    }
  }

  // ---- twin outputs ----
  main.appendChild(ele('div', 'display:flex;align-items:center;gap:12px;margin:0 0 14px',
    `<span style="font-size:11px;letter-spacing:1.8px;text-transform:uppercase;color:var(--text-dim);font-weight:600">Twin Outputs</span>
     <span style="flex:1;height:1px;background:var(--border)"></span>
     <span style="font-family:var(--mono);font-size:10px;color:var(--text-mute)">10 solved fields</span>`));
  const gaugeGrid = ele('div', 'display:grid;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));gap:14px');
  const gauges = GAUGE_CFG.map((cfg) => { const g = createGauge(cfg); gaugeGrid.appendChild(g.el); return { cfg, g }; });
  main.appendChild(gaugeGrid);

  // ---- live refresh: fetch the solve, push values into chamber + gauges ----
  let pending = false, queued = false;
  async function refresh() {
    paintModes();
    if (pending) { queued = true; return; }
    pending = true;
    try {
      const sim = await api.simulate(state.rf_power_w, state.pressure_mtorr);
      chamber.setOutputs(sim);
      for (const { cfg, g } of gauges) g.update(sim[cfg.key]);
    } finally {
      pending = false;
      if (queued) { queued = false; refresh(); }
    }
  }

  paintModes();
  const sim0 = await api.simulate(state.rf_power_w, state.pressure_mtorr);
  chamber.setOutputs(sim0);
  for (const { cfg, g } of gauges) g.update(sim0[cfg.key]);
  chamber.start();

  return { cleanup: () => chamber.stop() };
}
