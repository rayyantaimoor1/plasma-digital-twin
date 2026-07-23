/* Component-library showcase. Renders every shared component with LIVE backend
 * data, proving the library works end-to-end before the pages are built. Every
 * value shown is fetched from the API; none is computed here. */
import { api } from './api.js';
import {
  createChamber, createGauge, createComparisonBar, createSparkline, createMeter,
} from './components.js';

const root = document.getElementById('demo');

function heading(text) {
  const h = document.createElement('h2');
  h.textContent = text;
  root.appendChild(h);
  return h;
}
function container(cls) {
  const d = document.createElement('div');
  d.className = cls;
  root.appendChild(d);
  return d;
}

// The 10 twin-output gauges, same scales the Reactor View uses.
const GAUGE_CFG = [
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

async function main() {
  // ---- Chamber + gauges (from /api/simulate) ----
  heading('Chamber + Gauges · /api/simulate (150 W · 10 mTorr)');
  const two = container('two');
  const sim = await api.simulate(150, 10);
  const chamber = createChamber({ minHeight: 400 });
  two.appendChild(chamber.el);
  chamber.setOutputs(sim);
  chamber.start();

  const gaugeGrid = document.createElement('div');
  gaugeGrid.className = 'grid';
  gaugeGrid.style.gridTemplateColumns = 'repeat(2, 1fr)';
  two.appendChild(gaugeGrid);
  for (const c of GAUGE_CFG.slice(0, 4)) {
    const g = createGauge(c);
    gaugeGrid.appendChild(g.el);
    g.update(sim[c.key]);
  }

  heading('Full gauge row · all 10 twin outputs');
  const allGauges = container('grid');
  for (const c of GAUGE_CFG) {
    const g = createGauge(c);
    allGauges.appendChild(g.el);
    g.update(sim[c.key]);
  }

  // ---- Comparison bars (from /api/physics-validation) ----
  heading('Comparison bars · /api/physics-validation');
  const bars = container('stack');
  const checks = await api.physicsValidation();
  for (const check of checks.slice(0, 3)) {
    const bar = createComparisonBar();
    bars.appendChild(bar.el);
    bar.update(check);
  }

  // ---- Sparkline (a metric swept across live backend operating points) ----
  heading('Sparkline · process quality across a live power sweep (/api/simulate ×6)');
  const sparks = container('stack');
  const powers = [50, 100, 150, 200, 250, 300];
  const sweep = await Promise.all(powers.map((P) => api.simulate(P, 10)));
  const spark = createSparkline({ label: 'Process Quality', unit: '0–1 · vs RF power', color: 'var(--purple)', decimals: 3 });
  sparks.appendChild(spark.el);
  spark.update({ values: sweep.map((s) => s.process_quality), currentIndex: 2 });

  // ---- Meters (from /api/system/stats) ----
  heading('Meters · /api/system/stats');
  const meters = container('stack');
  const stats = await api.systemStats();
  const row = (name, pct, gradient) => {
    const r = document.createElement('div');
    r.style.cssText = 'display:flex;align-items:center;gap:12px';
    r.innerHTML = `<span style="width:44px;font-size:11px;color:var(--text-dim);font-weight:600">${name}</span>`;
    const m = createMeter({ gradient });
    m.el.style.flex = '1';
    r.appendChild(m.el);
    const v = document.createElement('span');
    v.className = 'mono';
    v.style.cssText = 'width:48px;font-size:12px;color:var(--text)';
    v.textContent = pct.toFixed(1) + '%';
    r.appendChild(v);
    meters.appendChild(r);
    m.update(pct);
  };
  row('CPU', stats.cpu_percent, 'linear-gradient(90deg,var(--cyan),var(--purple))');
  row('RAM', stats.ram_percent, 'linear-gradient(90deg,var(--green),var(--cyan))');
}

main().catch((e) => {
  root.innerHTML = `<div class="err">Failed to load: ${e.message}<br>Is the backend running? (uvicorn reactor_control_room.backend.app:app)</div>`;
});
