/* Reactor Control Room - app shell: sidebar nav, hash router, and the live
 * footer system strip. Holds the shared operating point (RF power / pressure) so
 * the Reactor View and AI Verdict pages read the same configuration, exactly like
 * the mockup. Page modules do the rendering; every number they show is fetched. */
import { api } from './api.js';
import { createMeter } from './components.js';
import { renderReactor } from './pages/reactor.js';
import { renderVerdict } from './pages/verdict.js';
import { renderValidation } from './pages/validation.js';
import { renderReplay } from './pages/replay.js';

// Shared operating point, persisted across page navigation (the Reactor View
// writes it; other pages read it). Defaults match the mockup's "Stable" preset.
export const state = { rf_power_w: 100, pressure_mtorr: 10 };

const PAGES = [
  { id: 'reactor', idx: '01', label: 'Reactor View', render: renderReactor },
  { id: 'verdict', idx: '02', label: 'AI Verdict', render: renderVerdict },
  { id: 'validation', idx: '03', label: 'Physics Validation', render: renderValidation },
  { id: 'replay', idx: '04', label: 'Session Replay', render: renderReplay },
];

const navEl = document.getElementById('nav');
const mainEl = document.getElementById('main');
let active = null; // { cleanup } returned by the current page

function renderNav(activeId) {
  navEl.innerHTML = '';
  for (const p of PAGES) {
    const btn = document.createElement('button');
    btn.className = 'nav-item' + (p.id === activeId ? ' active' : '');
    btn.innerHTML = `<span class="nav-idx">${p.idx}</span><span class="nav-label">${p.label}</span>`;
    btn.onclick = () => { location.hash = p.id; };
    navEl.appendChild(btn);
  }
}

async function route() {
  const id = location.hash.replace('#', '') || 'reactor';
  const page = PAGES.find((p) => p.id === id) || PAGES[0];
  if (active && typeof active.cleanup === 'function') active.cleanup();
  active = null;
  renderNav(page.id);
  mainEl.innerHTML = '<div class="loading"><span class="dot"></span> Solving…</div>';
  try {
    active = (await page.render(mainEl, { state, api })) || {};
  } catch (e) {
    mainEl.innerHTML = `<div class="err">Failed to load: ${e.message}<br>Is the backend running?</div>`;
  }
}

window.addEventListener('hashchange', route);
route();

// ---- footer system strip (live host metrics, polled from the backend) ----
const cpuMeter = createMeter({ gradient: 'linear-gradient(90deg,var(--cyan),var(--purple))' });
const ramMeter = createMeter({ gradient: 'linear-gradient(90deg,var(--green),var(--cyan))' });
document.getElementById('cpu-meter').appendChild(cpuMeter.el);
document.getElementById('ram-meter').appendChild(ramMeter.el);
const cpuVal = document.getElementById('cpu-val');
const ramVal = document.getElementById('ram-val');
const gpuNote = document.getElementById('gpu-note');

async function pollSystem() {
  try {
    const s = await api.systemStats();
    cpuMeter.update(s.cpu_percent);
    ramMeter.update(s.ram_percent);
    cpuVal.textContent = s.cpu_percent.toFixed(1) + '%';
    ramVal.textContent = s.ram_percent.toFixed(1) + '%';
    // GPU idle note comes straight from the backend, not hardcoded here.
    gpuNote.textContent = 'GPU — ' + s.gpu.status + ' · core models are CPU-based tree ensembles';
  } catch (_) { /* transient; next tick retries */ }
}
setInterval(pollSystem, 1500);
pollSystem();
